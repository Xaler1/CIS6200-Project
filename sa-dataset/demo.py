import streamlit as st
import json
import json
import os
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from pycocotools import mask as coco_mask
import cv2
from streamlit_image_comparison import image_comparison
from keys import openai_key
import requests
import base64

files = os.listdir("data")
state = st.session_state
headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {openai_key}"
}


def rle_to_bitmap(rle):
    bitmap = coco_mask.decode(rle)
    return bitmap

def get_cropped(id):
    with open(f"data/{files[id]}") as f:
        data = json.load(f)
    img = plt.imread(f"images/{data['image']['file_name']}")
    masks = np.zeros((data["image"]["height"], data["image"]["width"]))
    for id, annotation in enumerate(data["annotations"]):
        bitmap = rle_to_bitmap(annotation["segmentation"])
        masks[bitmap > 0] = id + 1

    # crop image to 1:1 ratio
    width = data["image"]["width"]
    height = data["image"]["height"]
    if width > height:
        diff = width - height
        img = img[:, diff // 2:width - diff // 2]
        cropped_masks = masks[:, diff // 2:width - diff // 2]
    else:
        diff = height - width
        img = img[diff // 2:height - diff // 2, :]
        cropped_masks = masks[diff // 2:height - diff // 2, :]

    # filter based on cropping
    for i, annotation in enumerate(data["annotations"]):
        old_area = np.sum(masks == i + 1) + 1e-6
        new_area = np.sum(cropped_masks == i + 1)
        if (new_area / old_area) < 0.7:
            cropped_masks[cropped_masks == i + 1] = 0

    # resize to 512x512
    img = cv2.resize(img, (512, 512))
    masks = cv2.resize(cropped_masks, (512, 512), interpolation=cv2.INTER_NEAREST)
    return img, masks, id

def get_filtered(id, threshold, top=8):
    img, masks, max_id = get_cropped(id)
    areas = {}
    total_area = 512 * 512
    unique, counts = np.unique(masks, return_counts=True)
    for id, count in zip(unique, counts):
        areas[id] = count / total_area
    areas.pop(0)

    to_remove = [id for id, area in areas.items() if area < threshold]
    for id in to_remove:
        masks[masks == id] = 0
        areas.pop(id)
        
    
    areas = dict(sorted(areas.items(), key=lambda item: item[1], reverse=True)[:top])
    
    # Remove masks that are not in the top N
    for id in np.unique(masks):
        if id not in areas:
            masks[masks == id] = 0
    
    return img, masks, areas

def encode_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")

def get_chatgpt_annotation(img, masks, combination, prompt, mask_inverted, low_res):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encoded_img = encode_image(img)
    if mask_inverted:
        encoded_masks = encode_image((masks == 0) * 255)
    else:
        encoded_masks = encode_image((masks > 0) * 255)
    content = [
        {
            "type": "text",
            "text": prompt
        }
    ]
    masked = None
    if combination == 0:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_img}"
            }
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_masks}"
            }
        })
    elif combination == 1:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_masks}"
            }
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_img}"
            }
        })
    elif combination == 2:
        masked = img * (masks[:, :, np.newaxis] > 0)
        encoded_masked = encode_image(masked)
        content.append({
            "type": "image_url",
            "image_url": {
            "url": f"data:image/jpeg;base64,{encoded_masked}"
          }
        })
        masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
    elif combination == 3:
        masked = img * (masks[:, :, np.newaxis] > 0)
        encoded_masked = encode_image(masked)
        content.append({
            "type": "text",
            "text": "The original image comes first"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_img}",
                "detail": "low"
            }
        })
        content.append({
            "type": "text",
            "text": "The masked image comes next"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_masked}",
                "detal": "low"
            }
        })
        masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
        pass
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return response.json()["choices"][0]["message"]["content"], masked


st.title("Mask prompt experiment")

default_prompt = """Describe what you see in the mask of the image.
Describe the colors of the objects.
An example output would be:
"The table is brown, the sky is blue and the banana is yellow" 
DO NOT say anything about the presence of a mask or the fact that the background is black. make it like a natural description. Make sure to describe ALL objects. 
Only use the original image for context, do not include it in the output, no matter how significant.
Stay concise, keep the response under 70 words.
"""

# Area for checkboxes and settings
with st.container(border=True):
    st.header("Settings")
    id = st.number_input("Image id", min_value=0, max_value=len(files) - 1, value=0, step=1)
    threshold = st.number_input("Mask area threshold", min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.3f")
    top = st.number_input("Top N masks", min_value=1, max_value=20, value=8, step=1)
    types = ["image followed by mask", "mask followed by image", "masked image", "original followed by masked"]
    chat_input_type = st.radio("ChatGPT image combnation type", types, index=0)
    mask_inverted = st.checkbox("Invert the mask (will not affect the 'masked image' option)")
    low_res = st.checkbox("Low resolution mode on OpenAI API")
    prompt = st.text_area("ChatGPT prompt", default_prompt, height=200)

chat_input_type = types.index(chat_input_type)

generate = st.button("Generate")
if generate:
    with st.status("Processing"):
        st.write("Resizing and filtering masks")
        img, masks, areas = get_filtered(id, threshold, top)
        plt.imsave("img.jpg", img)
        plt.imsave("masks.jpg", masks)
        st.write("Querying chatgpt")
        response, masked = get_chatgpt_annotation(img, masks, chat_input_type, prompt, mask_inverted, low_res)
    if chat_input_type == 2:
        plt.imsave("masked.jpg", masked)
        st.image("masked.jpg")
    if chat_input_type == 3:
        plt.imsave("masked.jpg", masked)
        image_comparison("img.jpg", "masked.jpg", "", "")
    else:
        image_comparison("img.jpg", "masks.jpg", "", "")
    st.write(response)

