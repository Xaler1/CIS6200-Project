import json
import os
from tqdm import tqdm, trange
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from pycocotools import mask as coco_mask
import cv2
from streamlit_image_comparison import image_comparison
from keys import openai_key
import requests
import base64
import os
from multiprocessing import Pool

files = os.listdir("data")
headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {openai_key}"
}

prompt = """Describe what you see in the mask of the image.
Describe the colors of the objects.
An example output would be:
"The table is brown, the sky is blue and the banana is yellow" 
DO NOT say anything about the presence of a mask or the fact that the background is black. make it like a natural description. Make sure to describe ALL objects. 
Only use the original image for context, do not include it in the output, no matter how significant.
Keep the response under 70 words, only describe colors."""


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

def get_chatgpt_body(img, masks, prompt):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encoded_img = encode_image(img)
    content = [
        {
            "type": "text",
            "text": prompt
        }
    ]
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
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 70
    }
    return payload


def get_chatgpt_annotation(img, masks, combination, prompt, mask_inverted):
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
    if combination == 0:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_img}",
                "detail": "low"
            }
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_masks}",
                "detail": "low"
            }
        })
    elif combination == 1:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_masks}",
                "detail": "low"
            }
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_img}",
                "detail": "low"
            }
        })
    elif combination == 2:
        masked = img * (masks[:, :, np.newaxis] > 0)
        encoded_masked = encode_image(masked)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_masked}",
                "detail": "low"
          }
        })
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
        ],
        "max_tokens": 70
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

def process_image(i):
    try:
        destination = "test_split/"
        img, masks, areas = get_filtered(i, 0.01, 8)
        if len(areas) < 2:
            return
        plt.imsave(f"{destination}/{i}.jpg", img)
        bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        plt.imsave(f"{destination}/{i}_gray.jpg", bw, cmap="gray")
        plt.imsave(f"{destination}/{i}_mask.jpg", masks, cmap="gray")
        annotation = get_chatgpt_annotation(img, masks, 3, prompt, False)
        with open(f"{destination}/{i}.txt", "w", encoding="utf-8") as f:
            f.write(annotation)
    except Exception as e:
        return

def main():
    destination = "test_split/"
    os.makedirs(destination, exist_ok=True)
    n = 1000
    with Pool(10) as p:
        for _ in tqdm(p.imap(process_image, range(n)), total=n):
            pass

def process_batch(i):
    destination = "train_split/"
    img, masks, areas = get_filtered(i, 0.01, 8)
    if len(areas) < 2:
        return
    plt.imsave(f"{destination}/{i}.jpg", img)
    bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.imsave(f"{destination}/{i}_gray.jpg", bw, cmap="gray")
    plt.imsave(f"{destination}/{i}_mask.jpg", masks, cmap="gray")
    payload = get_chatgpt_body(img, masks, prompt)
    line = {
        "custom_id": f"request-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": payload
    }
    return line


def batch():
    destination = "train_split/"
    os.makedirs(destination, exist_ok=True)
    start = 1000
    n = 5000
    with Pool(16) as p:
        for line in tqdm(p.imap(process_batch, range(start, start + n)), total=n):
            if line:
                with open(f"train_job.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(line) + "\n")



if __name__ == "__main__":
    batch()