# -*- coding: utf-8 -*-
"""Data Extraction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/183JrJ37f2BAEAtVg7TBMDed-k_RA_O30

Dataset: SA-1B - MetaAI
"""

import os
import tarfile
import cv2
import numpy as np
from pycocotools.coco import COCO
import openai

OPENAI_API_KEY = 'openai_api_key'
openai.api_key = OPENAI_API_KEY

def download_and_extract(url, target_path):
    response = requests.get(url)
    with open(target_path, 'wb') as file:
        file.write(response.content)
    with tarfile.open(target_path) as tar:
        tar.extractall(path=os.path.dirname(target_path))

def decode_mask(rle, height, width):
    return COCO.decode({'size': [height, width], 'counts': rle})

def generate_label(image_path, mask):
    image = cv2.imread(image_path)
    masked_image = apply_mask(image, mask)
    _, buffer = cv2.imencode('.jpg', masked_image)
    response = openai.Image.create(
        image=buffer.tobytes(),
        model="gpt-4",
        task="image-label",
    )
    return response['data']['text']

def apply_mask(image, mask):
    mask = mask.astype(bool)
    for i in range(3):
        image[:, :, i] = image[:, :, i] * mask
    return image

def process_dataset(dataset_path, annotations_path):
    coco = COCO(annotations_path)
    image_ids = coco.getImgIds()

    for image_id in image_ids:
        img = coco.loadImgs(image_id)[0]
        img_path = os.path.join(dataset_path, img['file_name'])
        ann_ids = coco.getAnnIds(imgIds=img['id'])
        annotations = coco.loadAnns(ann_ids)

        for ann in annotations:
            mask = decode_mask(ann['segmentation']['counts'], img['height'], img['width'])
            label = generate_label(img_path, mask)
            print(label)

DATA_URL = 'link'
download_and_extract(DATA_URL, 'sa_000000.tar')

process_dataset('path_to_extracted_dataset', 'path_to_annotations_file.json')