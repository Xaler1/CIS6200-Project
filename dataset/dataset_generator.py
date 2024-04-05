import json
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageOps
import pandas as pd
from sklearn.cluster import KMeans
from skimage import color as colorkit
import re
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
import os
import warnings

def get_closest_color(labels, colors, rgb):
    idx = np.argmin(np.linalg.norm(colors - rgb, axis=1))
    return labels[idx]

def main(args):
    model = SentenceTransformer('clip-ViT-L-14', device="cuda")

    split = args.split
    with open(f"data/{split}_formatted.json", "r") as f:
        data = json.load(f)

    categories = data["categories"]

    colors = pd.read_csv("final_data_colors.csv")
    basic_labels = colors["label"].tolist()
    basic_colors = colors[["red", "green", "blue"]].values

    colors = pd.read_csv("color_names.csv")
    fancy_labels = colors["Name"].tolist()
    fancy_colors = colors[["Red (8 bit)", "Green (8 bit)", "Blue (8 bit)"]].values
    fancy_labels = [re.sub(r'\(.*\)', '', label) for label in fancy_labels]

    results = {"images": {}}
    kmeans = KMeans(n_clusters=10, n_init="auto")


    for key in tqdm(data["images"]):
        image = data["images"][key]
        annotations = image["annotations"]
        viable_annotations = []
        seen_categories = set()
        for annotation in annotations:
            if annotation["area"] > 150 and annotation["category_id"] not in seen_categories:
                viable_annotations.append(annotation)
                seen_categories.add(annotation["category_id"])
        if len(viable_annotations) < args.min_masks:
            continue

        img = Image.open(f"data/{split}2017/{image['file_name']}")
        img_size = img.size
        if args.colorspace == "lab":
            img_color = colorkit.rgb2lab(img)
        elif args.colorspace == "yuv":
            img_color = colorkit.rgb2yuv(img)
        else:
            img_color = np.array(img)

        annotation_n = np.random.randint(args.min_masks, min(args.max_masks, len(viable_annotations) + 1))
        selected_annotations = np.random.choice(viable_annotations, annotation_n, replace=False)
        combined_mask = np.zeros(img_size).T
        mask_file = image["file_name"].replace(".jpg", ".png")
        embedding_file = image["file_name"].replace(".jpg", ".pt")
        results["images"][key] = {
            "img_file": image["file_name"],
            "mask_file": mask_file,
            "embedding_file": embedding_file,
            "labels": []
        }
        labels = []
        combined_basic_sentence = []
        combined_fancy_sentence = []
        for i, annotation in enumerate(selected_annotations):
            category = annotation["category_id"]
            name = categories[str(category)]["name"]
            name = name.replace("_", "")
            name = re.sub(r'\(.*\)', '', name)
            mask = Image.new("L", img_size, 0)
            polygon = np.array(annotation["segmentation"][0]).reshape(-1, 2)
            ImageDraw.Draw(mask).polygon(list(polygon.flatten()), outline=1, fill=1)
            mask = np.array(mask)
            combined_mask += mask * (i+1)
            pixels = np.array(img_color)[mask == 1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kmeans.fit(pixels)
            if args.color_select == "common":
                counts = np.bincount(kmeans.labels_)
                color = kmeans.cluster_centers_[np.argmax(counts)]
            elif args.color_select == "representative":
                color = np.mean(kmeans.cluster_centers_, axis=0)
            else:
                image_mean = np.mean(pixels, axis=0)
                distances = np.linalg.norm(kmeans.cluster_centers_ - image_mean, axis=1)
                color = kmeans.cluster_centers_[np.argmax(distances)]

            if args.colorspace == "lab":
                color = colorkit.lab2rgb([[color]])[0][0] * 255
            elif args.colorspace == "yuv":
                color = colorkit.yuv2rgb([[color]])[0][0] * 255

            basic_color = get_closest_color(basic_labels, basic_colors, color)
            fancy_color = get_closest_color(fancy_labels, fancy_colors, color)
            basic_sentence = f"A {name} colored {basic_color}"
            fancy_sentence = f"A {name} colored {fancy_color}"
            labels.append({
                "category": name,
                "basic_color": basic_color,
                "fancy_color": fancy_color,
                "fancy_sentence": fancy_sentence,
                "basic_sentence": basic_sentence
            })
            combined_basic_sentence.append(basic_sentence)
            combined_fancy_sentence.append(fancy_sentence)

        combined_basic_sentence = ". ".join(combined_basic_sentence)
        combined_basic_sentence += "."
        combined_fancy_sentence = ". ".join(combined_fancy_sentence)
        combined_fancy_sentence += "."
        # print("Basic Sentence: ", combined_basic_sentence)
        # print("Fancy Sentence: ", combined_fancy_sentence)

        results["images"][key]["labels"] = labels

        fancy_embedding = model.encode(combined_fancy_sentence)
        basic_embedding = model.encode(combined_basic_sentence)
        torch.save(fancy_embedding, f"data/processed_{split}/fancy_{embedding_file}")
        torch.save(basic_embedding, f"data/processed_{split}/basic_{embedding_file}")
        combined_mask = Image.fromarray(combined_mask)
        combined_mask = combined_mask.resize((args.hw, args.hw))
        combined_mask = combined_mask.convert("RGB")
        combined_mask.save(f"data/processed_{split}/{mask_file}")

        img = img.resize((args.hw, args.hw))
        grayscale = ImageOps.grayscale(img)

        # save the image
        img.save(f"data/processed_{split}/{image['file_name']}")
        grayscale.save(f"data/processed_{split}/{image['file_name'].replace('.jpg', '_gray.jpg')}")

    with open(f"data/processed_{split}.json", "w") as f:
        json.dump(results, f)





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--split", type=str, required=True, choices=["train", "val"])
    args.add_argument("--min_masks", type=int, default=5)
    args.add_argument("--max_masks", type=int, default=15)
    args.add_argument("--colorspace", type=str, default="lab", choices=["yuv", "lab", "rgb"])
    args.add_argument("--color_select", type=str, default="representative", choices=["common", "representative", "dominant"])
    args.add_argument("--color_type", type=str, default="basic", choices=["basic", "fancy"])
    args.add_argument("--hw", type=int, default=512, help="height and width of the image")
    args = args.parse_args()

    print(args)
    main(args)
