#!/bin/bash

echo "Downloading one part of the dataset"

mkdir "data"
mkdir "output"
pip install tqdm
python downloader.py

echo "Download complete"
echo "Setting up SSA"

git clone https://github.com/fudan-zvg/Semantic-Segment-Anything.git
cd Semantic-Segment-Anything
conda env create -f environment.yml

echo "Part 1 complete, activate the ssa environment and run part 2"
