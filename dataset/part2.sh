#!/bin/bash

echo "Installing and Running SSA"

python -m spacy download en_core_web_sm
python scripts/main_ssa_engine.py --data_dir=../data --out_dir=../output --world_size=1 --save_img --light_mode

