#!/bin/bash

# git clone https://github.com/antonfefilov/assignment1-basics.git
cd assignment1-basics
pip install uv
uv sync
source .venv/bin/activate
mkdir -p data
cd data
wget https://github.com/antonfefilov/assignment1-basics/releases/download/v1.0.0/TinyStoriesV2-GPT4-train.txt
wget https://github.com/antonfefilov/assignment1-basics/releases/download/v1.0.0/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
cd ..
# ./train.sh -i data/TinyStoriesV2-GPT4-train.txt -p 4 -v 10000