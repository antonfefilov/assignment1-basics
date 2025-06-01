#!/bin/bash

# git clone https://github.com/antonfefilov/assignment1-basics.git
# cd assignment1-basics
pip install uv
uv sync
mkdir -p data
cd data
wget https://github.com/antonfefilov/assignment1-basics/releases/download/v1.0.0/TinyStoriesV2-GPT4-train.txt
wget https://github.com/antonfefilov/assignment1-basics/releases/download/v1.0.0/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
cd ..
tmux new-session -d -s "train"
tmux send-keys "source .venv/bin/activate" C-m
tmux send-keys "python train_memory_efficient.py --input data/owt_train.txt --processes 16 --vocab-size 32000" C-m
tmux attach-session -t "train"

# source .venv/bin/activate
# ./train.sh -i data/TinyStoriesV2-GPT4-train.txt -p 4 -v 10000