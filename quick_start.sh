#!/bin/bash

# # Preprocess
python src/preprocess/process.py \
 --data_dir ../../data \
 --log_dir logs \
 --name varus \

# train sovits model
python src/train/train_sovits.py \
 -c src/configs/sovits.json \
 -n varus \
 -t sovits \
 -e 50 \
 -lr 0.4 \
 -bs 16 \
 -nw 0 \
 --save_every_epoch 10 \
 --keep_ckpts 5 \

# train gpt mpdel
python src/train/train_gpt.py \
 -c src/configs/s1longer.yaml \
 -n varus \
 -e 50 \
 -bs 16 \
 -nw 0 \
 --save_every_epoch 10


