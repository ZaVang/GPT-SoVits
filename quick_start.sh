#!/bin/bash

# 获取传入的name参数
NAME=$1

# # Preprocess
python src/preprocess/process.py \
 --data_dir data \
 --log_dir logs \
 --name $NAME \

# train sovits model
python src/train/train_sovits.py \
 -c src/configs/sovits.json \
 -n $NAME \
 -t sovits \
 -e 30 \
 -lr 0.4 \
 -bs 32 \
 -nw 0 \
 --save_every_epoch 10 \
 --keep_ckpts 10

# train gpt model
python src/train/train_gpt.py \
 -c src/configs/s1longer.yaml \
 -n $NAME \
 -e 30 \
 -bs 32 \
 -nw 0 \
 --save_every_epoch 10


