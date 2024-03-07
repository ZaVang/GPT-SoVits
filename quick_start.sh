#!/bin/bash

# # Preprocess
python src/preprocess/process.py \
 --data_dir ../../data \
 --log_dir logs \
 --name Freddie \

# train sovits model
python src/train/train_sovits.py \
 -c src/configs/sovits.json \
 -n Freddie \
 -t sovits \
 -e 20 \
 -lr 0.4 \
 -bs 19 \
 -nw 0 \
 --save_every_epoch 5 \
 --keep_ckpts 5 \

# train gpt mpdel
python src/train/train_gpt.py \
 -c src/configs/s1longer.yaml \
 -n Freddie \
 -e 25 \
 -bs 19 \
 --is_half \
 -nw 0 \
 --save_every_epoch 5


