#!/bin/bash

python src/inference/inference.py \
 --sovits_weights pretrained_models/sovits_weights/kuidou_cn/kuidou_cn_e30_s540.pth \
 --gpt_weights pretrained_models/gpt_weights/kuidou_cn/kuidou_cn-e30.ckpt \
 --parameters_file inference_parameters.json \
 --output_folder output_audio/kuidou_cn \