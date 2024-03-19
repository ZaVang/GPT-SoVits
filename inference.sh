#!/bin/bash

python src/inference/inference.py \
 --sovits_weights pretrained_models/sovits_weights/fulani_cn/fulani_cn_e25_s175.pth \
 --gpt_weights pretrained_models/gpt_weights/fulani_cn/fulani_cn-e25.ckpt \
 --parameters_file inference_parameters.txt \
 --output_folder output_audio/fulani_cn \

#  python src/inference/inference.py \
#  --sovits_weights pretrained_models/sovits_weights/dolly/dolly_e20_s140.pth \
#  --gpt_weights pretrained_models/gpt_weights/dolly/dolly-e25.ckpt \
#  --parameters_file inference_parameters.json \
#  --output_folder output_audio/dolly \