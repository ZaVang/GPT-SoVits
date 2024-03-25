#!/bin/bash

python src/inference/inference.py \
 --sovits_weights pretrained_models/sovits_weights/varus/varus_e50_s650.pth \
 --gpt_weights pretrained_models/gpt_weights/varus/varus-e50.ckpt \
 --parameters_file inference_parameters.txt \
 --output_folder output_audio/varus \

#  python src/inference/inference.py \
#  --sovits_weights pretrained_models/sovits_weights/dolly/dolly_e20_s140.pth \
#  --gpt_weights pretrained_models/gpt_weights/dolly/dolly-e25.ckpt \
#  --parameters_file inference_parameters.json \
#  --output_folder output_audio/dolly \