#!/bin/bash
conda install -c conda-forge gcc
conda install -c conda-forge gxx
conda install ffmpeg cmake
conda install pytorch==2.1.0 torchvision==0.20.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
