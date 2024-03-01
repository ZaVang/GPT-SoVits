import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traceback
import numpy as np
import librosa
from scipy.io import wavfile
import torch

from utils.utils import load_audio
from feature_extractor.cnhubert import CNHubert


MAXX = 0.95
ALPHA = 0.5

def name2go(wav_name: str, 
            wav_path: str,
            features_path: str,
            wav32k_path: str,
            model: CNHubert,
            device: str,
            is_half: bool=False):
    tmp_audio = load_audio(wav_path, 32000)
    tmp_max = np.abs(tmp_audio).max()
    if tmp_max > 2.2:
        print("%s-filtered,%s" % (wav_name, tmp_max))
        return
    tmp_audio32 = (tmp_audio / tmp_max * (MAXX * ALPHA*32768)) + ((1 - ALPHA)*32768) * tmp_audio
    tmp_audio32b = (tmp_audio / tmp_max * (MAXX * ALPHA*1145.14)) + ((1 - ALPHA)*1145.14) * tmp_audio
    tmp_audio = librosa.resample(
        tmp_audio32b, orig_sr=32000, target_sr=16000
    )#不是重采样问题
    tensor_wav16 = torch.from_numpy(tmp_audio)
    if (is_half == True):
        tensor_wav16=tensor_wav16.half().to(device)
    else:
        tensor_wav16 = tensor_wav16.to(device)
    ssl=model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1,2).cpu()#torch.Size([1, 768, 215])
    if np.isnan(ssl.detach().numpy()).sum()!= 0:
        print("nan filtered:%s"%wav_name)
        return wav_name
    wavfile.write(
        os.path.join(wav32k_path,wav_name),
        32000,
        tmp_audio32.astype("int16"),
    )
    torch.save(ssl, f"{features_path}/{wav_name}.pt")
    
    
def get_ssl_features(input_txt_path: str,
                     save_path: str,
                     input_wav_path: str=None,
                     cnhubert_path: str='pretrained_models/chinese-hubert-base',
                     is_half: bool=False,
                     **kwargs
                     ):
    wav32k_path = os.path.join(save_path, "wav32k")
    features_path = os.path.join(save_path, "cnhubert_features")
    os.makedirs(wav32k_path, exist_ok=True)
    os.makedirs(features_path, exist_ok=True)
    
    model = CNHubert(cnhubert_path)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if is_half:
        model = model.half()
    model = model.to(device)
    model.eval()
    
    with open(input_txt_path, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")
    
    nan_fails = []
    for line in lines:
        try:
            wav_name, _, _, _ = line.split("|")
            if input_wav_path:
                wav_name = os.path.basename(wav_name)
                wav_path = os.path.join(input_wav_path, wav_name)
            else:
                wav_path = wav_name
            if not os.path.exists(wav_path):
                print(f"{wav_path} does not exist")
                continue
            wav_name = name2go(wav_name, wav_path, features_path, wav32k_path, model, device, is_half)
            if wav_name:
                nan_fails.append(wav_name)
        except:
            print(line, traceback.format_exc())
    
    if(len(nan_fails)>0 and is_half==True):
        is_half=False
        model=model.float()
        for wav_name in nan_fails:
            try:
                name2go(wav_name, wav_path, features_path, wav32k_path, model, device, is_half)
            except:
                print(wav_name,traceback.format_exc())

    print('CnHubert特征提取已完成!')
    