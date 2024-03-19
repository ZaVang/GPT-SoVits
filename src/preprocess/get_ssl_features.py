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
    """
    对给定的音频文件进行处理和特征提取。

    参数:
    wav_name (str): 音频文件的名称。
    wav_path (str): 音频文件的完整路径。
    features_path (str): 用于保存提取出的特征文件的路径。
    wav32k_path (str): 用于保存处理后的32k采样率音频文件的路径。
    model (CNHubert): 预训练的CNHubert模型。
    device (str): 指定运行模型的设备，如'cuda:0'或'cpu'。
    is_half (bool): 是否将模型和数据转为半精度浮点数以节省内存，默认为False。

    返回:
    None或str: 如果处理中出现问题，返回音频文件的名称；否则不返回任何内容。
    """
    # 加载音频并调整为32k采样率
    tmp_audio = load_audio(wav_path, 32000)
    tmp_max = np.abs(tmp_audio).max()
    # 如果音频的最大绝对值大于2.2，则打印信息并返回
    if tmp_max > 2.2:
        print("%s-filtered,%s" % (wav_name, tmp_max))
        return
    # 对音频数据进行归一化和重采样处理
    tmp_audio32 = (tmp_audio / tmp_max * (MAXX * ALPHA * 32768)) + ((1 - ALPHA) * 32768) * tmp_audio
    tmp_audio32b = (tmp_audio / tmp_max * (MAXX * ALPHA * 1145.14)) + ((1 - ALPHA) * 1145.14) * tmp_audio
    tmp_audio = librosa.resample(tmp_audio32b, orig_sr=32000, target_sr=16000)  # 重采样到16k采样率

    tensor_wav16 = torch.from_numpy(tmp_audio)
    if is_half:
        tensor_wav16 = tensor_wav16.half().to(device)
    else:
        tensor_wav16 = tensor_wav16.to(device)
    # 使用CNHubert模型提取特征
    ssl = model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1, 2).cpu()
    # 检查提取的特征是否包含NaN值
    if np.isnan(ssl.detach().numpy()).sum() != 0:
        print("nan filtered:%s" % wav_name)
        return wav_name
    # 保存处理后的音频和特征文件
    wavfile.write(os.path.join(wav32k_path, wav_name), 32000, tmp_audio32.astype("int16"))
    torch.save(ssl, f"{features_path}/{wav_name}.pt")
    
    
def get_ssl_features(input_txt_path: str,
                     save_path: str,
                     input_wav_path: str = None,
                     cnhubert_path: str = 'pretrained_models/chinese-hubert-base',
                     is_half: bool = False,
                     **kwargs):
    """
    从文本文件中读取音频文件列表，对每个音频文件进行处理并提取特征。

    参数:
    input_txt_path (str): 包含音频文件信息的文本文件路径。
    save_path (str): 保存处理结果和特征文件的根目录。
    input_wav_path (str): 音频文件的输入目录。如果不为None，则会从这个目录读取音频文件，默认为None。
    cnhubert_path (str): CNHubert模型的路径，默认为预训练模型的路径。
    is_half (bool): 是否将模型和数据转为半精度浮点数以节省内存，默认为False。

    返回:
    None
    """
    # 创建保存32k采样率音频和特征文件的目录
    wav32k_path = os.path.join(save_path, "wav32k")
    features_path = os.path.join(save_path, "cnhubert_features")
    os.makedirs(wav32k_path, exist_ok=True)
    os.makedirs(features_path, exist_ok=True)
    
    # 加载模型并设置运行设备
    model = CNHubert(cnhubert_path)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if is_half:
        model = model.half()
    model = model.to(device)
    model.eval()
    
    # 读取音频文件列表并进行处理
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
    
    # 如果有处理失败的文件，尝试不使用半精度重新处理
    if len(nan_fails) > 0 and is_half:
        is_half = False
        model = model.float()
        for wav_name in nan_fails:
            try:
                name2go(wav_name, wav_path, features_path, wav32k_path, model, device, is_half)
            except:
                print(wav_name, traceback.format_exc())

    print('CnHubert特征提取已完成!')
