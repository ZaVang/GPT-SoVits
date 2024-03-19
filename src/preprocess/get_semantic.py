import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traceback
import torch
from module.models import SynthesizerTrn
from utils.utils import get_hparams_from_file


def name2go(wav_name: str, 
            cnhubert_features_path: str,
            model: SynthesizerTrn,
            device: str,
            is_half: bool=False):
    """
    根据音频文件名提取语义特征。

    参数:
    wav_name (str): 音频文件的名称。
    cnhubert_features_path (str): 存储cnhubert特征文件的路径。
    model (SynthesizerTrn): 用于提取语义特征的模型实例。
    device (str): 指定运行模型的设备（如'cpu'或'cuda:0'）。
    is_half (bool, 可选): 如果为True，则使用半精度浮点数处理数据以加快计算速度，默认为False。

    返回:
    str: 音频文件名和其对应的语义特征，使用制表符（'\t'）分隔。
    """
    if not os.path.exists(cnhubert_features_path):
        return
    ssl_content = torch.load(f"{cnhubert_features_path}/{wav_name}.pt", map_location="cpu")
    if is_half:
        ssl_content = ssl_content.half().to(device)
    else:
        ssl_content = ssl_content.to(device)    
    codes = model.extract_latent(ssl_content)
    semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
    return f"{wav_name}\t{semantic}"


def get_semantic(input_txt_path: str,
                save_path: str,
                G_path: str='pretrained_models/sovits_weights/pretrained/s2G488k.pth',
                config_path: str='src/configs/sovits.json',
                is_half: bool=False,
                **kwargs
                ):
    """
    从文本文件中读取音频文件名，并提取相应的语义特征。

    参数:
    input_txt_path (str): 包含音频文件名的输入文本文件路径。
    save_path (str): 保存结果的路径。
    G_path (str, 可选): 指向预训练模型权重的路径，默认为'sovits_weights'下的路径。
    config_path (str, 可选): 配置文件的路径，默认为'src/configs'下的sovits.json。
    is_half (bool, 可选): 是否使用半精度计算，默认为False。

    使用kwargs接受任何额外的参数，以便未来的扩展。
    """
    hubert_features_path = os.path.join(save_path, "cnhubert_features")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    hps = get_hparams_from_file(config_path)
    model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )    
    # 加载模型权重
    print(
        model.load_state_dict(
            torch.load(G_path, map_location=device)["weight"], strict=False
        )
    )
    
    if is_half:
        model = model.half()
    model = model.to(device)
    model.eval() 
    
    with open(input_txt_path, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")
        
    semantic_results = []
    for line in lines:
        try:
            wav_name, _, _, _ = line.split("|")
            wav_name = os.path.basename(wav_name)
            semantic = name2go(wav_name, hubert_features_path, model, device, is_half)
            semantic_results.append(semantic)
        except Exception as e:
            # 输出错误信息
            print(line, str(e))
    
    with open(f"{save_path}/name2semantic.tsv", "w", encoding="utf8") as f:
        f.write("\n".join(semantic_results))
        
    print('语义特征提取完成!')