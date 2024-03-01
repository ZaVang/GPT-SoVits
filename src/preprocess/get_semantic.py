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
    if os.path.exists(cnhubert_features_path) == False:
        return
    ssl_content = torch.load(f"{cnhubert_features_path}/{wav_name}.pt", map_location="cpu")
    if is_half == True:
        ssl_content = ssl_content.half().to(device)
    else:
        ssl_content = ssl_content.to(device)
    codes = model.extract_latent(ssl_content)
    semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
    return f"{wav_name}\t{semantic}"

def get_semantic(input_txt_path: str,
                save_path: str,
                G_path: str='pretrained_models/s2G488k.pth',
                config_path: str='src/configs/sovits.json',
                is_half: bool=False,
                **kwargs
                ):
    hubert_features_path = os.path.join(save_path, "cnhubert_features")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    hps = get_hparams_from_file(config_path)
    model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
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
        except:
            print(line, traceback.format_exc())
    
    with open(f"{save_path}/name2semantic.tsv", "w", encoding="utf8") as f:
        f.write("\n".join(semantic_results))
        
    print('语义特征提取完成!')