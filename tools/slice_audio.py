import os,sys,numpy as np
import traceback
# parent_directory = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(parent_directory)
from my_utils import load_audio
from slicer2 import Slicer
import os.path
from argparse import ArgumentParser
import soundfile
import librosa

def slice(input_path: str,
          output_path: str,
          db_threshold: int=-40,
          min_length: int=5000,
          min_interval: int=300,
          hop_size: int=10,
          max_sil_kept: int=500,
          max_amp: float=1.0,
          alpha: float=0):
    
    os.makedirs(output_path, exist_ok=True)
    if os.path.isfile(input_path):
        input_files=[input_path]
    elif os.path.isdir(input_path):
        input_files=[os.path.join(input_path, name) for name in sorted(list(os.listdir(input_path)))]
    else:
        return "输入路径存在但既不是文件也不是文件夹"
    
    max_amp=float(max_amp)
    alpha=float(alpha)
    for file in input_files:
        try:
            audio, sr = librosa.load(file, sr=None, mono=False)
            # print(audio.shape)
            
            slicer = Slicer(
                sr=sr,  # 长音频采样率
                threshold=      int(db_threshold),  # 音量小于这个值视作静音的备选切割点
                min_length=     int(min_length),  # 每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值
                min_interval=   int(min_interval),  # 最短切割间隔
                hop_size=       int(hop_size),  # 怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）
                max_sil_kept=   int(max_sil_kept),  # 切完后静音最多留多长
            )
            
            chunks = slicer.slice(audio)
            for i, (chunk, start, end) in enumerate(chunks):
                if len(chunk.shape) > 1:
                    chunk = chunk.T
                tmp_max = np.abs(chunk).max()
                if(tmp_max>1):chunk/=tmp_max
                chunk = (chunk / tmp_max * (max_amp * alpha)) + (1 - alpha) * chunk
                soundfile.write(
                    os.path.join(
                        output_path,
                        f"%s_%d.wav"
                        % (os.path.basename(file).rsplit(".", maxsplit=1)[0], i),
                    ),
                    chunk,
                    sr,
                )
        except:
            print(file,"->fail->",traceback.format_exc())
    return "执行完毕，请检查输出文件"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path", 
        type=str, 
        help="The audios to be sliced, can be a single file or a directory"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        help="Output directory of the sliced audio clips"
    )
    parser.add_argument(
        "--db_threshold",
        type=float,
        required=False,
        default=-40,
        help="The dB threshold for silence detection",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        required=False,
        default=5000,
        help="The minimum milliseconds required for each sliced audio clip",
    )
    parser.add_argument(
        "--min_interval",
        type=int,
        required=False,
        default=300,
        help="The minimum milliseconds for a silence part to be sliced",
    )
    parser.add_argument(
        "--hop_size",
        type=int,
        required=False,
        default=10,
        help="Frame length in milliseconds",
    )
    parser.add_argument(
        "--max_sil_kept",
        type=int,
        required=False,
        default=500,
        help="The maximum silence length kept around the sliced clip, presented in milliseconds",
    )
    parser.add_argument(
        "--max_amp",
        type=float,
        required=False,
        default=1.0,
        help="The maximum amplitude of the sliced audio clips",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=False,
        default=0,
        help="The alpha value for amplitude adjustment",
    )
    args = parser.parse_args()
    
    slice(**args.__dict__)
