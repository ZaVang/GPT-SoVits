import os
import argparse
from get_phonemes import get_phonemes
from get_ssl_features import get_ssl_features
from get_semantic import get_semantic

def main(data_dir="../../data/", log_dir="logs/", name="dolly"):
    params = {
        "input_txt_path": os.path.join(data_dir, f"{name}/{name}.txt"),
        "save_path": f"{log_dir}/{name}",
        "input_wav_path": os.path.join(data_dir, f"{name}/vocal/")
    }
    get_phonemes(**params)
    get_ssl_features(**params)
    get_semantic(**params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../data/", help="Directory to save data")
    parser.add_argument("--log_dir", type=str, default="logs/", help="Directory to save logs")
    parser.add_argument("--name", type=str, default="dolly", help="Name of the logs")
    args = parser.parse_args()

    main(args.data_dir, args.log_dir, args.name)