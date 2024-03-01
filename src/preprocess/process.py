import argparse
from get_phonemes import get_phonemes
from get_ssl_features import get_ssl_features
from get_semantic import get_semantic

def main(log_dir="logs/", name="dolly"):
    params = {
        "input_txt_path": "../../data/dolly/dolly.txt",
        "save_path": f"{log_dir}/{name}",
        "input_wav_path": "../../data/dolly/vocal/"
    }
    get_phonemes(**params)
    get_ssl_features(**params)
    get_semantic(**params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="logs/", help="Directory to save logs")
    parser.add_argument("--name", default="dolly", help="Name of the logs")
    args = parser.parse_args()

    main(args.log_dir, args.name)