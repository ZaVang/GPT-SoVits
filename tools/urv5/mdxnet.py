import os
import sys
import argparse
import glob
import soundfile as sf
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def parse_args():
    parser = argparse.ArgumentParser(description="UVR5 Audio Separation CLI")
    parser.add_argument("--input_root", type=str, required=True, help="Path to the input audio directory")
    parser.add_argument("--output_vocal", type=str, required=True, help="Path to the output directory for separated vocals")
    parser.add_argument("--output_inst", type=str, default=None, help="Path to the output directory for separated instruments (optional)")
    parser.add_argument("--onnx_path", type=str, default="pretrained_models/uvr5/vocals.onnx", help="Path to the ONNX model file")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for inference")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.input_root):
        print(f"Error: Input root directory {args.input_root} does not exist.")
        return

    os.makedirs(args.output_vocal, exist_ok=True)
    if args.output_inst:
        os.makedirs(args.output_inst, exist_ok=True)

    print(f"Loading ONNX model from {args.onnx_path}...")
    try:
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider'] if args.device == 'cuda' else ['CPUExecutionProvider']
        sess = ort.InferenceSession(args.onnx_path, providers=providers)
        print("ONNX load successful.")
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")
        return

    wav_files = glob.glob(os.path.join(args.input_root, "*.wav"))
    if not wav_files:
        print("No WAV files found in the input directory.")
        return

    print(f"Found {len(wav_files)} audio file(s). Starting processing...")
    for i, wav_path in enumerate(wav_files, 1):
        try:
            print(f"[{i}/{len(wav_files)}] Processing: {os.path.basename(wav_path)}")
            audio, sr = sf.read(wav_path)
            # Placeholder for actual UVR5 inference logic
            # In a real scenario, this would call the UVR5 processing pipeline
            vocal_audio = audio
            output_vocal_path = os.path.join(args.output_vocal, os.path.basename(wav_path))
            sf.write(output_vocal_path, vocal_audio, sr)

            if args.output_inst:
                output_inst_path = os.path.join(args.output_inst, os.path.basename(wav_path))
                sf.write(output_inst_path, np.zeros_like(audio), sr)
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")

    print("Processing completed.")

if __name__ == "__main__":
    main()
