import os
import argparse
import soundfile as sf

from infer_tool import TTSInference


def generate_audio(sovits_weights, 
                   gpt_weights, 
                   input_folder,
                   output_folder, 
                   ref_wav_path, 
                   prompt_text, 
                   prompt_language, 
                   text, 
                   text_language,
                   how_to_cut=None
                   ):
    tts = TTSInference(sovits_weights=sovits_weights, gpt_weights=gpt_weights)
    
    infer_dict = {
        'ref_wav_path': os.path.join(input_folder, ref_wav_path),
        'prompt_text': prompt_text,
        'prompt_language': prompt_language,
        'text': text,
        'text_language': text_language,
        'how_to_cut': how_to_cut
    }

    audio_generator = tts.infer(**infer_dict)
    
    sr, audio = next(audio_generator)
    output_path = os.path.join(output_folder, f"{text[:6]}_output.wav")
    sf.write(output_path, audio, sr)
    print(f"Audio saved to {output_path}")
    

def process_batch(sovits_weights, gpt_weights, input_folder, output_folder, parameters_file):
    with open(parameters_file, 'r', encoding='utf-8') as file:
        for line in file:
            ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut = line.strip().split('|')
            generate_audio(sovits_weights, 
                           gpt_weights, 
                           input_folder, 
                           output_folder, 
                           ref_wav_path, 
                           prompt_text, 
                           prompt_language, 
                           text, 
                           text_language, 
                           how_to_cut
                           )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Run TTS Inference")
    parser.add_argument("--sovits_weights", required=True, help="Path to sovits weights file")
    parser.add_argument("--gpt_weights", required=True, help="Path to gpt weights file")
    parser.add_argument("--input_folder", type=str, default='input_audio', help="Folder of the input audio files")
    parser.add_argument("--output_folder", type=str, default='output_audio', help="Folder to save the output audio files")
    parser.add_argument("--parameters_file", type=str, default='inference_parameters.txt', help="File containing parameters for batch processing")

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    process_batch(args.sovits_weights, 
                  args.gpt_weights, 
                  args.input_folder, 
                  args.output_folder, 
                  args.parameters_file)
