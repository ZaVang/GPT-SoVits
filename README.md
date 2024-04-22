# GPT-SoVITS

[English](README.md) | [简体中文](docs/README_zh.md)

This project is a fork of the original [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) project, with several adjustments and enhancements aimed at improving clarity, functionality, and ease of use.

## Features

- **Refined Structure**: Adjustments have been made to the project's structure for enhanced clarity.
- **Command-Line Interface**: Added functionality to allow direct invocation via the command line for both training and inference.
- **Redesigned WebUI and API**: The implementations of the web interface and API have been completely rewritten for better performance and usability.
- **Bash Scripts**: Includes additional bash scripts for one-click training and batch inference, streamlining the process of using the project.

## Installation

To set up your environment, please follow these steps:

### Environment Setup

First, create a new conda environment and activate it:

```bash
conda create -n GPTSoVits python=3.10.13
conda activate GPTSoVits
```

### Installation Script

Run the provided installation script to install the necessary dependencies:

```bash
bash install.sh
```

## Data Preparation

Your dataset should be organized according to the following structure:

```
data_dir/
└── name/
    ├── name.txt  # Text annotations
    └── vocal/
        └── xxx.wav  # Example audio file
```

- `name.txt` contains the text annotations for TTS (Text-to-Speech). The format for each line in this file should be:

  ```
  vocal_path|speaker_name|language|text
  ```
- `vocal_path`: The relative path to the audio file within the `vocal` folder.
- `speaker_name`: The name of the speaker.
- `language`: The language code for the text. Supported languages are:

  - `'zh'`: Chinese
  - `'ja'`: Japanese
  - `'en'`: English
- `text`: The actual text that corresponds to the audio.
- `vocal/`: This folder contains the preprocessed WAV audio files, such as `xxx.wav`. Make sure that the audio files have been preprocessed according to the guidelines mentioned in the original project's README.

#### New Command Line Operation for Audio Splitting
Focus primarily on the first two parameters: `input_path` is the path to the original audio file or directory, and `output_path` is the directory where the subdivided audio clips will be saved. If the original audio is too quiet, you can increase the `alpha` parameter, which ranges from 0 to 1, with higher values making the sound louder.

## Model Storage

All models used by the project are stored in the `pretrained_models` directory. BERT and HuBERT models should be placed directly under `pretrained_models`.

The `gpt_weights` and `sovits_weights` directories contain models for GPT and SoVITS, respectively.

Within each of these directories, there is a folder named after the `name` parameter used during training, where models are copied upon training completion.

Pretrained models for GPT and SoVITS are located in their respective `pretrained` subdirectories, e.g., `pretrained_models/gpt_weights/pretrained`.

## Training

To start training with a single command, use the provided `quick_start.sh` script:

```bash
bash quick_start.sh
```

You may need to modify the `data_dir` and `name` parameters in the script according to your dataset's location and structure.

## Inference

For inference, both a WebUI and the command line are supported, providing flexibility in how you generate outputs from your models.

### WebUI

To use the WebUI, simply start the web server with the following command:

```bash
python server/webui.py
```

### Command-Line

To inference from a Command-Line, you can:

1. Using a Bash Script:

   ```bash
   bash inference.sh
   ```
2. Using Python Directly:

   ```bash
   python src/inference/inference.py \
    --sovits_weights your_sovits_model_path \
    --gpt_weights your_gpt_model_path \
    --parameters_file inference_parameters.json
   ```

In the command above, you can adjust the paths to the SoVITS and GPT model weights (`--sovits_weights` and `--gpt_weights`, respectively), depending on where you have stored your pretrained models.

The `--parameters_file` argument allows you to specify a JSON or a TXT file containing inference parameters, enabling batch processing.

- **TXT Format**: Each line represents a single inference request, formatted as follows:

  `Reference Audio | Prompt Text | Prompt Language | Target Text | Target Text Language | How to Cut`
- **JSON Format**: Each item in the JSON file represents a single inference request. Here's an example structure for a JSON entry:

  ```json
  {
    "ref_wav_path": "your_ref_audio_path",
    "prompt_text": "prompt_text",
    "prompt_language": "prompt_text_language",
    "text": "target_text",
    "text_language": "target_text_language",
    "how_to_cut": null
  }
  ```

This flexible approach allows you to tailor the inference process to your specific needs, whether you're processing a single request or running batch operations.

## API Usage

To start the API service, run:

```
python server/app.py
```

You can call the API as follows:

```python
import requests
import json

url = 'http://localhost:8888/ai-speech/api/tts/inference'
data = {
    "ref_audio_path": "input_audio/test.wav",
    "sovits_weights": "your_sovits_model_path",
    "gpt_weights": "your_gpt_model_path",
    "prompt_text": "...",
    "prompt_language": "中文",
    "text": "...",
    "text_language": "中文",
    "how_to_cut": "不切",
    "top_k": 5,
    "top_p": 0.7,
    "temperature": 0.7,
    "ref_free": False
}
headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json.dumps(data), headers=headers)

if response.status_code == 200:
    with open('output_audio/test.wav', 'wb') as f:
        f.write(response.content)
```

At the same time, this API also hosts a WebUI similar to the one described above, which can be accessed through:

```plaintext
http://localhost:8888/ai-speech/api/gradio
```

# Acknowledgements

This project builds upon the work of GPT-SoVITS, originally developed by RVC-Boss. We extend our deepest gratitude to RVC-Boss and all contributors to the GPT-SoVITS project for their pioneering work in the field and for making their code available to the community under the MIT License. This fork aims to explore further enhancements and applications of the original project, and we hope it contributes positively to the community.

Please visit the original project at: https://github.com/RVC-Boss/GPT-SoVITS
