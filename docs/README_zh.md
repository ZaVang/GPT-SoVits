# GPT-SoVITS

[英文](README.md) | [简体中文](docs/README_zh.md)

此项目是原始[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)项目的一个分支，进行了若干调整和优化，旨在提高清晰度、功能性和易用性。

## 特性

- **优化结构**：对项目结构进行了调整，以增强清晰度。
- **命令行界面**：增加了直接通过命令行调用训练和推理的功能。
- **重新设计的WebUI和API**：完全重写了Web界面和API的实现，以提高性能和可用性。
- **Bash脚本**：包含了额外的bash脚本，用于一键训练和批量推理，简化了使用项目的过程。

## 安装

请按照以下步骤设置您的环境：

### 环境设置

首先，创建一个新的conda环境并激活它：

```bash
conda create -n GPTSoVits python=3.10.13
conda activate GPTSoVits
```

### 安装脚本

运行提供的安装脚本以安装必要的依赖项：

```bash
bash install.sh
```

## 数据准备

数据集应该按照以下结构存放：

```
data_dir/
└── name/
    ├── name.txt  # 文本注释
    └── vocal/
        └── xxx.wav  # 示例音频文件
```

- `name.txt` 所有音频文件所对应的文本注释。该文件中每一行的格式应为：

  ```
  vocal_path|speaker_name|language|text
  ```
- `vocal_path`：音频文件在 `vocal`文件夹内的相对路径。
- `speaker_name`：说话者的名字（其实只要和上面的name对应一样就行）。
- `language`：文本的语言代码。支持的语言有：

  - `'zh'`：中文
  - `'ja'`：日文
  - `'en'`：英文
- `text`：对应音频的实际文本。
- `vocal/`：此文件夹包含所有预处理过的WAV音频文件，如 `xxx.wav`。请确保音频文件已根据原始项目进行了预处理，详情参考原项目的 `(Optional) If you need, here will provide the command line operation mode` 章节。

#### 新增切分音频的命令行操作
```
python slice_audio.py \
    --input_path path_to_original_audio_file_or_directory \
    --output_path directory_where_subdivided_audio_clips_will_be_saved \
    --db_threshold The dB threshold for silence detection \
    --min_length minimum_duration_of_each_subclip \
    --min_interval shortest_time_gap_between_adjacent_subclips \ 
    --hop_size step_size_for_computing_volume_curve \
    --max_sil_kept maximum silence length kept around the sliced clip \
    --max_amp maximum amplitude of the sliced audio clips \
    --alpha alpha value for amplitude adjustment
```
主要看前两个参数就好，`input_path`是输入音频文件或文件夹的路径，`output_path`是切分后的音频文件保存的文件夹路径。如果觉得原始音频声音太小，可以调高`alpha`参数，范围为0到1，值越大声音越大。

## 模型位置

项目使用的所有模型都存储在 `pretrained_models`目录中。BERT和HuBERT模型直接放在 `pretrained_models`下。

`gpt_weights`和 `sovits_weights`目录分别包含GPT和SoVITS的模型。

在这些目录中，每一个都有一个以训练时使用的 `name`参数命名的文件夹，在训练完成后，模型会被复制到这个文件夹中。

预训练的GPT和SoVITS模型位于它们各自的 `pretrained`子目录中，例如，`pretrained_models/gpt_weights/pretrained`。

## 训练

要快速开始训练，请使用提供的 `quick_start.sh`脚本：

```bash
bash quick_start.sh
```

您可能需要根据您的数据集的位置和结构，修改脚本中的 `data_dir`和 `name`参数（或者其他参数）。

## 推理

对于推理，支持WebUI和命令行两种方式。

### WebUI

要使用WebUI，只需用以下命令启动：

```bash
python server/webui.py
```

### 命令行

要从命令行进行推理，您可以：

1. 使用Bash脚本：

   ```bash
   bash inference.sh
   ```
2. 直接使用Python：

   ```bash
   python src/inference/inference.py \
    --sovits_weights your_sovits_model_path \
    --gpt_weights your_gpt_model_path \
    --parameters_file inference_parameters.json
   ```

在上面的命令中，您可以根据您存储预训练模型的位置，调整SoVITS和GPT模型权重的路径（`--sovits_weights`和 `--gpt_weights`）。

`--parameters_file`参数允许您指定包含推理参数的JSON或TXT文件，以实现批处理。

- **TXT格式**：每行代表一个单独的推理请求，格式如下：

  `参考音频 | 提示文本 | 提示语言 | 目标文本 | 目标文本语言 | 如何切割`
- **JSON格式**：JSON文件中的每个项目代表一个单独的推理请求。以下是JSON条目的示例结构：

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

## API使用

要启动API服务，请运行：

```
python server/app.py
```

您可以如下调用API：

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

同时，这个API还挂载了一个与上述类似的WebUI，可以通过以下方式访问：

```plaintext
http://localhost:8888/ai-speech/api/gradio
```

# 致谢

本项目建立在GPT-SoVITS的工作基础上，原始开发者为RVC-Boss。我们向RVC-Boss以及所有对GPT-SoVITS项目做出贡献的人表示最深的感谢，感谢他们在该领域的开创性工作，并且将他们的代码在MIT许可下提供给社区。此分支旨在探索原始项目的进一步增强和应用，我们希望它能对社区做出积极的贡献。

请访问原始项目：https://github.com/RVC-Boss/GPT-SoVITS
