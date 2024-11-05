import pandas as pd
import os
import argparse
import re

def convert_xlsx_to_txt(name, language) -> None:
    # 读取xlsx文件
    xlsx_path = os.path.join('data', name, 'lines.xlsx')
    df = pd.read_excel(xlsx_path, header=None)  # 不使用表头
    
    # 创建一个字典来存储音频名称和对应的文本
    audio_text_dict = {}
    for index, row in df.iterrows():
        text = row[0]  # 第一列
        audio_name = row[1]  # 第二列
        if pd.isna(text):
            continue
        # 删除括号里的部分，包括中文括号
        text = re.sub(r'\(.*?\)', '', text).strip()
        text = re.sub(r'\[.*?\]', '', text).strip()
        text = re.sub(r'（.*?）', '', text).strip()  # 处理中文括号
        if not text:
            continue
        audio_text_dict[audio_name] = text
    
    # 遍历vocal文件夹下的所有wav文件
    vocal_dir = os.path.join('data', name, 'vocal')
    wav_files = [f for f in os.listdir(vocal_dir) if f.endswith('.wav')]
    
    total_wav_count = len(wav_files)
    no_text_count = 0
    
    # 打开name.txt文件进行写入
    txt_path = os.path.join('data', name, f'{name}.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        for wav_file in wav_files:
            # 去掉末尾的-00x
            base_name = re.sub(r'-\d{3}$', '', wav_file[:-4])
            
            if base_name in audio_text_dict:
                text = audio_text_dict[base_name]
                f.write(f"{wav_file}|{name}|{language}|{text}\n")
            else:
                print(f"Warning: {wav_file} does not have a corresponding text.")
                no_text_count += 1
    
    print(f"Total WAV files: {total_wav_count}")
    print(f"WAV files without corresponding text: {no_text_count}")

def main() -> None:
    parser = argparse.ArgumentParser(description='Convert lines.xlsx to name.txt')
    parser.add_argument('-n', '--name', type=str, help='Name of the dataset')
    parser.add_argument('-l', '--language', type=str, help='Language code (e.g., zh, ja, en)')
    
    args = parser.parse_args()
    
    convert_xlsx_to_txt(args.name, args.language)

if __name__ == '__main__':
    main()
