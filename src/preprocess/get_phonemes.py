import os 
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traceback
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from text.cleaner import clean_text


def get_bert_feature(text, word2ph, tokenizer, bert_model, device):
   """
   获取指定文本的BERT特征表示。

   Args:
       text (str): 输入文本
       word2ph (list): 单词到音素的映射列表
       tokenizer (BertTokenizer): BERT分词器对象
       bert_model (BertModel): BERT模型对象
       device (str): 设备类型（'cuda'或'cpu'）

   Returns:
       torch.Tensor: 音素级别的BERT特征表示，形状为(bert_hidden_size, 音素数量)
   """
   with torch.no_grad():
       inputs = tokenizer(text, return_tensors="pt")
       for i in inputs:
           inputs[i] = inputs[i].to(device)
       res = bert_model(**inputs, output_hidden_states=True)
       # 获取倒数第三和倒数第二层的隐藏状态，并在序列长度维度上进行拼接
       res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

       # 检查word2ph长度与输入文本长度是否相等
       assert len(word2ph) == len(text)

       phone_level_feature = []
       for i in range(len(word2ph)):
           # 对每个单词的BERT特征进行重复，重复次数等于该单词对应的音素数量
           repeat_feature = res[i].repeat(word2ph[i], 1)
           phone_level_feature.append(repeat_feature)

       # 将所有音素的BERT特征拼接成一个张量
       phone_level_feature = torch.cat(phone_level_feature, dim=0)

       return phone_level_feature.T

def process(data, save_dir, tokenizer, bert_model, device):
    """
    处理输入数据，获取音素序列及BERT特征。

    Args:
        data (list): 输入数据列表，每个元素为[wav_name, text, language]
        save_dir (str): BERT特征保存目录
        tokenizer (BertTokenizer): BERT分词器对象
        bert_model (BertModel): BERT模型对象
        device (str): 设备类型（'cuda'或'cpu'）

    Returns:
        list: 处理结果列表，每个元素为[wav_name, 音素序列, word2ph, norm_text]
    """
    res = []
    os.makedirs(save_dir, exist_ok=True)

    for name, text, lan in data:
        try:
            name = os.path.basename(name)
            # 清理文本并获取音素序列、单词到音素的映射以及规范化后的文本
            phones, word2ph, norm_text = clean_text(
                text.replace("%", "-").replace("￥", ","), lan
            )
            path_bert = f"{save_dir}/{name}.pt"

            # 如果是中文文本且对应的BERT特征文件不存在，则计算并保存BERT特征
            if os.path.exists(path_bert) == False and lan == "zh":
                bert_feature = get_bert_feature(norm_text, word2ph, tokenizer, bert_model, device)
                assert bert_feature.shape[-1] == len(phones)
                torch.save(bert_feature, path_bert)
            phones = " ".join(phones)
            res.append([name, phones, word2ph, norm_text])
        except:
            print(name, text, traceback.format_exc())

    return res

def get_phonemes(input_txt_path: str, 
                 save_path: str, 
                 bert_pretrained_dir: str='pretrained_models/chinese-roberta-wwm-ext-large', 
                 is_half: bool=False, 
                 **kwargs) -> None:
   """
   从输入文本文件中获取音素序列和BERT特征。

   Args:
       input_txt_path (str): 输入文本文件路径
       save_path (str): 保存结果的路径
       bert_pretrained_dir (str, optional): BERT预训练模型路径. Defaults to 'pretrained_models/chinese-roberta-wwm-ext-large'.
       is_half (bool, optional): 是否使用半精度（FP16）模式. Defaults to False.

   Returns:
       None
   """
   os.makedirs(save_path, exist_ok=True)
   device = "cuda:0" if torch.cuda.is_available() else "cpu"

   tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)
   bert_model = AutoModelForMaskedLM.from_pretrained(bert_pretrained_dir)

   if is_half:
       bert_model = bert_model.half().to(device)
   else:
       bert_model = bert_model.to(device)

   bert_model.eval()

   todo = []
   with open(input_txt_path, "r", encoding="utf8") as f:
       lines = f.read().strip("\n").split("\n")
       for line in lines:
           try:
               wav_name, spk_name, language, text = line.split("|")
               todo.append([wav_name, text, language.lower()])
           except:
               print(line, traceback.format_exc())

   res = process(todo, f'{save_path}/bert_features', tokenizer, bert_model, device)

   opt = []
   for name, phones, word2ph, norm_text in res:
       opt.append("%s\t%s\t%s\t%s" % (name, phones, word2ph, norm_text))

   with open(f"{save_path}/text2phonemes.txt", "w", encoding="utf8") as f:
       f.write("\n".join(opt) + "\n")

   print("文本转音素已完成！")