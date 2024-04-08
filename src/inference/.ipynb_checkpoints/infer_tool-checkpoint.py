import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Optional, Union
from abc import abstractmethod, ABC
from time import time as ttime

from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa, torch
import LangSegment

from feature_extractor.cnhubert import CNHubert
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from utils.utils import SupportedLanguage, load_audio
from utils.config import DictToAttrRecursive
from utils.cut import CUT_DICT, SPLITS, get_first


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


class InferenceModule(ABC):
    def __init__(self, 
                 *,
                 sovits_weights: str=None,
                 gpt_weights: str=None,
                 is_half: bool=False,
                 bert_path: str='pretrained_models/chinese-roberta-wwm-ext-large',
                 cnhubert_path: str='pretrained_models/chinese-hubert-base',
                 **kwargs,
                 ) -> None:
        self.sovits_model_path = sovits_weights
        self.gpt_model_path = gpt_weights
        self.sovits_model = None
        self.gpt_model = None
        self.sovits_config = None
        self.gpt_config = None
        self.is_half = is_half
        self.hz = 50
        
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        if self.sovits_model_path is not None:
            self.change_sovits_weights(sovits_weights)
        if self.gpt_model_path is not None:
            self.change_gpt_weights(gpt_weights)
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        self.ssl_model = CNHubert(cnhubert_path)
        self.ssl_model.eval()
        if is_half == True:
            self.bert_model = self.bert_model.half()
            self.ssl_model = self.ssl_model.half()
        self.bert_model.to(self.device)
        self.ssl_model.to(self.device)

                
    def change_sovits_weights(self, sovits_weights: str):
        model_dict = torch.load(sovits_weights, map_location="cpu")
        self.sovits_config = DictToAttrRecursive(model_dict["config"])
        # sovits_config.model.semantic_frame_rate = "25hz"
        self.sovits_model = SynthesizerTrn(
            self.sovits_config.data.filter_length // 2 + 1,
            self.sovits_config.train.segment_size // self.sovits_config.data.hop_length,
            n_speakers=self.sovits_config.data.n_speakers,
            **self.sovits_config.model
        )
        # # enc q在推理时不需要
        # if ("pretrained" not in sovits_path):
        #     del vq_model.enc_q
        self.sovits_model.load_state_dict(model_dict["weight"], strict=False)
        if self.is_half == True:
            self.sovits_model = self.sovits_model.half()
        self.sovits_model = self.sovits_model.to(self.device)
        self.sovits_model.eval()
        print(f'Model changed to: {sovits_weights}')
        
    def change_gpt_weights(self, gpt_weights: str):
        model_dict = torch.load(gpt_weights, map_location="cpu")
        self.gpt_config = DictToAttrRecursive(model_dict["config"])
        self.gpt_model = Text2SemanticLightningModule(self.gpt_config, "****", is_train=False)
        self.gpt_model.load_state_dict(model_dict["weight"])
        if self.is_half == True:
            self.gpt_model = self.gpt_model.half()
        self.gpt_model = self.gpt_model.to(self.device)
        self.gpt_model.eval()
        print(f'Model changed to: {gpt_weights}') 
        
       
    @abstractmethod
    def infer(self, **kwargs):
        pass
    
    
class TTSInference(InferenceModule):
    def __init__(self, 
                 *, 
                 sovits_weights: str = None,
                 gpt_weights: str = None,
                 is_half: bool = False,
                 **kwargs,
                 ) -> None:
        super().__init__(sovits_weights=sovits_weights, 
                         gpt_weights=gpt_weights, 
                         is_half=is_half,
                         **kwargs)
        
    def infer(self,
              ref_wav_path: Union[str, np.ndarray],
              prompt_text: Optional[str],
              prompt_language: SupportedLanguage,
              text: str,
              text_language: SupportedLanguage,
              how_to_cut: str,
              top_k: int=5,
              top_p: float=0.7,
              temperature: float=0.7,
              ref_free: bool=False,
              **kwargs
              ):
        if prompt_text is None or len(prompt_text) == 0:
            ref_free = True
            
        if not ref_free:
            prompt_text = prompt_text.strip("\n")
            if (prompt_text[-1] not in SPLITS): 
                prompt_text += "。" if prompt_language!= "en" else "."
            print("实际输入的参考文本:", prompt_text)
        
        text = text.strip("\n")
        if (text[0] not in SPLITS and len(get_first(text)) < 4):
            text = "。" + text if text_language!= "en" else "." + text
        print("实际输入的目标文本:", text)
        
        zero_wav = np.zeros(
            int(self.sovits_config.data.sampling_rate * 0.3),
            dtype=np.float32
        )
        
        with torch.no_grad():
            if isinstance(ref_wav_path, str):
                wav16k, _ = librosa.load(ref_wav_path, sr=16000)
            else:
                wav16k, _ = ref_wav_path
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 16000):
                raise OSError("参考音频在3~10秒范围外，请更换！")
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if self.is_half == True:
                wav16k = wav16k.half().to(self.device)
                zero_wav_torch = zero_wav_torch.half().to(self.device)
            else:
                wav16k = wav16k.to(self.device)
                zero_wav_torch = zero_wav_torch.to(self.device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()
            codes = self.sovits_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
        
        cut_func = CUT_DICT.get(how_to_cut, None)
        if cut_func:
            text = cut_func(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        print("实际输入的目标文本(切句后):", text)
        
        texts = text.split("\n")
        texts = merge_short_text_in_array(texts, 5)
        audio_opt = []
        if not ref_free:
            phones1, bert_features1, norm_text1 = self.get_phones_and_bert(prompt_text, prompt_language)
            
        for text in texts:
            # 解决输入目标文本的空行导致报错的问题
            if (len(text.strip()) == 0):
                continue
            if (text[-1] not in SPLITS): 
                text += "。" if text_language != "en" else "."
            print("实际输入的目标文本(每句):", text)
            
            phones2, bert_features2, norm_text2 = self.get_phones_and_bert(text, text_language)
            print("前端处理后的文本(每句):", norm_text2)
            
            if not ref_free:
                bert_features = torch.cat([bert_features1, bert_features2], 1)
                all_phoneme_ids = torch.LongTensor(phones1+phones2).to(self.device).unsqueeze(0)
            else:
                bert_features = bert_features2
                all_phoneme_ids = torch.LongTensor(phones2).to(self.device).unsqueeze(0)

            bert_features = bert_features.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)
            prompt = prompt_semantic.unsqueeze(0).to(self.device)

            with torch.no_grad():
                # pred_semantic = t2s_model.model.infer(
                pred_semantic, idx = self.gpt_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if ref_free else prompt,
                    bert_features,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.hz * self.gpt_config.data.max_sec,
                )

            # print(pred_semantic.shape,idx)
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(
                0
            )  # .unsqueeze(0)#mq要多unsqueeze一次
            refer = self.get_spepc(ref_wav_path)  # .to(device)
            if self.is_half == True:
                refer = refer.half()
            refer = refer.to(self.device)
            # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
            audio = (
                self.sovits_model.decode(
                    pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0), refer
                ).detach().cpu().numpy()[0, 0]
            )  ###试试重建不带上prompt部分
            max_audio=np.abs(audio).max()#简单防止16bit爆音
            if max_audio>1:audio/=max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav)
        yield self.sovits_config.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
            np.int16
        )

            
    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T


    def clean_text_inf(self, text, language):
        phones, word2ph, norm_text = clean_text(text, language)
        phones = cleaned_text_to_sequence(phones)
        return phones, word2ph, norm_text


    def get_phones_and_bert(self, text, language):
        dtype=torch.float16 if self.is_half == True else torch.float32
        if language in {"en","all_zh","all_ja"}:
            language = language.replace("all_","")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            else:
                # 因无法区别中日文汉字,以用户输入为准
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            phones, word2ph, norm_text = self.clean_text_inf(formattext, language)
            if language == "zh":
                bert_features = self.get_bert_feature(norm_text, word2ph).to(self.device)
            else:
                bert_features = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float16 if self.is_half == True else torch.float32,
                ).to(self.device)
        elif language in {"zh", "ja","auto"}:
            textlist=[]
            langlist=[]
            LangSegment.setfilters(["zh","ja","en", "ko"])
            if language == "auto":
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "ko":
                        langlist.append("zh")
                        textlist.append(tmp["text"])
                    else:
                        langlist.append(tmp["lang"])
                        textlist.append(tmp["text"])
            else:
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            print(textlist)
            print(langlist)
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang)
                bert_feature = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert_feature)
            bert_features = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)

        return phones, bert_features.to(dtype), norm_text


    def get_bert_inf(self, phones, word2ph, norm_text, language):
        language=language.replace("all_","")
        if language == "zh":
            bert_feature = self.get_bert_feature(norm_text, word2ph).to(self.device)#.to(dtype)
        else:
            bert_feature = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if self.is_half == True else torch.float32,
            ).to(self.device)

        return bert_feature


    def get_spepc(self, filename):
        hps = self.sovits_config
        audio = load_audio(filename, int(hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        return spec