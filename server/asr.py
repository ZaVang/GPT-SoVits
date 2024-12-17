from agent_kernel import get_llm_client, ASRRequest
from pydub import AudioSegment
import os
import pandas as pd
import re
import asyncio


async def get_asr_result(audio_path: str) -> str:
    # 加载音频文件
    audio = AudioSegment.from_wav(audio_path)
    
    # 检查音频是否为16位和单声道
    if audio.sample_width != 2 or audio.channels != 1:
        # 如果不是16位或单声道，则转换成16位和单声道并保存成一个临时的文件
        temp_audio_path = audio_path.replace(".wav", "_tmp.wav")
        audio = audio.set_sample_width(2).set_channels(1)  # 2 bytes = 16 bits, 1 channel = mono
        audio.export(temp_audio_path, format="wav")
    else:
        temp_audio_path = audio_path

    # 调用agent kernel的asr接口，返回识别结果
    client = get_llm_client()
    r = ASRRequest.from_file(temp_audio_path)
    r.lang_code = "cmn-Hans-CN"
    res = await client.asr_async(r)  # Async API
    transcript = res.response.alternatives[0].transcript # type: ignore

    # 删除临时文件
    if temp_audio_path != audio_path:
        os.remove(temp_audio_path)

    return transcript


def calculate_edit_distance(hypothesis, reference, lang="zh"):
    """
    计算 Translation Edit Rate (TER)
    """
    # 去除标点符号
    hypothesis = re.sub(r'[^\w\s]', '', hypothesis)
    reference = re.sub(r'[^\w\s]', '', reference)
    
    # 将字符串分割成单词列表
    if lang == "en":
        hyp_words = hypothesis.split(" ")
        ref_words = reference.split(" ")
    else:
        hyp_words = list(hypothesis)
        ref_words = list(reference)
    
    # 初始化编辑距离矩阵
    d = [[0] * (len(ref_words) + 1) for _ in range(len(hyp_words) + 1)]
    
    # 初始化边界条件
    for i in range(len(hyp_words) + 1):
        d[i][0] = i
    for j in range(len(ref_words) + 1):
        d[0][j] = j
    
    # 动态规划计算编辑距离
    for i in range(1, len(hyp_words) + 1):
        for j in range(1, len(ref_words) + 1):
            if hyp_words[i - 1] == ref_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j] + 1,    # 删除
                              d[i][j - 1] + 1,    # 插入
                              d[i - 1][j - 1] + 1)  # 替换
    
    # 编辑距离
    edit_distance = d[len(hyp_words)][len(ref_words)]
    return edit_distance


def match_result(asr_result, lines):
    ## 使用手动计算的TER来计算匹配度
    edit_distance = calculate_edit_distance(asr_result, lines)
    
    if edit_distance == 0:
        return 0
    elif edit_distance <= 2:
        return 1
    else:
        return 2


async def process_audio(resource_name, line, audio_folder):
    audio_path = os.path.join(audio_folder, resource_name + '.wav')  # 修正路径拼接
    if os.path.exists(audio_path):
        try:
            # 调用预先写好的get_asr_result函数
            asr_result = await get_asr_result(audio_path)
        except Exception as e:
            print(f"Error processing {resource_name}: {e}")
            asr_result = ""
    else:
        print(f"Audio file {resource_name} not found in {audio_folder}")
        asr_result = ""
    
    print(f"ASR Result for {resource_name}: {asr_result}")
    match_result_value = match_result(asr_result, line)
    return asr_result, match_result_value


async def audio_line_check(excel_path, audio_folder):
    # 读取Excel文件
    df = pd.read_excel(excel_path, header=0)
    
    # 获取台词和资源命名
    lines = df.iloc[:, 0]
    resource_names = df.iloc[:, 1]
    
    # 使用asyncio.gather并行处理音频文件
    tasks = [process_audio(resource_name, line, audio_folder) for resource_name, line in zip(resource_names, lines)]
    results = await asyncio.gather(*tasks)
    
    # 分离ASR结果和匹配结果
    asr_results, match_results = zip(*results)
    
    ## 重新组装excel，前两列不变，加一列asr result命名为AI识别文本，和一列match result命名为匹配度
    df['AI识别文本'] = asr_results
    df['匹配度'] = match_results
    
    # 保存新的Excel文件
    new_excel_path = excel_path.replace("需求表.xlsx", "审查结果.xlsx")
    df.to_excel(new_excel_path, index=False)
    print(f"Results saved to {new_excel_path}")
