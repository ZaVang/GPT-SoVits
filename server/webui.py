import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gradio as gr
import argparse
from functools import partial
from modelhandler import ModelHandler
from src.inference import TTSInference
from src.utils.cut import CUT_DICT


gpt_model_handler = ModelHandler('pretrained_models/gpt_weights/')
sovits_model_handler = ModelHandler('pretrained_models/sovits_weights/')
tts_inference = TTSInference(is_half=False)
LANG_DICT = {
    "中文": 'zh',
    "英文": 'en',   
    "日文": 'jp',
    "中英混合": 'zh,en',
    "日英混合": 'ja,en',
    "多语种混合": 'auto',
}

def create_model_refresh_button(handlers, refresh_components, refresh_method, refreshed_args, refresh_value):
    """
    创建一个webui界面的刷新按钮，支持同时刷新多个组件。
    
    :param handlers: 包含需要调用的方法的类的实例，数量和要刷新的组件配对。
    :param refresh_components: 需要被刷新的组件列表。
    :param refresh_methods: 类实例中将被调用的方法的名称，为字符串。
    :param refreshed_args: 一个函数或字典，提供更新组件所需的参数，数量和要刷新的组件配对。
    :param refresh_value: 按钮的显示文本。
    """
    def refresh():
        for handler, refresh_component, refreshed_arg in zip(handlers, refresh_components, refreshed_args):
            getattr(handler, refresh_method)()
            args = refreshed_arg() if callable(refreshed_arg) else refreshed_arg
            updates = {}
            for k, v in args.items():
                setattr(refresh_component, k, v)
            updates[refresh_component] = gr.update(**(args or {}))
        return updates

    refresh_button = gr.Button(value=refresh_value)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=refresh_components
    )
    return refresh_button


def select_model_func(speaker_name, handler):
    if speaker_name == 'All speakers':
        model_list = []
        for speaker in handler.models_info:
            model_list.extend(list(handler.models_info[speaker].keys()))
        
    else:
        model_list = list(handler.models_info[speaker_name].keys())
        
    return gr.Dropdown(
                    choices = model_list,
                    label = '选择角色对应模型',
                    info = 'Models to choose',
                    value = model_list[0],
                    visible=True,
                    interactive = True
                )


def get_tts_wav(sovits_speaker,
                sovits_model,
                gpt_speaker,
                gpt_model,
                tts_ref_audio,
                tts_prompt_text,
                tts_prompt_language,
                tts_text,
                tts_text_language,
                how_to_cut="不切",
                top_k=5,
                top_p=0.7,
                temperature=0.7,
                ref_free = False):
    sovits_model = sovits_model_handler.models_info[sovits_speaker][sovits_model]
    gpt_model = gpt_model_handler.models_info[gpt_speaker][gpt_model]
    if tts_inference.sovits_model != sovits_model:
        tts_inference.change_sovits_weights(sovits_model)
    if tts_inference.gpt_model!= gpt_model:
        tts_inference.change_gpt_weights(gpt_model)
    
    audio_generator = tts_inference.infer(
        tts_ref_audio,
        tts_prompt_text,
        LANG_DICT[tts_prompt_language],
        tts_text,
        LANG_DICT[tts_text_language],
        how_to_cut=how_to_cut,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        ref_free=ref_free)
    
    sr, audio = next(audio_generator)
    return (sr, audio)
    

def webui():
    with gr.Blocks(theme=gr.themes.Default()) as ui:
        gr.Markdown(
        """
        <h1 style="text-align: center; font-size: 40px;">音频合成服务</h1>
        """     
        )
        with gr.Tab('文本合成音频'):
            gr.Markdown("Step1: 选择模型")
            with gr.Row():
                with gr.Column():
                    gpt_speakers_list = gr.Dropdown(
                        choices = ['All speakers'] + list(gpt_model_handler.models_info.keys()),
                        label = '选择角色',
                        value = 'All speakers',
                        info = 'Speakers to choose',
                        interactive = True
                    )
                    gpt_model_list = gr.Dropdown(
                        choices = [],
                        label = '选择角色对应模型',
                        info = 'Models to choose',
                        visible=False,
                        interactive = True
                    )
                with gr.Column():
                    sovits_speakers_list = gr.Dropdown(
                        choices = ['All speakers'] + list(sovits_model_handler.models_info.keys()),
                        label = '选择角色',
                        value= 'All speakers',
                        info = 'Speakers to choose',
                        interactive = True
                    )
                    sovits_model_list = gr.Dropdown(
                        choices = [],
                        label = '选择角色对应模型',
                        info = 'Models to choose',
                        visible=False,
                        interactive = True
                    )
                ## 刷新模型按钮
                model_refresh_button = create_model_refresh_button(
                    handlers=[gpt_model_handler, sovits_model_handler],
                    refresh_components=[gpt_speakers_list, sovits_speakers_list], 
                    refresh_method="load_models_info", 
                    refreshed_args=[lambda: {'choices': ['All speakers'] + list(gpt_model_handler.models_info.keys())},
                                    lambda: {'choices': ['All speakers'] + list(sovits_model_handler.models_info.keys())}],
                    refresh_value="刷新模型"
                )
                ## 当选择角色后，弹出对应文件夹下的模型
                gpt_speakers_list.change(partial(select_model_func, handler=gpt_model_handler), 
                                         inputs=[gpt_speakers_list], 
                                         outputs=gpt_model_list)
                sovits_speakers_list.change(partial(select_model_func, handler=sovits_model_handler),
                                            inputs=[sovits_speakers_list], 
                                            outputs=sovits_model_list)
            
            gr.Markdown("Step2: 上传参考音频和文本")
            with gr.Row():
                tts_ref_audio = gr.Audio(label="请上传3~10秒内参考音频，超过会报错！", type="filepath")
                with gr.Column():
                    tts_ref_text_free = gr.Checkbox(label="开启无参考文本模式。不填参考文本亦相当于开启。",
                                                info="使用无参考文本模式时建议使用微调的GPT，听不清参考音频说的啥(不晓得写啥)可以开，开启后无视填写的参考文本。",
                                                value=False, 
                                                interactive=True, 
                                                show_label=True)
                    tts_prompt_text = gr.Textbox(label="参考文本",
                                             placeholder="参考音频所对应的文本标注",
                                             value="", 
                                             interactive=True, 
                                             show_label=True)
                tts_prompt_language = gr.Dropdown(label="参考文本语言",
                                            choices=["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"],
                                            value="中文",
                                            interactive=True,
                                            show_label=True)
            
            gr.Markdown("Step3: 输入需要合成的文本")
            with gr.Row():
                tts_infer_text = gr.Textbox(label="合成文本",
                                placeholder="需要合成的文本",
                                value="",
                                interactive=True,
                                show_label=True)
                with gr.Column():
                    tts_infer_text_language = gr.Dropdown(label="合成文本语言",
                                            choices=["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"],
                                            value="中文",
                                            interactive=True,
                                            show_label=True)
                    tts_how_to_cut = gr.Dropdown(label="切分方式",
                                            choices=["不切"]+list(CUT_DICT.keys()),
                                            value="不切",
                                            interactive=True,
                                            show_label=True)
                with gr.Column():
                    tts_infer_top_k = gr.Slider(label="Top K",
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    value=5,
                                    interactive=True,
                                    show_label=True)
                    tts_infer_top_p = gr.Slider(label="Top P",
                                    minimum=0.1,
                                    maximum=1,
                                    step=0.05,
                                    value=0.7,
                                    interactive=True,
                                    show_label=True)
                    tts_infer_temperature = gr.Slider(label="Temperature",
                                        minimum=0.1,
                                        maximum=1,
                                        value=0.7,
                                        interactive=True,
                                        show_label=True)
                with gr.Column():
                    inference_button = gr.Button("开始合成", variant="primary")
                    tts_output_audio = gr.Audio(label="输出音频")
            inference_button.click(
                get_tts_wav,
                inputs = [sovits_speakers_list,
                          sovits_model_list,
                          gpt_speakers_list,
                          gpt_model_list,
                          tts_ref_audio,
                          tts_prompt_text,
                          tts_prompt_language,
                          tts_infer_text,
                          tts_infer_text_language,
                          tts_how_to_cut,
                          tts_infer_top_k,
                          tts_infer_top_p,
                          tts_infer_temperature,
                          tts_ref_text_free],
                outputs = [tts_output_audio]
            )
            
            
        with gr.Tab("音色转换"):
            gr.Markdown(
                """
                <h2 style="text-align: center; font-size: 30px;">开发中。。。敬请期待</h2>
                """
            )
        
        with gr.Tab("训练模型"):
            gr.Markdown(
                """
                <h2 style="text-align: center; font-size: 30px;">开发中。。。敬请期待</h2>
                """
            )
    
        with gr.Tab("预处理工具"):
            gr.Markdown(
                """
                <h2 style="text-align: center; font-size: 30px;">开发中。。。敬请期待</h2>
                """
            )
    return ui

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", default=False, help="make link public (used in colab)")

    args = parser.parse_args()
    ui = webui()
    # ui.queue(concurrency_count=2, max_size=10)
    ui.launch(share=args.share)

