import os
import sys
from typing import Optional
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, APIRouter
from contextlib import asynccontextmanager
from pydantic import BaseModel
import json
import tempfile
import soundfile as sf
import logging
from logging.handlers import RotatingFileHandler
from starlette.responses import FileResponse
import gradio as gr
import uvicorn

from server.webui import webui, LANG_DICT
from server.modelhandler import ModelHandler
from src.inference import TTSInference


gpt_model_handler = ModelHandler('pretrained_models/gpt_weights/')
sovits_model_handler = ModelHandler('pretrained_models/sovits_weights/')
tts_inference = TTSInference(is_half=False)


# 模型请求参数数据模型
class TTSModelRequest(BaseModel):
    ref_audio_path: str
    sovits_weights: str
    gpt_weights: str
    prompt_text: str
    prompt_language: str
    text: str
    text_language: str
    how_to_cut: Optional[str] = "不切"
    top_k: Optional[int] = 5
    top_p: Optional[float] = 0.7
    temperature: Optional[float] = 0.7
    ref_free: Optional[bool] = False
    

async def remove_temp_file(path: str):
    os.remove(path)

def setup_logging():
    #设置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    rotating_handler = RotatingFileHandler('app.log')
    rotating_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(rotating_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(stream_handler)
        
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 设置日志
    setup_logging()
    yield
    
app = FastAPI(lifespan=lifespan)
router = APIRouter()


@app.exception_handler(Exception)
async def exception_handler(request, exc):
    return {"error": str(exc)}, 500

# @router.post("/api/tts/upload-audio")
# async def upload_audio(audio_file: UploadFile):
#     return {"filename": audio_file.filename}


@router.post("/api/tts/inference")
async def predict(
                #   audio_file: UploadFile, 
                  data: TTSModelRequest, 
                  background_tasks: BackgroundTasks
                  ):
    try:
        if tts_inference.sovits_model != data.sovits_weights:
            tts_inference.change_sovits_weights(data.sovits_weights)
        if tts_inference.gpt_model!= data.gpt_weights:
            tts_inference.change_gpt_weights(data.gpt_weights)
        # 进行预测
        try:
            print('generating......')
            audio_generator = tts_inference.infer(
                data.ref_audio_path,
                data.prompt_text,
                LANG_DICT[data.prompt_language],
                data.text,
                LANG_DICT[data.text_language],
                how_to_cut=data.how_to_cut,
                top_k=data.top_k,
                top_p=data.top_p,
                temperature=data.temperature,
                ref_free=data.ref_free)
            
            sr, audio = next(audio_generator)
            print('generation finished!')

        except Exception as e:
            raise HTTPException(status_code=500, detail="Error during inference")
        
        # 创建一个临时文件来保存音频数据
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, audio, sr, 'PCM_24')
        
        # 在后台删除临时文件
        background_tasks.add_task(remove_temp_file, temp_file.name)

        return FileResponse(temp_file.name, media_type='audio/wav')  # 发送文件

    except KeyError as e:
        return {"error": f"Missing necessary parameter: {e.args[0]}"}, 400
    
# 挂载 Gradio 接口到 FastAPI 应用
ui = webui()
app = gr.mount_gradio_app(app, ui, path="/ai-speech/api/gradio")
app.include_router(router, prefix="/ai-speech")


if __name__ == '__main__':
    # 运行Uvicorn服务器
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")