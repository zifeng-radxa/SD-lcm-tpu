import gradio as gr
from sd import StableDiffusionPipeline, StableDiffusionXLPipeline, SD2_1_MODELS, SD_XL_MODELS
from sd import SD15Parallel, SDXL2Parallel
from sd import UpscaleModel
from PIL import Image
import numpy as np
import os
import time
import random
import torch
def seed_torch(seed=1029):
    seed=seed%4294967296
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("set seed to:", seed)

DEVICE_ID = 0
BASENAME  = "babes20lcm"
scheduler = "LCM"

pipe = StableDiffusionPipeline(
    basic_model=BASENAME,
    scheduler=scheduler
)
pipe.set_height_width(512,512)


def generate_image(text):
    # 这里是一个虚构的图像生成函数
    # 在实际应用中，您需要替换为实际的图像生成模型代码
    # 比如调用 DALL-E、Imagen 或其他模型
    # 这个函数应该根据输入的文本生成并返回图像
    img_pil = pipe(
        prompt=text,
        negative_prompt="low resolution",
        num_inference_steps=4,
        scheduler=scheduler,
        guidance_scale=1.05,
        seeds=[random.randint(0,100000)]
    )
    time_stamp = str(time.time()).split(".")[0]
    img_pil.save("{}.png".format(time_stamp))
    return "{}.png".format(time_stamp)

# 创建 Gradio 接口
iface = gr.Interface(
    fn=generate_image,                # 指定处理函数
    inputs="text",                    # 输入类型为文本
    outputs="image",                  # 输出类型为图像
    title="SDv1.5 Text-to-Image Generator🛸",  # 界面标题
    description="Generate images from text descriptions."  # 描述
)

# 运行 Gradio 应用
iface.launch(server_port=8999, server_name="0.0.0.0")

