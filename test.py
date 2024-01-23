from sd import StableDiffusionPipeline, StableDiffusionXLPipeline, SD2_1_MODELS, SD_XL_MODELS
from sd import SD15Parallel, SDXL2Parallel
from sd import UpscaleModel
from PIL import Image
import numpy as np
import os
import time

DEVICE_ID = 0
BASENAME  = "wujielcm"
scheduler = "LCM"
# scheduler = 'Euler a'

pipe = StableDiffusionPipeline(
    basic_model=BASENAME,
    scheduler=scheduler
)
pipe.set_height_width(512,512)

#npz_file = "/data/aigc/demos/tpukern/test/mw/coeffex/lcm_lora_reorder.npz"
#match_file = "/data/aigc/demos/tpukern/test/mw/coeffex/unet_match.csv"
#pipe.unet_pure.load_lora_file(npz_file, match_file)

img_pil = pipe(
    prompt="a chinese girl, 4k",
    negative_prompt="low resolution",
    num_inference_steps=4,
    scheduler=scheduler,
    guidance_scale=1.05
)
img_pil.save("test.png")

# print(img_pil)
