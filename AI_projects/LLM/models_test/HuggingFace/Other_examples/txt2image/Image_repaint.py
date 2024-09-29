
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

from PIL import Image
from functools import lru_cache
import os
import pathlib
import sys

import datetime
import time

''' 
Below code will download model if there is no model files from local cache dir

Below code will throw exception if running on CPU-only environment because
CPU doesnot support fp16 


pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cpu")
'''


os.environ["HUGGINGFACE_API_KEY"] = "xxxx"
os.environ['HUGGING_FACE_HUB_TOKEN'] = "xxxx"

os.environ["HF_HOME"] = "G:/models_local_cache/HuggingFace"  # update it to match your test env

@lru_cache(maxsize=32)
def load_model():
    pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1").to("cpu")
    return pipe


def generate_image(pipeline,prompt,ori_image,mask_image,generator):
    if (pipeline is not None):
        new_image = pipeline(
            prompt=prompt,
            torch_dtype=torch.float32,
            image=ori_image,
            mask_image=mask_image,
            guidance_scale=8.0,
            num_inference_steps=15,  # steps between 15 and 30 work well
            strength=0.9,  # make sure to use `strength` below 1.0
            generator=generator,
        ).images[0]
        
        return new_image
        

def main():
    # A dog sitting on the chair
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"

    # an image to be masked
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

    ori_image = load_image(img_url).resize((1024, 1024))
    mask_image = load_image(mask_url).resize((1024, 1024))

    prompt = "A beautiful monkey is sitting on a park bench"
    
    st_time=time.time()
    cur_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print("Start time:{0}".format(cur_time))
    
    generator = torch.Generator(device="cpu").manual_seed(0)

    my_pipe = load_model()
    my_image = generate_image(my_pipe,prompt,ori_image,mask_image,generator)

    ''' Pillow usage
        https://pillow.readthedocs.io/en/stable/reference/Image.html
    '''
    if (my_image is not None):
        end_time=time.time()
        exec_time=end_time-st_time
            
        cur_time_end=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        print("End time:{0}".format(cur_time_end))
        
        new_png_file="masked_new_"+ cur_time +".png"
        my_image.save(new_png_file)
        print("Time spents:[{0}] seconds, new masked image file successfull generated, please check it manually.".format(exec_time))
            
    else:
        print("Error occurred during image generating, please retry.")


if __name__ == '__main__':
    main()

''' Errors collection
ValueError: The deprecation tuple ('no variant default', '0.24.0', "You are trying to load the model files of the `variant=fp32`, but no such modeling files are available.The default model files: {'vae/diffusion_pytorch_model.safetensors', 'text_encoder_2/model.safetensors', 'text_encoder/model.safetensors', 'unet/diffusion_pytorch_model.safetensors'} will be loaded instead. Make sure to not load from `variant=fp32`if such variant modeling files are not available. Doing so will lead to an error in v0.24.0 as defaulting to non-variantmodeling files is deprecated.") should be removed since diffusers' version 0.27.1 is >= 0.24.0

## Answers from GPT-4
这个错误消息提示的问题与模型的变体和数据类型有关，特别是在尝试将数据类型从float16改为float32时遇到的。这里的核心问题是尝试加载一个指定变体（在这种情况下是fp32，即float32）的模型，但是该变体的模型文件不存在，导致回退到默认的模型文件。

错误消息还提到了一个弃用的问题，即从特定版本开始（在这个例子中是0.24.0），如果请求的变体模型文件不存在，就不应该默认回退到非变体模型文件，而你使用的diffusers库版本是0.27.1，已经超过了引入这一变更的版本。

解决这个问题，你可以采取以下几个步骤：

确认模型变体：确保你尝试加载的模型变体（在这个案例中是float32版本的模型）确实可用。可能的话，查看你的模型来源或官方文档，确认是否提供了fp32变体的模型文件。

检查diffusers库的版本：尽管错误信息提到的是关于弃用的警告，但确保你使用的diffusers库是最新的，或者至少是兼容你的需求的版本。如果有更新可用，考虑升级。

显式指定模型文件：如果可能，尝试显式指定模型文件路径，而不是依赖于自动解析变体。这可能需要你下载或以其他方式访问特定变体的模型文件。

回退或选择其他变体：如果fp32变体不可用，考虑使用其他可用的变体，或者如果默认模型文件满足需求，可以尝试不指定变体。

联系模型或库的维护者：如果问题持续存在，可能需要联系模型或diffusers库的维护者，寻求他们的帮助或报告这个问题。

在处理这类问题时，理解你所使用的具体模型和库的版本及其兼容性要求是非常重要的，因为这可以帮助你更准确地定位问题并找到解决方案。此外，如果模型的不同变体在使用时有明确的性能或资源使用考虑，了解这些差异也同样重要。
'''

