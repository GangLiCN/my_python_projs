
from diffusers import StableDiffusionPipeline
import torch
import os
import time
from functools import lru_cache


@lru_cache(maxsize=32)
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")

    return pipe


os.environ["HUGGINGFACE_API_KEY"] = "xxxx"
os.environ["HF_HOME"] = "G:/models_local_cache"  # update it to match your test env
    
begin_time=time.time()
my_pipe=load_model()

prompt = "画一幅日本著名漫画七龙珠中主人公悟空的图片"
image = my_pipe(prompt).images[0]  

end_time=time.time()
exec_time = end_time - begin_time
exec_time_conv = ("%.3f" % float(exec_time))
print("Time spents: {0} seconds.".format(exec_time_conv))
    
img_file=prompt+".png"
img_file_full_path=os.getcwd() + os.sep + img_file
    
b_overwrite=0
if os.path.exists(img_file_full_path):
   print("Detected existing image file, will overwrite it...")
   b_overwrite=1
if not os.path.exists(img_file_full_path) or (b_overwrite==1):
   image.save(img_file_full_path)
   if os.path.exists(img_file_full_path):
      print("Successfully generated image file [{0}].".format(img_file))
   else:
      print("Error occurred when writing image to local disk, please try again.")
      
