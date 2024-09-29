
''' Note that you need to download model "blip2-opt-2.7b" first
    from ModelScope web site.
'''

from modelscope import AutoModelForCausalLM, AutoTokenizer
import time
import torch

import requests

from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"

processor =Blip2Processor.from_pretrained("./blip2/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("./blip2/blip2-opt-2.7b",device_map="auto")

raw_img=Image.open("./images/to_be_checked_pic_3.png").convert("RGB")


prompt = "这张图片展示的哪个地方？位于什么国家的什么地区?"

inputs = processor(raw_img,prompt,return_tensors="pt").to(device)


start_time=time.time()
print("开始推理，请等待...")

out=model.generate(**inputs,max_length=1024)
out_final=processor.decode(out[0],skip_special_tokens=True)
print(f"Response: {out_final}")

end_time=time.time()
exec_time=end_time - start_time
exec_time_conv=("%.4f" % float(exec_time))

print("推理耗时 [{0}] 秒。".format(exec_time_conv))
