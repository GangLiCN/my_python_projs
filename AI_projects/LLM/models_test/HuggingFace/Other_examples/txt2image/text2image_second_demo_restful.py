
import litellm
from litellm import completion 
import requests
import matplotlib.pyplot as plt

import io
from PIL import Image

import os
import time


'''
{'error': 'Model requires a Pro subscription; check out hf.co/pricing to learn more. Make sure to include your HF token in your query.'}
'''

litellm.set_verbose = True

os.environ["HUGGINGFACE_API_KEY"] = "xxxx"
os.environ["HF_HOME"] = "G:/models_local_cache"  # update it to match your test env

API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

# prompt="Astronaut riding a horse"
prompt="画一幅日本著名漫画 七龙珠 主人公悟空的图片"

begin_time=time.time()

image_bytes = query({
	"inputs": prompt,
})

# You can access the image with PIL.Image
try:
    image = Image.open(io.BytesIO(image_bytes))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Image')
    plt.show()
    
    end_time=time.time()
    exec_time=end_time - begin_time
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
       
except Exception as ex:
	print("Error occurred, detailed error message is:"+ex)


'''
messages = [{ "content": "If you feel unhappy and can't control yourself, what should you do to get around it ?","role": "user"}]

# e.g. Call 'Llama-2-7b-chat-hf' hosted on HF Inference endpoints
response = completion(
  model="huggingface/meta-llama/Llama-2-7b-chat-hf", 
  messages=messages, 
  stream=True
)

print(response)
for chunk in response:
  print(chunk)
'''
