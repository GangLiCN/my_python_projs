
import os
import pathlib
import datetime
from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi
from safetensors.torch import load_file
import torch

os.environ["HUGGINGFACE_API_KEY"] = xxxx"
global sf_file_name

def recursive_search(start_dir):
    b_find_model = 0
    for foldername, subfolders, filenames in os.walk(start_dir):
        if b_find_model ==1:
           break

        for filename in filenames:
            if filename.lower().endswith('.safetensors'):
               # print("subfolders="+str(subfolders))
               file_path = os.path.join(foldername,filename)
               b_find_model=1
               print("Local cached file:{0}".format(file_path))
               break

    return (b_find_model,file_path)


def search_models(keyword):
    api = HfApi()
    models = api.list_models(filter=keyword)
    for model in models:

        print(f"Model ID (repo_id): {model.modelId}")
        print(f"Tags: {model.tags}")
        print("------\n")

# 示例：搜索包含"stable-diffusion"关键词的模型
search_models("stable-diffusion-xl-1.0")


''' In conda env, you must install torch via:
conda install pytorch torchvision torchaudio cpuonly -c pytorch

Ref: https://blog.csdn.net/weixin_46277020/article/details/123251014
'''

filename = "vae/diffusion_pytorch_model.safetensors"

filename_short="diffusion_pytorch_model.safetensors"

repo_id="diffusers--stable-diffusion-xl-1.0-inpainting-0.1"

model_file_base_dir=str(pathlib.Path.home()) + os.sep +"/.cache/huggingface/hub"
model_file_sub_dir="models--"+ repo_id
model_file_final_dir=model_file_base_dir+ os.sep + model_file_sub_dir+os.sep+"snapshots"

(b_check_model_exist,local_file_path)=recursive_search(model_file_base_dir)
if (b_check_model_exist ==1):
    print("Found model file from local cache dir, skip model file downloading...")
    sf_filename=local_file_path
else:
    ## Download safetensors & torch weights for [diffusers/stable-diffusion-xl-1.0-inpainting-0.1] model
    print("Start model file downloading, please wait...")
    try:
       sf_filename = hf_hub_download("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", filename=filename)

       print("Finished model file downloading, You can find the model file from local cache dir [(0}:"
             .format(model_file_final_dir))
     # pt_filename = hf_hub_download("stable-diffusion-v1-4/", filename="pytorch_model.bin")
    except Exception as ex:
       print("Error occured while downloading:{0}".format(ex))


## CPU基准测试
## Load safetensors model file
start_st = datetime.datetime.now()
print("Safe tensor file:{0}".format(sf_filename))
weights = load_file(sf_filename, device="cpu")
load_time_st = datetime.datetime.now() - start_st
print("Loaded safetensors model file in [{0}] seconds.".format(load_time_st))


'''
## GPU基准测试
os.environ["SAFETENSORS_FAST_GPU"] = "1"
torch.zeros((2, 2)).cuda()
start_st = datetime.datetime.now()
weights = load_file(sf_filename, device="cuda:0")
load_time_st = datetime.datetime.now() - start_st
print(f"Loaded safetensors {load_time_st}")

start_pt = datetime.datetime.now()
weights = torch.load(pt_filename, map_location="cuda:0")
load_time_pt = datetime.datetime.now() - start_pt

print(f"Loaded pytorch {load_time_pt}")
print(f"on GPU, safetensors is faster than pytorch by: {load_time_pt/load_time_st:.1f} X")
'''

