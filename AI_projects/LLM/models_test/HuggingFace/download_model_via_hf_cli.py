# Modified      Date           Comments
# Li Gang       2024-07-06     Fix: For HF login action,add new optional parameter "token"
# Li Gang       2024-06-11     Fix: using 'argparse' to parse command line parameters 
# Li Gang       2024-04-30     Latest update: introduce command line parameters


import os, sys
import pathlib
import time
import argparse
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

''' 
Alternative ways to download models from Huggingface:
  Call HuggingfaceCLI to download module
  ### Important ###
     Note that for some repositories which only allow 
   authorized user to access, You must execute [huggingface-cli login] 
   command first before downloading. for example: 

  ### Steps:
  1) huggingface-cli login --token xxx    # need access token

  2) Only download original pytorch model
     huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --resume-download 
     --include "original/*" --local-dir meta-llama/Meta-Llama-3-8B-Instruct
  Or: Exclude original pytorch model downloading
     huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --resume-download 
     --exclude "original/*" --local-dir meta-llama/Meta-Llama-3-8B-Instruct

## Google flan-t5-xxl model
https://huggingface.co/google/flan-t5-xxl/tree/main  
'''

os.environ["HUGGINGFACE_API_KEY"] = "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"
os.environ['HUGGING_FACE_HUB_TOKEN'] = "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"


os.environ["HF_HOME"] = "G:/models_local_cache/HuggingFace"
models_local_dir = os.environ["HF_HOME"]

'''Speed up HF model downloading
   Refer: https://huggingface.co/docs/huggingface_hub/v0.22.2/package_reference/environment_variables#hfhubenablehftransfer
   pip install huggingface_hub[hf_transfer], then set environment variable
'''
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "0"


def download_model(model_id, keyword_include, keyword_exclude, local_dir):
    str_download_cmd = ""
    str_option_1 = ""
    str_option_2 = ""
    str_option_3 = ""
    try:
        if not (keyword_include == ""):
            str_option_1 = " --include " + keyword_include
        if not (keyword_exclude == ""):
            str_option_2 = " --exclude " + keyword_exclude
        if not (local_dir == ""):
            str_option_3 = " --local-dir " + local_dir + os.sep + model_id

        str_option_reserved = " --resume-download "
        str_download_cmd = "huggingface-cli download " + model_id + str_option_reserved + str_option_1 + str_option_2 + str_option_3
        # debug
        print(f"Whole download cmd: {str_download_cmd}")

        logging.info("开始下载模型...")
        start_time = time.time()
        ## Execute download
        os.system(str_download_cmd)

        end_time = time.time()

        download_time = end_time - start_time
        download_time_conv = ("%.4f" % float(download_time))
        print(f"Model download finished, spent {download_time_conv} seconds.")
        logging.info(f"模型下载完成，花费{download_time_conv}秒.")

    except Exception as ex:
        print("Error occurred while downloading:{0}".format(ex))


def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description="Download HuggingFace Models via 'huggingface-cli'.")

    # 添加必需参数（model_id）
    parser.add_argument('-m', '--model_id', type=str, help='Model ID', required=True)

    # 添加可选参数（带有前缀的参数）
    parser.add_argument('-l', '--login', type=str, help='Login to HuggingFace before downloading model', default='no')

    parser.add_argument('-t', '--token', type=str, help='HuggingFace token', default='')

    parser.add_argument('-i', '--include_dir_or_file', type=str, help='Additional sub-dirs to include', default='')
    parser.add_argument('-e', '--exclude_dir_or_file', type=str, help='Additional sub-dirs to exclude', default='')
    parser.add_argument('-s', '--local_store_dir', type=str, help='Local store dir of model file', default='')

    parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity')

    # 解析参数
    args = parser.parse_args()

    # If the value of parameter "--login" =="yes" then login to HuggingFace before downloading model
    if args.login.lower() == "yes":
        if args.token:
            os.system("huggingface-cli login --token=" + args.token)
        else:
            print("You must provide token to access HuggingFace.")
            sys.exit(2)
    download_model(args.model_id, args.include_dir_or_file, args.exclude_dir_or_file, args.local_store_dir)


if __name__ == '__main__':
    main()
