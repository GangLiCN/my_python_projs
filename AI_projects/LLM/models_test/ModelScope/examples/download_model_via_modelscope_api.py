
import time
import argparse
from modelscope import snapshot_download
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
def download_model(model_id):
    try:
        start_time = time.time()
        logging.info("开始下载模型...")
        model_dir = snapshot_download(model_id)

        end_time = time.time()
        exec_time = end_time - start_time

        exec_time_conv = ("%.4f" % float(exec_time))
        info_str="模型下载完毕，耗时 [{0}]秒.".format(exec_time_conv)
        print(info_str)
        logging.info(info_str)

    except Exception as ex:
        print("Error occurred while downloading:{0}".format(ex))


def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description="Download HuggingFace Models via 'huggingface-cli'.")

    # 添加必需参数（model_id）
    parser.add_argument('-m','--model_id', type=str, help='Model ID',required=True)

    # 解析参数
    args = parser.parse_args()

    # 下载模型
    download_model(args.model_id)

if __name__ == '__main__':
    main()