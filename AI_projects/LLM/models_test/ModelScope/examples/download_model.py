
#模型下载
from modelscope import snapshot_download
import logging
import time

# 设置日志
logging.basicConfig(level=logging.INFO)
# tclf90/glm-4-9b-chat-GPTQ-Int4
model_id="tclf90/glm-4-9b-chat-GPTQ-Int4"

start_time = time.time()

logging.info("开始下载模型，请等待...")
model_dir = snapshot_download(model_id)

end_time = time.time()
info_str="模型下载完毕，耗时{:.4f}秒".format(end_time - start_time)
logging.info(info_str)
