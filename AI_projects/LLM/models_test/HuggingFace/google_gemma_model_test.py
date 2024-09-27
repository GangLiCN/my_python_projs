from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import os

'''
torch.bfloat16：16位浮点数，用于神经网络，精度比torch.half略高。
主要用于深度学习训练，可以提高训练速度并减少内存使用。这是一种16位浮点数数据类型，
其中8个位用于指数部分，7个位用于小数部分。两者的区别在于指数部分的长度不同。

torch.qint8：8位量化整数。用于在模型推理阶段减少内存占用和提高计算速度，
但可能会牺牲一些模型精度。

torch.qint32：32位量化整数。与torch.qint8类似，用于在模型推理阶段减少内存
占用和提高计算速度，但可能会牺牲一些模型精度。

torch.quint8：8位无符号量化整数。用于在模型推理阶段减少内存占用和提高计算速度，
但可能会牺牲一些模型精度。无符号，因此只能表示非负整数

'''

os.environ["HUGGINGFACE_API_KEY"] = "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"
os.environ["HF_HOME"] = "G:/models_local_cache/HuggingFace"

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it")

print("Loaded model successfully...")

input_text = "用600字总结大语言模型LLM最近3年的发展历程，并对未来提出展望"

input_ids = tokenizer(input_text, return_tensors="pt")

start_time = time.time()
print("Starting model inference, please wait ...")

outputs = model.generate(**input_ids, max_length=1024)

end_time = time.time()
exec_time = end_time - start_time
exec_time_conv = ("%.4f" % float(exec_time))

print("Generation takes [{0}] seconds".format(exec_time_conv))
print(tokenizer.decode(outputs[0]))
