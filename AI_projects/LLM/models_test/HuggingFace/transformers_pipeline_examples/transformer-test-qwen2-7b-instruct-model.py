'''
Ref:

## 预处理模型
https://www.modelscope.cn/docs/%E5%8A%A0%E8%BD%BD%E6%A8%A1%E5%9E%8B%E5%92%8C%E9%A2%84%E5%A4%84%E7%90%86%E5%99%A8

## 模型推理
https://www.modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%8E%A8%E7%90%86Pipeline

## 模型训练
https://www.modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AE%AD%E7%BB%83Train

'''

import time
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

os.environ["HUGGINGFACE_API_KEY"] = "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"
os.environ["HF_HOME"] = "G:/models_local_cache"

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

prompt = "简单介绍一下你自己（Qwen2),同时告诉我们你和其他主流开源大模型例如llama3的优缺点。"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)


start_time=time.time()
print("开始推理，请等待...")

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)

end_time=time.time()
exec_time=end_time - start_time
exec_time_conv=("%.4f" % float(exec_time))

print("Generation takes [{0}] seconds".format(exec_time_conv))

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Answer:"+ response)
