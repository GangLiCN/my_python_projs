	
# 使用 transformers 后端进行推理
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

model_id="google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)

query = "你好,介绍一下你自己，然后介绍以下你和主要竞争对手的优缺点，用300字总结。"

inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )

inputs = inputs.to(device)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()


gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
