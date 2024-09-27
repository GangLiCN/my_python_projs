
'''
Run this script under Linux(e.g. WSL2 ubuntu)
'''

# 使用 VLLM后端进行推理
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

# GLM-4-9B-Chat-1M
# 如果遇见 OOM 现象，建议减少max_model_len，或者增加tp_size
max_model_len, tp_size = 1048576, 4

query = "你好,介绍一下你自己，然后介绍以下你和主要竞争对手的优缺点，用300字总结。"
model_name = "THUDM/glm-4-9b-chat-1m"
prompt = [{"role": "user", "content": query}]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# GLM-4-9B-Chat-1M 如果遇见 OOM 现象，建议开启下述参数
llm = LLM(
    model=model_name,
    tensor_parallel_size=tp_size,
    max_model_len=max_model_len,
    trust_remote_code=True,
    enforce_eager=True,
    enable_chunked_prefill=True,
    max_num_batched_tokens=8192
)
stop_token_ids = [151329, 151336, 151338]
sampling_params = SamplingParams(temperature=0.95, max_tokens=1024, stop_token_ids=stop_token_ids)

inputs = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
outputs = llm.generate(prompts=inputs, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
