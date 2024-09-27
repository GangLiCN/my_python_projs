
import time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 加载预训练模型和分词器
model_name = "meta/llama-3-8b-it"  # 假设这个模型名是正确的
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model_name)

# 定义测试输入
input_texts = [
    "Hello, how are you?",
    "What is the capital of France?",
    "Tell me a joke."
] * 100  # 扩展输入以增加测试负载

# 对输入进行分词
encoded_inputs = [tokenizer.encode(text, return_tensors="pt") for text in input_texts]

# 定义采样参数
sampling_params = SamplingParams(
    top_p=0.95,
    temperature=0.7,
    max_tokens=50
)

# 进行推理并测量性能
start_time = time.time()

# 使用 VLLM 进行推理
results = llm.sample(input_texts, sampling_params)

end_time = time.time()

# 计算和输出性能指标
total_time = end_time - start_time
avg_time_per_query = total_time / len(input_texts)

print(f"Total time taken: {total_time:.2f} seconds")
print(f"Average time per query: {avg_time_per_query:.4f} seconds")

# 打印部分结果
for i, result in enumerate(results[:5]):
    print(f"Input: {input_texts[i]}")
    print(f"Output: {result['text']}\n")