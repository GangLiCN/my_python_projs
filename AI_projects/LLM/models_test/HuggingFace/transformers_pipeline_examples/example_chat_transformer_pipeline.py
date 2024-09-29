'''
   https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/README.md
'''
import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

''' Note that only GPU based host can run below script
    Reason: Torch doest not support float16 accuray on CPU 
Detailed log:    
  Pipelines loaded with `dtype=torch.float16` cannot run with `cpu` device. 
  It is not recommended to move them to `cpu` as running them will fail. 
  Please make sure to use an accelerator to run the pipeline in inference, 
  due to the lack of support for`float16` operations on this device in PyTorch. 
  
  Please, remove the `torch_dtype=torch.float16` argument, or use another 
  device for inference.    
    
'''

os.environ["HUGGINGFACE_API_KEY"] = "xxxx"
os.environ["HF_HOME"] = "G:/models_local_cache"  # update it to match your env


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a chatbot who always responds in private speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=8192,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])
