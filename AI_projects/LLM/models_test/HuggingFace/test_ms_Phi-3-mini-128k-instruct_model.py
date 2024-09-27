# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import time


os.environ["HUGGINGFACE_API_KEY"] = "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"
os.environ["HF_HOME"] = "G:/models_local_cache"

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)

print("Loaded model successfully...")

input_text = '''
在windows的文件资源管理器中 鼠标右键点击某个文件，点击弹出菜单"属性", 
在"常规"页的最下方会看到名称为"安全"的标签，
如果"解除锁定"的复选框(checkbox)没有被选定，说明此文件处于"锁定"状态，
无法编辑，必须选中此复选框，才能解除文件的锁定。 但是我有一大堆这样的pdf
文件，手工解除锁定太麻烦。怎样编程实现? 请编写python代码 批量 执行"解除锁定"的操作。
注意: 解除锁定 不是指PDF文件的加密解密，也不是指文件本身的只读属性。
我机器上所有的pdf文件都没有加密。 它指的是从因特网上下载的pdf文件，
因为来源无法保证，也不是每个文件都有数字签名，所以默认情况下是无法编辑的，
只能 解除这个限制，你才可以编辑'''

input_ids = tokenizer(input_text, return_tensors="pt")

start_time=time.time()
print("Starting model inference, please wait ...")

outputs = model.generate(**input_ids,max_length=4096)

end_time=time.time()
exec_time=end_time - start_time
exec_time_conv=("%.4f" % float(exec_time))

print("Generation takes [{0}] seconds".format(exec_time_conv))
print(tokenizer.decode(outputs[0]))



''' 
   New sample inference code

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

messages = [
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
 
   
'''   