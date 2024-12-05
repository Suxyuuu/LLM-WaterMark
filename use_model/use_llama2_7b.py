# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

seed = 42
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子

local_dir = '../models/'
model_path = local_dir + 'meta-llama/Llama-2-7b-hf'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = "cuda" # the device to load the model onto

prompt = "A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I am the "

model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
model.to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=30, do_sample=False)
generated_text = tokenizer.batch_decode(generated_ids)[0]
print(generated_text)