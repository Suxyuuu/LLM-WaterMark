from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

local_dir = '../models/'
model_path = local_dir + 'facebook/opt-2.7b'
quant_path = local_dir + 'facebook/opt-2.7b-awq-4bit'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)