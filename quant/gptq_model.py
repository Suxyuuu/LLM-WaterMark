import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer
import torch
import argparse

# model_name = 'facebook/opt-1.3b'
# local_dir = '/root/autodl-tmp/watermark_project/models/'
# model_path = local_dir + model_name
# quant_path = local_dir + model_name + '-gptq-8bit'

def quantize_model(model_path, quant_path, model_seqlen):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

    quantizer = GPTQQuantizer(bits=8, dataset="c4", model_seqlen = model_seqlen)
    quantized_model = quantizer.quantize_model(model, tokenizer)

    quantized_model.save_pretrained(quant_path)
    tokenizer.save_pretrained(quant_path)

def main():
    parser = argparse.ArgumentParser(description='Quantize a model and save it to a specified path')
    parser.add_argument('--model_path', type=str, default='../models/facebook/opt-1.3b', help='Path to the pretrained model')
    parser.add_argument('--quant_path', type=str, default='../models/facebook/opt-1.3b-gptq-8bit', help='Path to save the quantized model')
    parser.add_argument('--model_seqlen', type=int, required=True, help='model max context length')
    args = parser.parse_args()
    print(args.model_seqlen)
    print(type(args.model_seqlen))
    quantize_model(args.model_path, args.quant_path, args.model_seqlen)

if __name__ == "__main__":
    main()
    print("quant done")