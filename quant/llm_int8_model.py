# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# local_dir = '../models/'
# model_path = local_dir + 'facebook/opt-2.7b'
# quant_path = local_dir + 'facebook/opt-2.7b-llm_int8-8bit'

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def quantize_model(model_path, quant_path):
    quantization_config = BitsAndBytesConfig(
        llm_int8_threshold=6.0,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
    )

    model_8bit = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model_8bit.save_pretrained(quant_path)
    tokenizer.save_pretrained(quant_path)

def main():
    parser = argparse.ArgumentParser(description='Quantize a model and save it to a specified path')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--quant_path', type=str, required=True, help='Path to save the quantized model')
    args = parser.parse_args()

    quantize_model(args.model_path, args.quant_path)

if __name__ == "__main__":
    main()
    print("quant done")