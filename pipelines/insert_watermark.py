import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
import numpy as np
import torch
from transformers import AutoTokenizer

from lib.insert import insert_watermark
from lib.utils import get_llm, format_time



def print_config(args):
    print(f'Config parameters:')
    print(f"+ model: {args.model}")
    print(f"+ hidden_size: {args.hidden_size}")
    print(f"+ watermark: {args.watermark}")
    print(f"+ password: {args.password}")
    print(f"+ calibdation data: {args.calibdation_dataset}")
    print(f"+ num of samples: {args.nsamples}")
    print(f"+ random seed: {args.seed}")
    print(f"+ select_ratio: {args.select_ratio}")
    print(f"+ gamma_layer: {args.gamma_1}")
    # print(f"+ gamma_row: {args.gamma_2}")
    # print(f"+ gamma_weight: {args.gamma_3}")
    print(f"+ xi: {args.xi}")
    print(f"+ position_num: {args.xi}")
    print(f"+ save_model_path: {args.save_model}")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model path')
    parser.add_argument('--hidden_size', type=int, required=True, help='the hidden size of model')
    parser.add_argument('--password', type=str, help='the key to encrypt')
    parser.add_argument('--watermark', type=str, help='the contents of watermark')
    parser.add_argument('--gamma_1', type=int, default=2, help='gamma of layer')
    # parser.add_argument('--gamma_2', type=int, default=512, help='gamma of row')
    # parser.add_argument('--gamma_3', type=int, default=2, help='gamma of weight')
    parser.add_argument('--xi', type=int, default=2, choices=[2, 4], help='the last xi bits may be changed. For fp16, xi=4. For int8, xi=2.')
    parser.add_argument('--position_num', type=int, default=6, choices=[6, 12], help='the length of position bits. For fp16, =12. For int8, =6.')
    parser.add_argument('--delta', type=int, default=20, help='the min redundancy of bits to be reversed.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--select_ratio', type=float, default=0.75, help='select select_ratio percent weights as inserting candidata')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the inserted model.')
    parser.add_argument('--calibdation_dataset', type=str, default='c4', choices=["c4", "wikitext2"])
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print_config(args)

    print("*"*30)
    print(f"Loading llm model and tokenizer: {args.model}...")
    model = get_llm(args.model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda")
    # device = torch.device("cpu")
    # if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
    #     device = model.hf_device_map["lm_head"]
    
    print("Insert starts.")
    start_time = time.time()
    change_weight_num, total_weight_num = insert_watermark(args, model, tokenizer, device, dataset_name=args.calibdation_dataset)

    if args.save_model:
        model.save_pretrained(args.save_model, safe_serialization=False)
        tokenizer.save_pretrained(args.save_model)
        print(f'model saved in {args.save_model}.')
    
    print("*"*30)
    print('Insert Done!')
    end_time = time.time()
    elapsed_time = end_time - start_time
    format_time(elapsed_time, "Total insert")
    print(f"Total change weights num: {change_weight_num}\tratio:{change_weight_num/total_weight_num}")

if __name__ == '__main__':
    main()