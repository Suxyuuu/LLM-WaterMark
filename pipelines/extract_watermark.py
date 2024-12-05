import argparse
import os 
import time
import numpy as np
import torch

from lib.extract import extract_watermark
from lib.utils import get_llm, format_time

def print_config(args):
    print(f'Config parameters:')
    print(f"+ watermark: {args.watermark}")
    print(f"+ chunk_length: {args.chunk_length}")
    print(f"+ password: {args.password}")
    print(f"+ random seed: {args.seed}")
    print(f"+ gamma_layer: {args.gamma_1}")
    # print(f"+ gamma_row: {args.gamma_2}")
    # print(f"+ gamma_weight: {args.gamma_3}")
    print(f"+ xi: {args.xi}")
    print(f"+ position_num: {args.position_num}")
    print(f"+ extract mode: {args.mode}")
    print(f"+ threshold_layer_acc: {args.threshold_layer_acc}")
    print(f"+ threshold_acc: {args.threshold_acc}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model path')
    parser.add_argument('--hidden_size', type=int, default=2, help='the hidden size of model')
    parser.add_argument('--password', type=str, help='the key to encrypt')
    parser.add_argument('--watermark', type=str, help='the contents of watermark')
    parser.add_argument('--chunk_length', type=int, default=8, choices=[8], help='watermark will be split into chunks.')
    parser.add_argument('--gamma_1', type=int, default=2, help='gamma of layer')
    # parser.add_argument('--gamma_2', type=int, default=512, help='gamma of row')
    # parser.add_argument('--gamma_3', type=int, default=2, help='gamma of weight')
    parser.add_argument('--xi', type=int, default=2, choices=[2, 4], help='the last xi bits may be changed. For fp16, xi=4. For int8, xi=2.')
    parser.add_argument('--position_num', type=int, default=6, choices=[6, 12], help='the length of position bits. For fp16, =12. For int8, =6.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for sampling the calibration data.')
    parser.add_argument('--mode', type=str, default='simple', choices=['simple', 'robust'], help='if the model has not been tuning, you can choose simple mode to extract watermark faster.')
    parser.add_argument('--threshold_layer_acc', type=float, default=0.70, help='qualified layer needs its watermark acc more than this threshold')
    parser.add_argument('--threshold_acc', type=float, default=0.99, help='if the acc more than this threshold, the extraction will be stopped.')
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print_config(args)

    print("*"*30)
    print(f"Loading llm model and tokenizer: {args.model}...")
    model = get_llm(args.model)
    model.eval()
    # tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # device = torch.device("cuda:0")
    # if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
    #     device = model.hf_device_map["lm_head"]
    
    print("Extract starts.")
    start_time = time.time()
    
    extracted_watermark_acc = extract_watermark(args, model)
    
    print("*"*30)
    print('Extract Done!')
    print(f'Extract ACC: {extracted_watermark_acc}')
    end_time = time.time()
    elapsed_time = end_time - start_time
    format_time(elapsed_time, "Total extract")
    

if __name__ == '__main__':
    main()