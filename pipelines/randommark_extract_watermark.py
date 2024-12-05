import argparse
import os 
import time

from lib.randommark_extract import extract_watermark
from lib.utils import get_llm, format_time

def print_config(args):
    print(f'Config parameters:')
    print(f"+ model: {args.model}")
    print(f"+ hidden size: {args.hidden_size}")
    print(f"+ watermark: {args.watermark}")
    print(f"+ random seed: {args.seed}")
    # print(f"+ random seed_row: {args.seed_row}")
    # print(f"+ random seed_column: {args.seed_column}")
    # print(f"+ random seed_position: {args.seed_position}")
    # print(f"+ random seed_bit: {args.seed_bit}")
    print(f"+ modify rate: {args.modify_rate}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model path')
    parser.add_argument('--hidden_size', type=int, required=True, help='the hidden size of model')
    parser.add_argument('--watermark', type=str, help='the contents of watermark')
    parser.add_argument('--seed', type=int, default=100, help='Seed to insert.')
    # parser.add_argument('--seed_column', type=int, default=200, help='Seed for randomly choosing weight to modify.')
    # parser.add_argument('--seed_position', type=int, default=300, help='Seed for randomly choosing the position of bit of weight to modify.')
    # parser.add_argument('--seed_bit', type=int, default=400, help='Seed for randomly choosing the position of bit of watermark to be inserted.')
    parser.add_argument('--modify_rate', type=float, default=0.75, help='the rate of hidden size to be modified')
    args = parser.parse_args()

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