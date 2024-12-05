import argparse
import os 
import time
import numpy as np
import torch

from lib.emmark_extract import extract_watermark
from lib.utils import get_llm, format_time

def print_config(args):
    print(f'Config parameters:')
    print(f"+ origin model: {args.model}")
    print(f"+ inserted model: {args.inserted_model}")
    print(f"+ hidden size: {args.hidden_size}")
    print(f"+ watermark: {args.watermark}")
    print(f"+ random seed: {args.seed}")
    print(f"+ candidate rate: {args.candidate_rate}")
    print(f"+ modify rate: {args.modify_rate}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model path')
    parser.add_argument('--inserted_model', type=str, help='model path')
    parser.add_argument('--hidden_size', type=int, required=True, help='the hidden size of model')
    parser.add_argument('--watermark', type=str, help='the contents of watermark')
    parser.add_argument('--seed', type=int, default=100, help='Seed for choosing weight to modify.')
    parser.add_argument('--candidate_rate', type=int, default=60, help='|B_c|/(|B|/n). 50: model_size < 6.7b; 60: model_size >= 6.7b')
    parser.add_argument('--modify_rate', type=float, default=0.75, help='the rate of hidden size to be modified')
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print_config(args)

    print("*"*30)
    print(f"Loading llm model and tokenizer: {args.model}...")
    model = get_llm(args.model)
    model.eval()
    print(f"Loading llm model and tokenizer: {args.inserted_model}...")
    inserted_model = get_llm(args.inserted_model)
    inserted_model.eval()

    indices = torch.load(args.inserted_model + '/total_indices.pt')
    # print(indices)
    # tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # device = torch.device("cuda:0")
    # if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
    #     device = model.hf_device_map["lm_head"]
    
    print("Extract starts.")
    start_time = time.time()
    extracted_watermark_acc = extract_watermark(args, model, inserted_model, indices)
    print("*"*30)
    print('Extract Done!')
    print(f'Extract ACC: {extracted_watermark_acc}')
    end_time = time.time()
    elapsed_time = end_time - start_time
    format_time(elapsed_time, "Total extract")
    

if __name__ == '__main__':
    main()