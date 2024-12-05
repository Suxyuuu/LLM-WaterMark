import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
import numpy as np
import torch
from transformers import AutoTokenizer

from lib.emmark_insert import insert_watermark
from lib.utils import get_llm, format_time



def print_config(args):
    print(f'Config parameters:')
    print(f"+ model: {args.model}")
    print(f"+ hidden size: {args.hidden_size}")
    print(f"+ watermark: {args.watermark}")
    print(f"+ calibdation data: {args.calibdation_dataset}")
    print(f"+ num of samples: {args.nsamples}")
    print(f"+ random seed: {args.seed}")
    print(f"+ candidate rate: {args.candidate_rate}")
    print(f"+ modify rate: {args.modify_rate}")
    print(f"+ save model path: {args.save_model}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model path')
    parser.add_argument('--hidden_size', type=int, required=True, help='the hidden size of model')
    parser.add_argument('--watermark', type=str, help='the contents of watermark')
    parser.add_argument('--seed', type=int, default=100, help='Seed for choosing weight to modify.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--candidate_rate', type=int, default=60, help='|B_c|/(|B|/n). 50: model_size < 6.7b; 60: model_size >= 6.7b')
    parser.add_argument('--modify_rate', type=float, default=0.75, help='the rate of hidden size to be modified')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the inserted model and candidate indices.pt.')
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
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    device = torch.device("cuda")
    # if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
    #     device = model.hf_device_map["lm_head"]
    
    print("Insert starts.")
    start_time = time.time()
    change_weight_num, total_weight_num, total_indices = insert_watermark(args, model, tokenizer, device, dataset_name=args.calibdation_dataset)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        total_indices = total_indices.to(torch.int32)
        torch.save(total_indices, args.save_model + '/total_indices.pt')
        # print(total_indices)
        print(f'model saved in {args.save_model}.')
    
    print("*"*30)
    print('Insert Done!')
    end_time = time.time()
    elapsed_time = end_time - start_time
    format_time(elapsed_time, "Total insert")
    print(f"Total change weights num: {change_weight_num}\tratio:{change_weight_num/total_weight_num}")

if __name__ == '__main__':
    main()