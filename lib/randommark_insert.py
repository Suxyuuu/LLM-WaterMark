import time 
import torch
import torch.nn as nn
import numpy as np
import tqdm
import gc

from .utils import (get_blocks, 
                    find_layers, 
                    string_to_binary, 
                    modify_bit_of_weight, 
                    qweight2weight, 
                    weight2qweight,
                    format_time
                    )

def insert_watermark(args, model, device=torch.device("cuda")):
    watermark = args.watermark
    watermark_bits = string_to_binary(watermark)
    watermark_length = len(watermark_bits)
    seed_row = args.seed
    seed_column = seed_row + 1
    seed_position = seed_column + 2
    seed_bit = seed_position + 3
    rng_row = np.random.default_rng(seed=seed_row)
    rng_column = np.random.default_rng(seed=seed_column)
    rng_position = np.random.default_rng(seed=seed_position)
    rng_bit = np.random.default_rng(seed=seed_bit)
    if '8bit' in args.model:
        weight_length = 8
    else:
        weight_length = 16
    # print(weight_length)

    # use_cache = model.config.use_cache 
    # model.config.use_cache = False 

    layers = get_blocks(model)
    layer_num = 0
    for i in range(len(layers)):
        temp_layer = layers[i]
        named_linears = find_layers(temp_layer)
        layer_num += len(named_linears)
    every_layer_insert_bits_num = watermark_length // layer_num + 1 # how many bits of watermark insert into every layer
    modify_num = int(args.hidden_size * args.modify_rate)
    every_layer_modify_weight_num_per_bit = modify_num // every_layer_insert_bits_num  # how many weights will be modified every layer for per bit
    print(f'every_layer_insert_bits_num: {every_layer_insert_bits_num}')
    
    change_weight_num = 0
    total_weight_num = 0
    layer_id = 0
    
    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running RandomMark insert..."):
    # for i in tqdm.tqdm(range(1), desc="Running RandomMark insert..."):
        layer = layers[i]
        # print(layer)
        layer = layer.to(device)
        named_linears = find_layers(layer)
        for name in named_linears:
            one_layer_time_start = time.time()
            print(f"[{layer_id}]: {name}: {named_linears[name]}")
            print(f"==> Insert watermark into weights...")
            if "8bit" in args.model:
                # print(named_linears[name].qweight)
                weight = qweight2weight(named_linears[name]).data.t()
            else:
                weight = named_linears[name].weight.data
            # print(weight)
            original_weight = weight.cpu()
            # original_weight = named_linears[name].weight.data
            original_weight_np = original_weight.detach().cpu().numpy()
            modified_weight_np = original_weight_np.copy()

            for i in range(every_layer_insert_bits_num):
                insert_bit_position = rng_bit.integers(0, watermark_length)
                insert_bit = watermark_bits[insert_bit_position]
                for j in range(every_layer_modify_weight_num_per_bit):
                    row = rng_row.integers(0, original_weight_np.shape[0])
                    column = rng_column.integers(0, original_weight_np.shape[1])
                    insert_position = rng_position.integers(0, 10)
                    # print(original_weight_np[row][column])
                    # print(f'[{row}, {column}]')
                    modified_weight_np[row][column] = modify_bit_of_weight(original_weight_np[row][column], insert_bit, insert_position)
                    # print(modified_weight_np[row][column])
                    # print(f'[{row}, {column}]: {insert_position} -> {insert_bit} from [{insert_bit_position}]')

            weight_count = original_weight.shape[0] * original_weight.shape[1]
            modified_weight_tensor =  torch.tensor(modified_weight_np, dtype=original_weight.dtype)
            # print(modified_weight_tensor)
            if "8bit" in args.model:
                modified_qweight = weight2qweight(modified_weight_tensor.t(), named_linears[name])
                # print(modified_qweight)
                named_linears[name].qweight.data = modified_qweight
                # print(named_linears[name].qweight.data)
            else:
                named_linears[name].weight.data = modified_weight_tensor
            
            # named_linears[name].weight.data = torch.tensor(modified_weight_np, device='cuda', dtype=original_weight.dtype)
            difference = modified_weight_tensor - weight.to(modified_weight_tensor.device)
            non_zero_count = torch.count_nonzero(difference)
            print(f'==> Modify num: {non_zero_count}\tratio:{non_zero_count/weight_count}')
            change_weight_num += non_zero_count
            total_weight_num += weight_count
            
            one_layer_time_end = time.time()
            one_layer_time = one_layer_time_end - one_layer_time_start
            format_time(one_layer_time, "Insert of one layer")

            layer_id += 1

        gc.collect()

    # model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    return change_weight_num, total_weight_num
