import time 
import torch 
import numpy as np
import hashlib
import tqdm

from .utils import (get_blocks, 
                    find_layers, 
                    string_to_binary, 
                    get_extract_acc, 
                    get_bit_from_weight,
                    qweight2weight,
                    format_time
                    )


def extract_watermark(args, model):
    real_watermark = args.watermark
    real_watermark_bits = string_to_binary(real_watermark)
    real_watermark_length = len(real_watermark_bits)
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

    # use_cache = model.config.use_cache 
    # model.config.use_cache = False

    layers = get_blocks(model)
    total_layer_num_of_model = 0
    for i in range(len(layers)):
        temp_layer = layers[i]
        temp_subset = find_layers(temp_layer)
        total_layer_num_of_model += len(temp_subset)

    every_layer_insert_bits_num = real_watermark_length // total_layer_num_of_model + 1 # how many bits of watermark insert into every layer
    modify_num = int(args.hidden_size * args.modify_rate)
    every_layer_modify_weight_num_per_bit = modify_num // every_layer_insert_bits_num  # how many weights will be modified every layer for per bit
    print(f'every_layer_insert_bits_num: {every_layer_insert_bits_num}')

    # for extract result
    extract_watermarks = [0] * real_watermark_length
    layer_id = 0

    for i in tqdm.tqdm(range(len(layers)), desc="Running RandomMark extract..."):
    # for i in tqdm.tqdm(range(1), desc="Running RandomMark extract..."):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            one_layer_time_start = time.time()
            print(f"[{layer_id}]: {name}: {subset[name]}")
            print(f"==> Extract watermark from weights...")
            if "8bit" in args.model:
                # print(subset[name].qweight)
                weight = qweight2weight(subset[name]).data.t()
            else:
                weight = subset[name].weight.data
            # print(weight)
            weight = weight.detach().cpu().numpy()

            for i in range(every_layer_insert_bits_num):
                one = 0
                zero = 0
                insert_bit_position = rng_bit.integers(0, real_watermark_length)
                for j in range(every_layer_modify_weight_num_per_bit):
                    
                    row = rng_row.integers(0, weight.shape[0])
                    column = rng_column.integers(0, weight.shape[1])
                    insert_position = rng_position.integers(0, weight_length)
                    # print(f'[{row}, {column}]')
                    # print(weight[row][column])
                    inserted_bit = get_bit_from_weight(weight[row][column], insert_position)
                    # print(f'[{row}, {column}]: {insert_position} -> {inserted_bit}')
                    if inserted_bit == "0":
                        zero += 1
                    elif inserted_bit == "1":
                        one += 1
                # print(f'one:{one}\tzero:{zero}\t{insert_bit_position}')
                
                if one > zero:
                    extract_watermarks[insert_bit_position] = 1
                else:
                    extract_watermarks[insert_bit_position] = 0
                    
            one_layer_time_end = time.time()
            one_layer_time = one_layer_time_end - one_layer_time_start
            format_time(one_layer_time, "Extract of one layer")
            
            layer_id += 1

    # model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    extracted_watermark_bits = ''
    for i in range(real_watermark_length):
        if extract_watermarks[i] == 0:
            extracted_watermark_bits += '0'
        else:
            extracted_watermark_bits += '1'

    print(f"Real watermark: {real_watermark_bits}")
    print(f"Extract watermark: {extracted_watermark_bits}")
    return get_extract_acc(real_watermark_bits, extracted_watermark_bits)


