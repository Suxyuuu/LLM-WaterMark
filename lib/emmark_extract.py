import time 
import torch 
import numpy as np
import hashlib
import tqdm

from .utils import (get_blocks, 
                    find_layers,
                    string_to_binary, 
                    get_extract_acc,
                    qweight2weight,
                    format_time
                    )


def extract_watermark(args, model, inserted_model, indices):
    real_watermark = args.watermark
    real_watermark_bits = string_to_binary(real_watermark)
    real_watermark_length = len(real_watermark_bits)
    seed = args.seed
    rng = np.random.default_rng(seed=seed)

    # use_cache = model.config.use_cache 
    # model.config.use_cache = False
    # use_cache = inserted_model.config.use_cache 
    # inserted_model.config.use_cache = False 

    layers = get_blocks(model)
    inserted_layers = get_blocks(inserted_model)

    total_layer_num_of_model = 0
    # total_layer_num_of_inserted_model = 0
    for i in range(len(layers)):
        temp_layer = layers[i]
        temp_layer = temp_layer.cuda()
        named_linears = find_layers(temp_layer)
        total_layer_num_of_model += len(named_linears)
    # for i in range(len(inserted_layers)):
    #     layer = inserted_layers[i]
    #     layer = layer.cuda()
    #     named_linears = find_layers(layer)
    #     total_layer_num_of_inserted_model += len(named_linears)

    every_layer_insert_bits_num = real_watermark_length // total_layer_num_of_model + 1 # how many bits of watermark insert into every layer
    modify_num = int(args.hidden_size * args.modify_rate)
    every_layer_modify_weight_num_per_bit = modify_num // every_layer_insert_bits_num  # how many weights will be modified every layer for per bit
    print(f'every_layer_insert_bits_num: {every_layer_insert_bits_num}')
    every_layer_insert_candidate_bits_num = modify_num * args.candidate_rate

    # for extract result
    extract_watermarks = [0] * real_watermark_length
    insert_bit_position = 0
    layer_id = 0

    for i in tqdm.tqdm(range(len(layers)), desc="Running EmMark extract..."):
        layer = layers[i]
        inserted_layer = inserted_layers[i]
        subset = find_layers(layer)
        inserted_subset = find_layers(inserted_layer)

        for name in subset:
            original_weight = qweight2weight(subset[name]).data
            inserted_weight = qweight2weight(inserted_subset[name]).data
            # print(name)
            # if original_weight.shape[0] > original_weight.shape[1]:
            #     continue
            one_layer_time_start = time.time()
            print(f"[{layer_id}]: {name}: {subset[name]}")

            layer_indices = indices[layer_id]
            layer_indices = layer_indices.view(-1, 2)

            print(f"==> Extract watermark from weights...")
            # original_weight = subset[name].weight.data
            # inserted_weight = inserted_subset[name].weight.data
            
            for i in range(every_layer_insert_bits_num):
                one = 0
                zero = 0
                for j in range(every_layer_modify_weight_num_per_bit):
                    _ = rng.integers(0, every_layer_insert_candidate_bits_num)
                    indice = layer_indices[_]
                    indice = indice.to(torch.int)
                    # print(indice)
                    delta = inserted_weight[indice[0], indice[1]] - original_weight[indice[0], indice[1]]
                    # print(f'[{indice}]: {inserted_weight[indice[0], indice[1]]} - {original_weight[indice[0], indice[1]]} = {delta}')
                    if delta == -1:
                        zero += 1
                    elif delta == 1:
                        one += 1
                # print(f'one:{one}\tzero:{zero}')
                if one > zero:
                    extract_watermarks[insert_bit_position] = 1
                else:
                    extract_watermarks[insert_bit_position] = 0
                # print(extract_watermarks)
                insert_bit_position += 1
                insert_bit_position = insert_bit_position % real_watermark_length
                    
            one_layer_time_end = time.time()
            one_layer_time = one_layer_time_end - one_layer_time_start
            format_time(one_layer_time, "Extract of one layer")
            
            layer_id += 1

    # model.config.use_cache = use_cache
    # inserted_model.config.use_cache = use_cache
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


