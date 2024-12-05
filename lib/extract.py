import time 
import torch 
import numpy as np
import hashlib
import tqdm

from .utils import (get_blocks, 
                    find_layers,
                    string_to_binary, 
                    get_bit_from_weight,
                    get_position_bits,
                    get_extract_acc,
                    get_extract_chunk_acc,
                    qweight2weight,
                    format_time
                    )


def extract_watermark(args, model):
    # use_cache = model.config.use_cache 
    # model.config.use_cache = False 
    is_robust = args.mode == 'robust'
    print(f'mode: is_robust={is_robust}')

    layers = get_blocks(model)

    # for extract result
    
    threshold_acc = args.threshold_acc
    threshold_layer_acc = args.threshold_layer_acc
    real_watermark = args.watermark # watermark
    real_watermark_bits = string_to_binary(real_watermark)  # watermark bit string
    L = len(real_watermark_bits) # the length of watermark bit string
    chunk_L = args.chunk_length
    watermarks_from_layers = {'ones':[0]*L, 'zeors':[0]*L}

    gamma_layer = args.gamma_1
    layer_id = 0

    for i in tqdm.tqdm(range(len(layers)), desc="Running OurMark extract..."):
    # for i in tqdm.tqdm(range(1), desc="Running OurMark extract..."):
        layer = layers[i]
        # print(layer)
        subset = find_layers(layer)

        for name in subset:
            ones = [0] * chunk_L
            zeros = [0] * chunk_L
            one_layer_time_start = time.time()
            print(f"[{layer_id}]: {name}: {subset[name]}")

            K_layer = args.password + str(layer_id)
            K_layer = int(hashlib.md5(K_layer.encode()).hexdigest(), 16)
            if is_robust is not True and K_layer % gamma_layer != 0:
                print(f"[{layer_id}]: This layer is skipped.")
                layer_id += 1
                continue

            print(f"==> Extract watermark from weights...")
            
            if "8bit" in args.model:
                weight = qweight2weight(subset[name]).data.t()
            else:
                weight = subset[name].weight.data
            original_weight = weight.cpu()

            ones, zeros = extract_from_weight(args, layer_id, original_weight, ones, zeros, is_robust)
            print(f'ones: {ones}')
            print(f'zeros: {zeros}')

            extracted_chunk_watermark_bits = ''
            for i in range(chunk_L):
                if ones[i] > zeros[i]:
                    extracted_chunk_watermark_bits += '1'
                else:
                    extracted_chunk_watermark_bits += '0'
            print(f'=========[{layer_id}]========[{real_watermark_bits}]')
            layer_acc, chunk_id = get_extract_chunk_acc(real_watermark_bits, extracted_chunk_watermark_bits)
            print(extracted_chunk_watermark_bits)
            print(f"[{layer_id}] acc: {layer_acc}")

            one_layer_time_end = time.time()
            one_layer_time = one_layer_time_end - one_layer_time_start
            format_time(one_layer_time, "Extract of one layer")

            layer_id += 1

            if chunk_id != -1 and layer_acc > threshold_layer_acc:
                for i in range(chunk_L):
                    watermarks_from_layers['ones'][chunk_id * chunk_L + i] += ones[i]
                    watermarks_from_layers['zeors'][chunk_id * chunk_L + i] += zeros[i]
                # watermarks_from_layers.append(extracted_watermark_bits)
                print(f'Current ones: {watermarks_from_layers["ones"]}')
                print(f'Current zeros: {watermarks_from_layers["zeors"]}')
                
                extracted_watermark_bits = ''
                for i in range(L):
                    if watermarks_from_layers['ones'][i] > watermarks_from_layers['zeors'][i]:
                        extracted_watermark_bits += '1'
                    else:
                        extracted_watermark_bits += '0'

                cur_total_acc = get_extract_acc(real_watermark_bits, extracted_watermark_bits)
                print(f"Current acc: {cur_total_acc}")
                if cur_total_acc < threshold_acc:
                    continue
                else:
                    print(f"Real watermark: {real_watermark_bits}")
                    print(f"Extract watermark: {extracted_watermark_bits}")
                    # model.config.use_cache = use_cache 
                    torch.cuda.empty_cache()
                    return cur_total_acc

    # model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

    extracted_watermark_bits = ''

    for i in range(L):
        if watermarks_from_layers['ones'][i] > watermarks_from_layers['zeors'][i]:
            extracted_watermark_bits += '1'
        else:
            extracted_watermark_bits += '0'

    print(f"Real watermark: {real_watermark_bits}")
    print(f"Extract watermark: {extracted_watermark_bits}")
    return get_extract_acc(real_watermark_bits, extracted_watermark_bits)


def extract_from_weight(args, layer_id, original_weight, ones, zeros, is_robust):
    original_weight = original_weight.detach().cpu().numpy()

    chunk_length = args.chunk_length
    K = args.password # password
    # gamma_row = args.gamma_2
    gamma_row = args.hidden_size // 4
    gamma_weight = args.xi
    xi = args.xi

    if is_robust:
        for i in range(original_weight.shape[0]):
            for j in range(original_weight.shape[1]):
                if original_weight[i][j] != 0:
                    position_bits = get_position_bits(original_weight[i][j], args.position_num)
                    K_p = K + position_bits
                    K_p = int(hashlib.md5(K_p.encode()).hexdigest(), 16)
                    rng = np.random.default_rng(seed=K_p)
                    weight_id = rng.integers(0, gamma_weight)
                    if weight_id == 0:
                        insert_position = rng.integers(0, xi)  # which last bit in weight will be change
                        insert_bit_positon = rng.integers(0, chunk_length) # which bit in watermark will be insert
                        inserted_bit = get_bit_from_weight(original_weight[i][j], insert_position)
                        if inserted_bit == '1':
                            ones[insert_bit_positon] += 1
                        else:
                            zeros[insert_bit_positon] += 1 
                else:
                    continue
    else:
        for i in range(original_weight.shape[0]):
            K_row = K + str(layer_id) + str(i)
            K_row = int(hashlib.md5(K_row.encode()).hexdigest(), 16)
            if K_row % gamma_row != 0:
                continue
            for j in range(original_weight.shape[1]):
                if original_weight[i][j] != 0:
                    position_bits = get_position_bits(original_weight[i][j], args.position_num)
                    K_p = K + position_bits
                    K_p = int(hashlib.md5(K_p.encode()).hexdigest(), 16)
                    rng = np.random.default_rng(seed=K_p)
                    weight_id = rng.integers(0, gamma_weight)
                    if weight_id == 0:
                        insert_position = rng.integers(0, xi)  # which last bit in weight will be change
                        insert_bit_positon = rng.integers(0, chunk_length) # which bit in watermark will be insert
                        inserted_bit = get_bit_from_weight(original_weight[i][j], insert_position)
                        if inserted_bit == '1':
                            ones[insert_bit_positon] += 1
                        else:
                            zeros[insert_bit_positon] += 1 
                else:
                    continue

    return ones, zeros