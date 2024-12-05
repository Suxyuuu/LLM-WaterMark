import time 
import torch 
import numpy as np
import hashlib
import tqdm

from .data import get_loaders 
from .layerwrapper import WrappedGPT
from .utils import (prepare_calibration_input,
                    get_blocks, 
                    find_layers,
                    string_to_binary, 
                    modify_bit_of_weight,
                    get_position_bits,
                    weight2qweight,
                    qweight2weight,
                    get_bit_from_weight,
                    format_time
                    )


def insert_watermark(args, model, tokenizer, device, dataset_name='wikitext2'):
    # use_cache = model.config.use_cache 
    # model.config.use_cache = False 

    # if "model.embed_tokens" in model.hf_device_map:
    #     device = model.hf_device_map["model.embed_tokens"]

    print("Loading calibdation data...")
    dataloader, _ = get_loaders(dataset_name,nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    layers = get_blocks(model)
    with torch.no_grad():    
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args)
    # print(inps)

    inps.to(device)
    outs.to(device)
    attention_mask.to(device)
    if position_ids != None:
        position_ids.to(device)

    gamma_layer = args.gamma_1
    layer_id = 0
    insert_layer_num = 0
    change_weight_num = 0
    total_weight_num = 0

    total_preprocess_time = 0.0
    total_insert_time = 0.0

    for i in tqdm.tqdm(range(len(layers)), desc="Running OurMark insert..."):
    # for i in tqdm.tqdm(range(1), desc="Running OurMark insert..."):
        layer = layers[i]
        layer.to(device)
        # print(layer)
        subset = find_layers(layer)

        preprocess_time_start = time.time()
        
        wrapped_layers = {}
        for name in subset:
            # print(f'name: {name}')
            if "8bit" in args.model:
                wrapped_layers[name] = WrappedGPT(subset[name], layer_id, name, True)
            else:
                wrapped_layers[name] = WrappedGPT(subset[name], layer_id, name)

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        for j in range(args.nsamples):
        # for j in range(1):
            with torch.no_grad():
                if position_ids != None:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()
        
        preprocess_time_end = time.time()
        preprocess_time = preprocess_time_end - preprocess_time_start
        total_preprocess_time += preprocess_time
        
        for name in subset:
            print(f"[{layer_id}]: {name}: {subset[name]}")
            K_layer = args.password + str(layer_id)
            K_layer = int(hashlib.md5(K_layer.encode()).hexdigest(), 16)
        
            if K_layer % gamma_layer != 0:
                print(f"[{layer_id}]: This layer is skipped.")
                layer_id += 1
                continue

            print(f'==> Get socres of weights...')
            preprocess_time_start = time.time()
            if "8bit" in args.model:
                weight = qweight2weight(subset[name]).data.t()
            else:
                weight = subset[name].weight.data
            original_weight = weight.cpu()
            W_metric = torch.abs(original_weight) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))).to(original_weight.device)
            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:,:int(W_metric.shape[1]*args.select_ratio)]
            W_mask.scatter_(1, indices, True)

            preprocess_time_end = time.time()
            preprocess_time = preprocess_time_end - preprocess_time_start
            total_preprocess_time += preprocess_time

            print(f"==> Insert watermark into weights...")
            insert_time_start = time.time()
            modified_weight = change_weight(args, layer_id, insert_layer_num, original_weight, W_mask)
            modified_weight_tensor =  torch.tensor(modified_weight, dtype=original_weight.dtype)

            if "8bit" in args.model:
                modified_qweight = weight2qweight(modified_weight_tensor.t(), subset[name])
                # print(subset[name].qweight.data.shape)
                subset[name].qweight.data = modified_qweight
                # print(subset[name].qweight.data.shape)
            else:
                subset[name].weight.data = modified_weight_tensor
            
            # modify result
            difference = modified_weight_tensor - weight.to(modified_weight_tensor.device)
            non_zero_count = torch.count_nonzero(difference)
            weight_count = original_weight.shape[0]*original_weight.shape[1]
            print(f'==> Modify num: {non_zero_count}\tratio:{non_zero_count/weight_count}')
            
            insert_time_end = time.time()
            insert_time = insert_time_end - insert_time_start
            total_insert_time += insert_time
            
            change_weight_num += non_zero_count
            total_weight_num += weight_count
            layer_id += 1
            insert_layer_num += 1

        # for j in range(args.nsamples):
        #     with torch.no_grad():
        #         if position_ids != None:
        #             outs[j] = layer(inps[j].unsqueeze(0).to(device), attention_mask=attention_mask, position_ids=position_ids)[0]
        #         else:
        #             outs[j] = layer(inps[j].unsqueeze(0).to(device), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

    format_time(total_preprocess_time, 'Preprocess')
    format_time(total_insert_time, 'Insert')

    # model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    return change_weight_num, total_weight_num


def change_weight(args, layer_id, insert_layer_num, original_weight, W_mask):
    # print(f'[insert layer num]: {insert_layer_num}')
    original_weight = original_weight.detach().cpu().numpy()
    W = args.watermark # watermark
    chunk_id = insert_layer_num % len(W)
    chunk = W[chunk_id]
    W = string_to_binary(chunk)  # watermark bit string
    print(f'=========[{layer_id}]========[{W}]')
    L = len(W) # the length of watermark bit string
    K = args.password # password
    # gamma_row = args.gamma_2
    gamma_row = args.hidden_size // 4
    gamma_weight = args.xi
    xi = args.xi

    # 1. get the one-zero distribution of positions to be inserted
    zeros = [0] * L
    ones = [0] * L
    for i in range(original_weight.shape[0]):
        K_row = K + str(layer_id) + str(i)
        K_row = int(hashlib.md5(K_row.encode()).hexdigest(), 16)
        if K_row % gamma_row != 0:
            continue
        for j in range(original_weight.shape[1]):
            # if W_mask[i][j]:
            position_bits = get_position_bits(original_weight[i][j], args.position_num)
            K_p = K + position_bits
            K_p = int(hashlib.md5(K_p.encode()).hexdigest(), 16)
            rng = np.random.default_rng(seed=K_p)
            if rng.integers(0, gamma_weight) % gamma_weight == 0:
                insert_position = rng.integers(0, xi) % xi  # which last bit in weigth will be change
                insert_bit_positon = rng.integers(0, L) % L
                original_weight_bit = get_bit_from_weight(original_weight[i][j], insert_position)
                if original_weight_bit == '1':
                    ones[insert_bit_positon] += 1
                else:
                    zeros[insert_bit_positon] += 1
    
    print(f'current distribution: ')
    print(f'ones: {ones}')
    print(f'zeros: {zeros}')

    # 2. calculate reverse bits num
    reverse_bits_list = [0] * L
    for _ in range(L):
        difference = abs(ones[_] - zeros[_]) // 2
        if W[_] == "0":
            if ones[_] > zeros[_]:
                reverse_bits_list[_] += difference
                delta = (zeros[_] + difference) // 10
                delta = max(delta, args.delta)
                reverse_bits_list[_] += delta
            else:
                more = zeros[_] - ones[_]
                delta = (ones[_]) // 10
                delta = max(delta, args.delta)
                if more < delta:
                    reverse_bits_list[_] += delta - more

        else:
            if ones[_] < zeros[_]:
                reverse_bits_list[_] += difference
                delta = (ones[_] + difference) // 10
                delta = max(delta, args.delta)
                reverse_bits_list[_] += delta
            else:
                more = ones[_] - zeros[_]
                delta = (zeros[_]) // 10
                delta = max(delta, args.delta)
                if more < delta:
                    reverse_bits_list[_] += delta - more

    print(reverse_bits_list)


    # 3. begin reverse
    for i in range(original_weight.shape[0]):
        K_row = K + str(layer_id) + str(i)
        K_row = int(hashlib.md5(K_row.encode()).hexdigest(), 16)
        if K_row % gamma_row != 0:
            continue
        for j in range(original_weight.shape[1]):
            if W_mask[i][j]:
                position_bits = get_position_bits(original_weight[i][j], args.position_num)
                K_p = K + position_bits
                K_p = int(hashlib.md5(K_p.encode()).hexdigest(), 16)
                rng = np.random.default_rng(seed=K_p)
                if rng.integers(0, gamma_weight) % gamma_weight == 0:
                    insert_position = rng.integers(0, xi) % xi  # which last bit in weigth will be change
                    insert_bit_positon = rng.integers(0, L) % L # which bit in watermark will be insert

                    if sum(reverse_bits_list) == 0:
                        return original_weight
                    if reverse_bits_list[insert_bit_positon] == 0:
                        continue
                    insert_bit = W[insert_bit_positon]
                    # print(f'input weight: {original_weight[i][j]}')
                    original_weight_bit = get_bit_from_weight(original_weight[i][j], insert_position)
                    if original_weight_bit == insert_bit:
                        continue
                    original_weight[i][j] = modify_bit_of_weight(original_weight[i][j], insert_bit, insert_position)
                    reverse_bits_list[insert_bit_positon] -= 1
                    # print(f'output weight: {original_weight[i][j]}')
                    
            else:
                continue

    return original_weight