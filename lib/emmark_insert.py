import time 
import torch
import torch.nn as nn
import numpy as np
import tqdm
import gc
import functools
from collections import defaultdict
from datasets import load_dataset

from .data import get_loaders 
from .utils import (get_blocks, 
                    prepare_calibration_input,
                    find_layers, 
                    string_to_binary,
                    qweight2weight,
                    weight2qweight,
                    format_time
                    )


def insert_watermark(args, model, tokenizer, device=torch.device("cuda"), dataset_name='c4'):
    watermark = args.watermark
    watermark_bits = string_to_binary(watermark)
    watermark_length = len(watermark_bits)
    seed = args.seed
    rng = np.random.default_rng(seed=seed)

    # use_cache = model.config.use_cache 
    # model.config.use_cache = False 

    print("Loading calibdation data...")
    dataloader, _ = get_loaders(dataset_name,nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args)
    
    inps.to(device)
    outs.to(device)
    attention_mask.to(device)
    if position_ids != None:
        position_ids.to(device)

    blocks = get_blocks(model)
    layer_num = 0
    for i in range(len(blocks)):
        temp_block = blocks[i]
        named_layers = find_layers(temp_block)
        layer_num += len(named_layers)
    every_layer_insert_bits_num = watermark_length // layer_num + 1 # how many bits of watermark insert into every layer
    modify_num = int(args.hidden_size * args.modify_rate)
    every_layer_modify_weight_num_per_bit = modify_num // every_layer_insert_bits_num  # how many weights will be modified every layer for per bit
    print(f'every_layer_insert_bits_num: {every_layer_insert_bits_num}')
    every_layer_insert_candidate_bits_num = modify_num * args.candidate_rate
    
    insert_bit_position = 0
    change_weight_num = 0
    total_weight_num = 0
    total_indices = torch.empty((0, every_layer_insert_candidate_bits_num, 2)).to(device)
    layer_id = 0

    total_preprocess_time = 0.0
    total_insert_time = 0.0
    
    # solve layer by layer
    for i in tqdm.tqdm(range(len(blocks)), desc="Running EmMark insert..."):
        block = blocks[i]
        # print(layer)
        block = block.to(device)
        named_layers = find_layers(block)

        preprocess_time_start = time.time()

        def cache_output_hook(module, input, output, name, feat_dict):
            if len(output.shape) == 3:
                output = output[0]
            # print(output)
            output = output.clone().detach()
            feat_dict[name].append(output)

        output_feat = defaultdict(list)
        handles = []
        for name in named_layers:
            handles.append(
                named_layers[name].register_forward_hook(
                    functools.partial(cache_output_hook, name=name, feat_dict=output_feat)
                )
            )
        for j in range(args.nsamples):
            with torch.no_grad():
                if position_ids != None:
                    outs[j] = block(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                else:
                    outs[j] = block(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        output_feat = {k: torch.mean(torch.stack(v, dim=0), dim=0) for k, v in output_feat.items()}
        
        preprocess_time_end = time.time()
        preprocess_time = preprocess_time_end - preprocess_time_start
        total_preprocess_time += preprocess_time

        named_layers = find_layers(block)
        for name in named_layers:
            weight = qweight2weight(named_layers[name]).data

            # print(weight)
            one_layer_time_start = time.time()
            print(f"[{layer_id}]: {name}: {named_layers[name]}")
            print(f'==> Get socres of weights...')
            preprocess_time_start = time.time()
            act = output_feat[name].to(device)

            S_q = torch.abs(1/weight)
            S_r = torch.abs(act.max()/(act-act.min()))
            if S_r.shape[0] != S_q.shape[0]:
                S = S_q
            else:
                S = S_q + S_r

            flattened_S = S.view(-1)
            values, flat_indices = torch.topk(flattened_S, every_layer_insert_candidate_bits_num, largest=False)
            indices = torch.unravel_index(flat_indices, S.shape)
            indices = torch.concat(indices, dim=0).view(2, -1).T
            total_indices = torch.cat((total_indices, indices.unsqueeze(0)), dim=0)

            preprocess_time_end = time.time()
            preprocess_time = preprocess_time_end - preprocess_time_start
            total_preprocess_time += preprocess_time

            print(f"==> Insert watermark into weights...")
            insert_time_start = time.time()
            original_weight = weight.cpu()
            modified_weight = original_weight.detach().cpu().numpy()
            
            for i in range(every_layer_insert_bits_num):
                insert_bit = watermark_bits[(insert_bit_position + i) % watermark_length]
                for j in range(every_layer_modify_weight_num_per_bit):
                    _ = rng.integers(0, every_layer_insert_candidate_bits_num)
                    indice = indices[_]
                    # print(indice)
                    if insert_bit == '0':
                        modified_weight[indice[0], indice[1]] += -1
                    else:
                        modified_weight[indice[0], indice[1]] += 1
            insert_bit_position += every_layer_insert_bits_num
            insert_bit_position = insert_bit_position % watermark_length

            weight_count = S.shape[0] * S.shape[1]
            
            modified_weight_tensor = torch.tensor(modified_weight, dtype=weight.dtype)
            modified_qweight = weight2qweight(modified_weight_tensor, named_layers[name])
            named_layers[name].qweight.data = modified_qweight
            difference = modified_weight_tensor - weight.cpu()
            non_zero_count = torch.count_nonzero(difference)
            print(f'==> Modify num: {non_zero_count}\tratio:{non_zero_count/weight_count}')

            insert_time_end = time.time()
            insert_time = insert_time_end - insert_time_start
            total_insert_time += insert_time

            change_weight_num += non_zero_count
            total_weight_num += weight_count
            layer_id += 1
        
        # print(outs)
        # for j in range(args.nsamples):
        #     with torch.no_grad():
        #         if position_ids != None:
        #             outs[j] = block(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        #         else:
        #             outs[j] = block(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        # print(outs)
        inps, outs = outs, inps
        
        del output_feat
        gc.collect()

    format_time(total_preprocess_time, 'Preprocess')
    format_time(total_insert_time, 'Insert')
    # model.config.use_cache = use_cache 
    # torch.cuda.empty_cache()
    return change_weight_num, total_weight_num, total_indices
