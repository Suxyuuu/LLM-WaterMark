import torch 
import torch.nn as nn 
import numpy as np
from bitsandbytes.nn import Linear8bitLt
from transformers import AutoModelForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear
import traceback

def get_llm(model_name):
    if "8bit" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16, 
            device_map="auto"
        )

    model.seqlen = model.config.max_position_embeddings 
    return model

def prepare_calibration_input(model, dataloader, device, args):
    # use_cache = model.config.use_cache
    # model.config.use_cache = False
    layers = get_blocks(model)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if not isinstance(model, OPTForCausalLM):
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    # model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def get_blocks(model):
    if isinstance(model, LlamaForCausalLM) or isinstance(model, Phi3ForCausalLM):
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    else:
        raise NotImplementedError(type(model))
    return layers
    

# get every layer recursively
def find_layers(module, layers=[nn.Linear, Linear8bitLt, QuantLinear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            device
        )
    else:
        raise NotImplementedError(type(model))


# get the first 6 bits of weigth for position
def get_position_bits(weight, num):
    binary_representation = value_to_binary(weight)
    return binary_representation[:num]


# modify the last position-th bit of weight to insert_bit
def modify_bit_of_weight(weight, insert_bit, position):
    # print(f'weight\tinsert_bit\tposition')
    # print(f'{weight}\t{insert_bit}\t{position}')
    binary_representation = value_to_binary(weight)
    # print(f'{binary_representation}')
    index = -(position+1)
    if index == -1:
        binary_representation = binary_representation[:index] + str(insert_bit)
    else:
        binary_representation = binary_representation[:index] + str(insert_bit) + binary_representation[index+1:]

    # print(f'{binary_representation}')
    modified_weight = binary_to_value(binary_representation, weight.dtype)
    # print(f'modified_weight: {modified_weight}')
    return modified_weight


# convert binary_representation to value
def binary_to_value(binary_string, dtype):
    # print(dtype)
    # if dtype == np.uint8:
    #     # print('special processing')
    #     unsigned_int = int(binary_string, 2)
    #     return np.array([unsigned_int], dtype=dtype)
    # else:
    
    # 计算dtype的字节数
    byte_size = dtype.itemsize
    # 将二进制字符串转换为整数
    int_val = int(binary_string, 2)
    # 将整数转换为指定字节数的字节序列
    bytes_val = int_val.to_bytes(byte_size, byteorder='little')
    # return np.frombuffer(bytes_val, dtype=dtype)[0]
    # 将字节序列转换回原始的数据类型
    if np.issubdtype(dtype, np.floating):
        # 如果目标类型是浮点数
        return np.frombuffer(bytes_val, dtype=dtype)[0]
    elif np.issubdtype(dtype, np.integer):
        # 如果目标类型是整数
        return np.array(int.from_bytes(bytes_val, byteorder='little'), dtype=dtype)
    else:
        raise ValueError("Unsupported dtype")


def value_to_binary(weight):
    dtype = weight.dtype
    byte_size = dtype.itemsize
    # 将 weight 转换为 bytes，然后解释为一个整数
    weight_bytes = weight.tobytes()
    weight_int = int.from_bytes(weight_bytes, 'little')
    binary_representation = bin(weight_int)[2:].zfill(byte_size*8)
    # print(f'{weight}: {binary_representation}')
    return binary_representation


def get_bit_from_weight(weight, position):
    binary_representation = value_to_binary(weight)
    bit_value = binary_representation[-(position+1)]
    # print(f'weight\textract_bit\tposition')
    # print(f'{weight}\t{bit_value}\t{position}')
    # print(f'{binary_representation}')
    return bit_value


# convert string to binary representation
def string_to_binary(data):
    # 转换每个字符为其对应的ASCII值的8位二进制形式
    binary_stream = ''.join(format(ord(char), '08b') for char in data)
    return binary_stream


# calculate extract acc
def get_extract_acc(watermark_bits, extracted_watermark_bits):
    hit_num = 0
    for i in range(len(watermark_bits)):
        if watermark_bits[i] == extracted_watermark_bits[i]:
            hit_num += 1
    rate = hit_num/len(watermark_bits)
    return rate

def get_extract_chunk_acc(watermark_bits, extracted_watermark_bits):
    max_acc = 0
    max_acc_id = -1
    for chunk_id in range(len(watermark_bits) // 8):
        hit_num = 0
        for i in range(8):
            if watermark_bits[chunk_id*8 + i] == extracted_watermark_bits[i]:
                hit_num += 1
        chunk_acc = hit_num / 8
        if chunk_acc > max_acc:
            max_acc = chunk_acc
            max_acc_id = chunk_id
    
    return max_acc, max_acc_id


# convert qweight to weight
def qweight2weight(qlinear: QuantLinear):
    if qlinear.wf.device != qlinear.qzeros.device:
        qlinear.wf = qlinear.wf.to(qlinear.qzeros.device)

    if qlinear.bits in [2, 4, 8]:
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(qlinear.qweight, 1).expand(-1, 32 // qlinear.bits, -1),
            qlinear.wf.unsqueeze(-1),
        ).to(torch.int16 if qlinear.bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2**qlinear.bits) - 1)
        # weight = weight.reshape(-1, qlinear.group_size, weight.shape[2])
        weight = weight.reshape(-1, weight.shape[2])
        weight = weight.to(torch.uint8)
        return weight
    
def weight2qweight(weight, qlinear: QuantLinear):
    intweight = weight.cpu().numpy().astype(np.uint32)
    i = 0
    row = 0
    qweight = np.zeros((intweight.shape[0] // 32 * qlinear.bits, intweight.shape[1]), dtype=np.uint32)
    while row < qweight.shape[0]:
        if qlinear.bits in [2, 4, 8]:
            for j in range(i, i + (32 // qlinear.bits)):
                qweight[row] |= intweight[j] << (qlinear.bits * (j - i))
            i += 32 // qlinear.bits
            row += 1
    qweight = qweight.astype(np.int32)
    qweight = torch.from_numpy(qweight)
    return qweight


def format_time(elapsed_time, time_name):
    # 转换为毫秒
    elapsed_time_ms = elapsed_time * 1000

    # 计算小时、分钟、秒和毫秒
    hours = int(elapsed_time_ms // 3600000)  # 毫秒转换为小时
    minutes = int((elapsed_time_ms % 3600000) // 60000)  # 余数转换为分钟
    seconds = int((elapsed_time_ms % 60000) // 1000)  # 余数转换为秒
    milliseconds = int(elapsed_time_ms % 1000)  # 余数即为毫秒
    # hours = elapsed_time // 3600  # 整除得到小时数
    # minutes = (elapsed_time % 3600) // 60  # 余数再整除得到分钟数
    # seconds = elapsed_time % 60  # 余数得到秒数
    print(f"{time_name} use time: {int(hours)}h {int(minutes)}m {int(seconds)}s {int(milliseconds)}ms")
