#!/bin/bash

# source /etc/network_turbo #学术加速
export HF_ENDPOINT="https://hf-mirror.com" #HF镜像网站

#全局路径参数
model_dir="../models/"
log_dir="../logs/"

#全局模型参数
cuda_device=0
export CUDA_VISIBLE_DEVICES=$cuda_device
export TOKENIZERS_PARALLELISM=false
model_name="microsoft/Phi-3-medium-4k-instruct"  #e.g. model_name="facebook/opt-1.3b"
quant_name="microsoft/Phi-3-medium-4k-instruct-8bit"
hidden_size=5120        #水印嵌入需要
model_seqlen=2048       #模型量化需要
# model_path="${model_dir}${model_name}" 或者 "${model_dir}${quant_name}"
# log_path="${log_dir}${model_name}" 或者 "${log_dir}${quant_name}"

#全局水印参数
watermark="mark"    #（重水印攻击时更换mark）【用于所有方法】
seed=100            #（随机修改攻击和重水印攻击时更换seed）【用于RandomMark】
password="asdfqwer" #（重水印攻击时更换password）【用于OurMark】

#全局函数定义
# modify_rate=[0.375, 0.75]  #-> 全精度选0.375 8bit选0.75
# save_model="${model_path}-inserted-by-randommark"

run_random_insert () {
    log_file="${log_path}-inserted-by-randommark.log"
    python -u ../pielines/randommark_insert_watermark.py \
    --model $model_path \
    --hidden_size $hidden_size \
    --watermark $watermark \
    --seed $seed \
    --modify_rate $modify_rate \
    --save_model $save_model > "$log_file" 2>&1
    echo "=============== random insert done! ${model_path} --> ${save_model} "
}

run_random_attack () {
    log_file="${log_path}-random-attack.log"
    python -u ../pielines/randommark_insert_watermark.py \
    --model $model_path \
    --hidden_size $hidden_size \
    --watermark $watermark \
    --seed $seed \
    --modify_rate $modify_rate \
    --save_model $save_model > "$log_file" 2>&1
    echo "=============== random attack done! ${model_path} --> ${save_model} "
}

# candidate_rate=[50, 60] #-> 模型大小< 6.7b选50 ＞6.7b选60
# modify_rate=0.75
run_emmark_insert () {
    log_file="${log_path}-inserted-by-emmark.log"
    python -u ../pielines/emmark_insert_watermark.py \
    --model $model_path \
    --watermark $watermark \
    --candidate_rate $candidate_rate \
    --hidden_size $hidden_size \
    --modify_rate $modify_rate \
    --save_model $save_model > "$log_file" 2>&1
    echo "=============== emmark insert done! ${model_path} --> ${save_model} "
}

run_emmark_extract () {
    log_file="${log_path}-extracted-by-emmark.log"
    python -u ../pielines/emmark_extract_watermark.py \
    --model $model_path \
    --inserted_model $save_model \
    --watermark $watermark \
    --candidate_rate $candidate_rate \
    --modify_rate $modify_rate \
    --hidden_size $hidden_size > "$log_file" 2>&1
    echo "=============== emmark extract done! --> $save_model "
}

# xi=[2, 4] #-> 全精度选4 8bit选2
# position_num=[6, 12] #-> 全精度选12 8bit选6
# delta=20
# save_model="${model_path}-inserted-by-ourmark"
# mode="simple"  # （攻击时改为robust）
run_ourmark_insert () {
    log_file="${log_path}-inserted-by-ourmark.log"
    python -u ../pielines/insert_watermark.py \
    --password $password \
    --model $model_path \
    --hidden_size $hidden_size \
    --xi $xi \
    --position_num $position_num \
    --delta $delta \
    --watermark $watermark \
    --save_model $save_model > "$log_file" 2>&1 
    echo "=============== ourmark insert done! ${model_path} --> ${save_model} "
}

run_ourmark_extract () {
    log_file="${log_path}-extracted-by-ourmark.log"
    python -u ../pielines/extract_watermark.py \
    --password $password \
    --model $save_model \
    --hidden_size $hidden_size \
    --xi $xi \
    --position_num $position_num \
    --watermark $watermark \
    --mode $mode > "$log_file" 2>&1 
    echo "=============== ourmark extract done! --> $save_model "
} 

#peft_mode = [lora, ptuning, prompt, bitfit]
run_finetune () {
    log_file="${log_path}-lora.log"
    python -u ../finetune_lm.py \
    --model_name_or_path $model_path \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --peft_mode $1 \
    --num_train_epochs 1 \
    --block_size 256 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --max_train_samples 30000 \
    --max_eval_samples 128 \
    --output_dir $save_path \
    --overwrite_output_dir yes \
    --optim adafactor \
    --learning_rate=8e-4 \
    --save_strategy='epoch' \
    --logging_strategy='steps' \
    --logging_first_step \
    --logging_steps=10 \
    --save_total_limit=1 > "$log_file" 2>&1
}

quant_model () {
    log_file="${log_dir}${model_name}-quanted.log"
    python -u ../quant/gptq_model.py \
    --model_path "${model_dir}${model_name}" \
    --quant_path "${model_dir}${quant_name}" \
    --model_seqlen $model_seqlen > "$log_file" 2>&1
    echo "=============== quant model done! ${model_name} --> ${quant_name} "
}

run_model_evaluate () {
    python -u ../evaluate_lm.py \
    --model $1 \
    --ctx_length $model_seqlen \
    --eval_zero_shot True
    echo "=============== evaluate model done! --> $1 "
}

delete_model () {
    rm -rf $1
    echo "=============== deleta model done! --> $1 "
}

model_path="${model_dir}${model_name}"
log_path="${log_dir}${model_name}"
mkdir -p "$(dirname "$log_path")"

xi=4            #[2, 4] -> 全精度选4 8bit选2
position_num=12   #[6, 12] -> 全精度选12 8bit选6
delta=20
mode="simple"  # （攻击时改为robust）
save_model="${model_path}-inserted-by-ourmark"

run_ourmark_insert
run_ourmark_extract
# run_model_evaluate $save_model
# delete_model $save_model