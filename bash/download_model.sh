# . /etc/network_turbo #学术加速
# huggingface-cli login

# export HF_ENDPOINT="https://hf-mirror.com" #HF mirror website
export HF_API_TOKEN="hf_YourHuggingfaceApiToken"
#全局路径参数
model_dir="../models/"
log_dir="../logs/"

#全局模型参数

model_name="microsoft/Phi-3-medium-4k-instruct"
# model_name="meta-llama/Meta-Llama-3-8B"
# quant_name="facebook/opt-1.3b-8bit"
# 1.下载模型
#需要注意hf上模型保存格式(.bin / .safetensors)
python -u download_model_from_huggingface.py \
    --repo_id $model_name
echo "$model_name [DOWNLOAD] done!"