# ELLMark

This repository contains the code for the Paper: ***An Efficient White-box LLM Watermarking for IP Protection on Online Market Platforms***.

<!-- Due to time constraints, this code submission was made hastily, but the core code is included. **We assure that we will update the code and documentation with more user-friendly versions as soon as possible.** -->


### 1 Model Download Script

Run the `download_model.sh` script to begin downloading the model:

```bash
bash bash/download_model.sh
```

##### Customization

You can modify the script to download additional models, or adapt it for other purposes (e.g., model quantization, inference) by editing the appropriate variables(`model_name`) and paths in the script.

### 2 Experiment

##### Steps to modify `bash.sh` for watermark embedding and extraction for `facebook/opt-1.3b`:

1. **Watermarking Setup:**
   - You need to modify the script to handle watermarking for the `facebook/opt-1.3b` model. Ensure that `model_name` is set to `facebook/opt-1.3b` and configure watermark embedding/extraction methods.
2. **Embedding Watermark:**
   - Use the `RandomMark`, `EmMark`, and `ELLMark` watermark embedding techniques (with pre-existing Python scripts such as `randommark_insert_watermark.py`, `emmark_insert_watermark.py`, and `insert_watermark.py`). Each method will be defined in a separate function within the script.
3. **Watermark Extraction:**
   - The watermark extraction will be done using `randommark_extract_watermark.py`, `emmark_extract_watermark.py`, and `extract_watermark.py`.
4. **Quantization:**
   - Quantization is handled via a separate function (`quant_model`), which will reduce the model size to 8-bit precision.
5. **Model Evaluation:**
   - After the watermark embedding and extraction, the model will be evaluated to ensure the watermark did not significantly degrade performance.

##### Modified `bash.sh` content:

```bash
# Main Model Path
model_path="${model_dir}${model_name}"
log_path="${log_dir}${model_name}"
mkdir -p "$(dirname "$log_path")"

# Run OurMark Insert and Extract
save_model="${model_path}-inserted-by-ourmark"
run_ourmark_insert
run_ourmark_extract

# Optionally Quantize the Model
# quant_model

# Optionally Evaluate the Model
# run_model_evaluate $save_model

# Optionally Delete the Model
# delete_model $save_model
```

##### Running the Script:

Execute the script to embed watermarks, extract watermarks, and optionally quantize the model.

```bash
bash bash/bash.sh
```
