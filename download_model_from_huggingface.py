import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
from huggingface_hub import snapshot_download


def main(repo_id):
    local_dir = '../models/' + repo_id
    if "llama" in repo_id:
        snapshot_download(repo_id=repo_id, 
                        allow_patterns=["*.model", "*.json", "*.bin", "*.py", "*.md", "*.txt", "*.safetensors"],
                        ignore_patterns=["*.bin", "*.msgpack","*.h5", "*.ot"],
                        local_dir=local_dir, 
                        local_dir_use_symlinks=False)
    else:
        snapshot_download(repo_id=repo_id, 
                        allow_patterns=["*.model", "*.json", "*.bin", "*.py", "*.md", "*.txt"],
                        ignore_patterns=["*.safetensors", "*.msgpack","*.h5", "*.ot"],
                        local_dir=local_dir, 
                        local_dir_use_symlinks=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download model from Hugging Face Hub')
    parser.add_argument('--repo_id', type=str, default='meta-llama/Llama-2-7b-hf', help='Repository ID of the model to download')
    args = parser.parse_args()
    main(args.repo_id)
