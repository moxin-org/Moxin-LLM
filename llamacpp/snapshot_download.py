# !pip install huggingface_hub hf_transfer
import os
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

print("Starting download...")

snapshot_download(
    repo_id = "second-state/moxin-instruct-7b-GGUF",
    local_dir = "Moxin-7B-Instruct",
    allow_patterns=["*Q6*"], # adjust if not enough VRAM
)

print("Download finished.")
