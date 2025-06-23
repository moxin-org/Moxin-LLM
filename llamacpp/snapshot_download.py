# !pip install huggingface_hub hf_transfer
import os
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

print("Starting download...")

snapshot_download(
    repo_id = "moxin-org/Moxin-7B-VLM",
    local_dir = "Moxin-7B-VLM",
)

snapshot_download(
    repo_id = "bobchenyx/Moxin-7B-VLM-hf",
    local_dir = "Moxin-7B-VLM-hf",
)

print("Download finished.")
