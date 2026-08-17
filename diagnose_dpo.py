from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
import os

print("Downloading SFT adapter model file...")
sft_file = hf_hub_download(
    repo_id="manojpaul9986/smollm2-1.7b-sft-lora",
    filename="adapter_model.safetensors",
    local_dir="checkpoints/sft"
)

print("Downloading DPO adapter model file...")
dpo_file = hf_hub_download(
    repo_id="manojpaul9986/smollm2-1.7b-dpo-lora",
    filename="adapter_model.safetensors",
    local_dir="checkpoints/dpo"
)

print("Loading adapter weights...")
sft_w = load_file(sft_file)
dpo_w = load_file(dpo_file)

print("\n--- Safetensors weight differences ---")
total_norm_diff = 0.0
for k in sft_w:
    if k in dpo_w:
        # Load weights on CPU for comparison
        sft_tensor = sft_w[k].float()
        dpo_tensor = dpo_w[k].float()
        diff = (sft_tensor - dpo_tensor).norm().item()
        sft_norm = sft_tensor.norm().item()
        total_norm_diff += diff
        print(f"{k:60s} | SFT Norm: {sft_norm:10.4f} | Diff Norm: {diff:10.6f}")

print(f"\nTOTAL adapter weight diff norm: {total_norm_diff:.6f}")
