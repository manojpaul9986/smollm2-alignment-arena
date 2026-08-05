"""
config.py
Shared configuration + resume-state helpers for the SmolLM-1.7B LoRA pipeline.
Paste this as the FIRST cell in every notebook (01 through 06).
"""

import os
# Force single GPU to prevent PyTorch DataParallel device mismatch errors in Jupyter notebooks
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import random
import numpy as np
import torch

# ---------------------------------------------------------------------
# Paths (Kaggle-specific). /kaggle/working is ephemeral per-session but
# persists for the life of a session and is committed on "Save Version".
# ---------------------------------------------------------------------
BASE_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "./work"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
STATE_PATH = os.path.join(BASE_DIR, "state.json")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B"          # upgraded to 1.7B parameter model
SFT_OUT_DIR = os.path.join(CKPT_DIR, "sft")
DPO_OUT_DIR = os.path.join(CKPT_DIR, "dpo")

# If you want checkpoints to survive session death, set these and log in
# with `huggingface_hub.login()` in the notebook before training.
PUSH_TO_HUB = False
HUB_SFT_REPO = "your-username/SmolLM-1.7B-SFT-LoRA"
HUB_DPO_REPO = "your-username/SmolLM-1.7B-DPO-LoRA"

SEED = 42

# ---------------------------------------------------------------------
# LoRA Parameters (PEFT)
# ---------------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Standard target modules for Llama/SmolLM architecture
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ---------------------------------------------------------------------
# LLM Judge Configuration (Gemini API)
# ---------------------------------------------------------------------
# Ensure you upload this as a Kaggle Secret or set it in your environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    """True if running on the primary process under DDP, or if running single-process."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank == 0 and local_rank == 0


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"phase": "none", "step": 0, "notes": ""}


def save_state(phase: str, step: int = 0, notes: str = ""):
    if not is_main_process():
        return
    state = {"phase": phase, "step": step, "notes": notes}
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    print(f"[state] phase={phase} step={step} notes={notes}")


def find_latest_checkpoint(output_dir: str):
    """Return path to latest `checkpoint-N` dir, or None."""
    if not os.path.isdir(output_dir):
        return None
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.join(output_dir, ckpts[-1])


def save_metrics(name: str, metrics: dict):
    if not is_main_process():
        return
    path = os.path.join(METRICS_DIR, f"{name}_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[metrics] saved -> {path}")
    print(json.dumps(metrics, indent=2))


def load_metrics(name: str) -> dict:
    path = os.path.join(METRICS_DIR, f"{name}_metrics.json")
    with open(path) as f:
        return json.load(f)


def push_checkpoint_to_dataset(local_dir: str, kaggle_dataset_slug: str):
    print(f"kaggle datasets version -p {local_dir} -m 'checkpoint update' -d {kaggle_dataset_slug}")
