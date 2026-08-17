import os
import json

current_dir = os.path.dirname(os.path.abspath(__file__))

# Read config.py
with open(os.path.join(current_dir, "config.py"), "r", encoding="utf-8") as f:
    config_content = f.read()

# Read eval_utils.py
with open(os.path.join(current_dir, "eval_utils.py"), "r", encoding="utf-8") as f:
    eval_utils_content = f.read()

# Read SFT, Baseline, DPO, and compare scripts
def get_cleaned_code(script_name):
    py_path = os.path.join(current_dir, script_name)
    with open(py_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    lines = content.splitlines()
    cleaned_lines = []
    in_header = True
    for line in lines:
        if in_header:
            if line.startswith("# ==") or line.startswith("# 0") or line.startswith("# Goal:") or line.startswith("# Runtime:") or line.strip() == "":
                continue
            else:
                in_header = False
        if "### CELL 1:" in line or "### CELL 2:" in line or "### CELL 3:" in line or "### CELL 4:" in line:
            continue
        if line.strip().startswith("# !pip install"):
            continue
        cleaned_lines.append(line)
    return [line + "\n" for line in cleaned_lines]

baseline_code = get_cleaned_code("01_baseline_eval.py")
sft_eval_code = get_cleaned_code("03_sft_eval.py")
dpo_eval_code = get_cleaned_code("05_dpo_eval.py")
compare_code = get_cleaned_code("06_compare_and_report.py")

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ⚔️ SmolLM2-1.7B Post-Training Alignment Evaluation & Reporting Pipeline\n\n",
                "This combined notebook performs baseline, SFT, and DPO evaluation runs, computes quantitative statistics, and runs LLM-as-a-Judge using the Gemini API. It downloads already-trained LoRA adapter weights directly from Hugging Face Hub, allowing you to run evaluations without training."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 1: Install Required Dependencies"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install required dependencies and uninstall pre-installed incompatible torchao\n",
                "!pip uninstall -y torchao\n",
                "!pip install -q -U transformers trl datasets accelerate peft evaluate bitsandbytes huggingface_hub matplotlib pandas scikit-learn\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 2: Write configuration and helper utility files to disk"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%writefile config.py\n"
            ] + [line + "\n" for line in config_content.splitlines()]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%writefile eval_utils.py\n"
            ] + [line + "\n" for line in eval_utils_content.splitlines()]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 3: Download SFT & DPO Adapters from Hugging Face Hub"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from huggingface_hub import snapshot_download\n",
                "import os\n\n",
                "sft_dir = \"/kaggle/working/checkpoints/sft\"\n",
                "dpo_dir = \"/kaggle/working/checkpoints/dpo\"\n\n",
                "if not os.path.exists(sft_dir) or not os.listdir(sft_dir):\n",
                "    print(\"Downloading SFT LoRA adapters from Hugging Face Hub...\")\n",
                "    snapshot_download(\n",
                "        repo_id=\"manojpaul9986/smollm2-1.7b-sft-lora\",\n",
                "        local_dir=sft_dir\n",
                "    )\n",
                "    print(\"SFT adapter downloaded.\")\n",
                "else:\n",
                "    print(f\"Found existing SFT adapter at: {sft_dir}\")\n\n",
                "if not os.path.exists(dpo_dir) or not os.listdir(dpo_dir):\n",
                "    print(\"Downloading DPO LoRA adapters from Hugging Face Hub...\")\n",
                "    snapshot_download(\n",
                "        repo_id=\"manojpaul9986/smollm2-1.7b-dpo-lora\",\n",
                "        local_dir=dpo_dir\n",
                "    )\n",
                "    print(\"DPO adapter downloaded.\")\n",
                "else:\n",
                "    print(f\"Found existing/newly trained DPO adapter at: {dpo_dir} (skipping Hub download)\")\n\n",
                "print(\"All required adapters are ready on disk!\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 4: Run Baseline Evaluation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": baseline_code
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 5: Run SFT Evaluation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": sft_eval_code
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 6: Run DPO Evaluation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": dpo_eval_code
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 7: Compute Comparisons and Report"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": compare_code
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

notebook_path = os.path.join(current_dir, "evaluation_and_report_pipeline.ipynb")
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print(f"Generated combined notebook at: {notebook_path}")
