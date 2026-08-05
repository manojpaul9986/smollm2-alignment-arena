import os
import json

# Get current folder (v2_scale_up)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Read config.py
with open(os.path.join(current_dir, "config.py"), "r", encoding="utf-8") as f:
    config_content = f.read()

# Read eval_utils.py
with open(os.path.join(current_dir, "eval_utils.py"), "r", encoding="utf-8") as f:
    eval_utils_content = f.read()

scripts = [
    "01_baseline_eval.py",
    "02_sft_train.py",
    "03_sft_eval.py",
    "04_dpo_train.py",
    "05_dpo_eval.py",
    "06_compare_and_report.py"
]

pip_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Install required dependencies\n",
        "!pip install -q -U transformers trl datasets accelerate peft evaluate bitsandbytes huggingface_hub matplotlib pandas scikit-learn gradio torchao\n"
    ]
}

config_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "%%writefile config.py\n"
    ] + [line + "\n" for line in config_content.splitlines()]
}

eval_utils_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "%%writefile eval_utils.py\n"
    ] + [line + "\n" for line in eval_utils_content.splitlines()]
}

for script_name in scripts:
    py_path = os.path.join(current_dir, script_name)
    with open(py_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    lines = content.splitlines()
    header_lines = []
    code_lines = []
    
    in_header = True
    for line in lines:
        if in_header:
            if line.startswith("# ==") or line.startswith("# 0") or line.startswith("# Goal:") or line.startswith("# Runtime:") or line.strip() == "":
                header_lines.append(line)
            else:
                in_header = False
                code_lines.append(line)
        else:
            code_lines.append(line)
            
    header_text = "\n".join(header_lines).strip("# \n=")
    
    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {script_name.replace('.py', '')} (SmolLM-1.7B LoRA Pipeline)\n\n",
            "This notebook is generated automatically to run on Kaggle.\n\n",
            "### Goal & Metadata:\n",
            "```\n",
            header_text + "\n",
            "```\n"
        ]
    }
    
    cleaned_code_lines = []
    for line in code_lines:
        if "### CELL 1:" in line or "### CELL 2:" in line or "Paste contents of config.py" in line or "sys.path.append" in line:
            continue
        if line.strip().startswith("# !pip install") or line.strip().startswith("# import sys; sys.path.append"):
            continue
        cleaned_code_lines.append(line)
        
    code_source = [line + "\n" for line in cleaned_code_lines]
    
    main_code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code_source
    }
    
    notebook = {
        "cells": [
            markdown_cell,
            pip_cell,
            config_cell,
            eval_utils_cell,
            main_code_cell
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
    
    notebook_path = os.path.join(current_dir, script_name.replace(".py", ".ipynb"))
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Generated {notebook_path}")
