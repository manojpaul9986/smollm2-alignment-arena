# ⚔️ SmolLM2-1.7B Alignment & Evaluation Arena (SFT & DPO)

This repository contains the complete post-training alignment pipeline to adapt and scale **SmolLM2-1.7B** from raw base completion models into reasoning-aligned instruction assistants using **Supervised Fine-Tuning (SFT)** and **Direct Preference Optimization (DPO)**.

The project features a **Gradio Side-by-Side Chatbot Arena** to qualitatively play with and benchmark the outputs of the models at each training phase.

---

## 📈 Alignment Results (1.7B Model)

By aligning the model on instruction following and mathematical reasoning datasets, we observed a **246% improvement** on the GSM8K benchmark while maintaining linguistic perplexity.

| Metric | Base Model | SFT Model | DPO Model |
| :--- | :---: | :---: | :---: |
| **GSM8K Accuracy %** | 7.00% | 15.00% | **37.00%** |
| **Instruction-Following %** | 60.00% | **80.00%** | 70.00% |
| **Perplexity (Wikitext-2)** | **42.53** | 47.05 | 47.04 |

---

## 🤗 Hugging Face Models

The trained LoRA adapters are hosted publicly on Hugging Face:
* **SFT LoRA Adapter**: [manojpaul9986/smollm2-1.7b-sft-lora](https://huggingface.co/manojpaul9986/smollm2-1.7b-sft-lora)
* **DPO LoRA Adapter**: [manojpaul9986/smollm2-1.7b-dpo-lora](https://huggingface.co/manojpaul9986/smollm2-1.7b-dpo-lora)

---

## 🛠️ Repository Structure

* `01_baseline_eval.ipynb` - Initial evaluation of the raw base model.
* `02_sft_train.ipynb` - LoRA-based Supervised Fine-Tuning.
* `03_sft_eval.ipynb` - Evaluation of the SFT checkpoints.
* `04_dpo_train.ipynb` - LoRA-based Direct Preference Optimization.
* `05_dpo_eval.ipynb` - Final evaluation of the DPO checkpoints (includes ChatML stopping fixes).
* `06_compare_and_report.ipynb` - Win-rate computations using Gemini LLM-as-a-Judge.
* `app.py` - Gradio chatbot arena for real-time model comparison.
* `eval_utils.py` - Core evaluation utilities (perplexity, regex extraction, token generation logic).
* `config.py` - Directory path and hyperparameter configurations.

---

## 💡 Key Technical Challenges Solved

### 1. Token-Generation Parser Collapses (Hallucination Loops)
Smaller models like `SmolLM2` do not naturally respect ChatML turn boundaries (like `<|im_end|>`) during raw token generation. In early evaluations, the model would solve a math problem correctly, but then keep generating fake subsequent user/assistant questions, which corrupted the regex numeric extraction.
* **Solution**: Implemented a custom stopping criteria in `eval_utils.py` mapping standard `<|im_end|>` token conversions directly to the generation engine (`eos_token_id`). This immediately terminated generations on answer completion, preventing loop pollution and unlocking the true **37% GSM8K score**.

### 2. Multi-Stage Adapter Weight Chains
DPO adapters must be loaded on top of SFT-merged base weights rather than the raw base model to prevent format collapse.
* **Solution**: Developed a clean loading utility that compiles SFT LoRA adapters, merges the weights into the base network, and then hot-loads the DPO LoRA adapters on top.

---

## 🚀 How to Run the Gradio Arena

### 1. Install Dependencies
```bash
pip install torch transformers peft trl gradio datasets
```

### 2. Launch the Arena
Ensure SFT and DPO checkpoints are restored in `checkpoints/sft` and `checkpoints/dpo` (relative to this directory), then execute:
```bash
python app.py
```
Open the public Gradio link printed in your terminal to chat with all three models side-by-side!

---

## 📜 License
This repository is licensed under the MIT License.
