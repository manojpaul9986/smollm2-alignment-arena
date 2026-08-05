# =====================================================================
# 03_sft_eval.ipynb
# Goal: run the identical eval harness on the SFT checkpoint.
# Runtime: ~5 min
# =====================================================================

### CELL 1: config + utils ###
# !pip install -q -U transformers trl datasets accelerate peft evaluate bitsandbytes

if __name__ == "__main__":
    from config import set_seed, save_state, save_metrics, SFT_OUT_DIR
    from eval_utils import run_full_eval
    
    set_seed()
    save_state(phase="sft_eval_start")
    
    ### CELL 2: run eval (chat=True now — model has ChatML structure) ###
    results = run_full_eval(
        model_path=SFT_OUT_DIR,
        chat=True,
        ppl_n=200,
        gsm8k_n=100,
    )
    
    ### CELL 3: save + compare vs baseline ###
    from config import load_metrics
    save_metrics("sft", results)
    save_state(phase="sft_eval_done")
    
    baseline = load_metrics("baseline")
    
    print("\n--- Base vs SFT ---")
    print(f"{'Metric':<28}{'Base':>10}{'SFT':>10}")
    print(f"{'Perplexity':<28}{baseline['perplexity']:>10.2f}{results['perplexity']:>10.2f}")
    print(f"{'Instr-follow acc %':<28}{baseline['instruction_following_acc_pct']:>10.1f}{results['instruction_following_acc_pct']:>10.1f}")
    print(f"{'GSM8K acc %':<28}{baseline['gsm8k_acc_pct']:>10.1f}{results['gsm8k_acc_pct']:>10.1f}")
    print(f"{'CoT presence %':<28}{baseline['cot_presence_pct']:>10.1f}{results['cot_presence_pct']:>10.1f}")
