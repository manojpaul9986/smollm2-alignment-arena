# =====================================================================
# 05_dpo_eval.ipynb
# Goal: run the identical eval harness on the final DPO checkpoint.
# Runtime: ~5-10 min
# =====================================================================

### CELL 1: config + utils ###
# !pip install -q -U transformers trl datasets accelerate peft evaluate bitsandbytes

if __name__ == "__main__":
    from config import set_seed, save_state, save_metrics, load_metrics, DPO_OUT_DIR
    from eval_utils import run_full_eval
    
    set_seed()
    save_state(phase="dpo_eval_start")
    
    ### CELL 2: run eval ###
    results = run_full_eval(
        model_path=DPO_OUT_DIR,
        chat=True,
        ppl_n=200,
        gsm8k_n=100,
    )
    
    ### CELL 3: save + compare across all three phases ###
    save_metrics("dpo", results)
    save_state(phase="dpo_eval_done")
    
    try:
        baseline = load_metrics("baseline")
        sft = load_metrics("sft")
        
        print("\n--- Base vs SFT vs DPO ---")
        rows = [
            ("Perplexity", "perplexity"),
            ("Instr-follow acc %", "instruction_following_acc_pct"),
            ("GSM8K acc %", "gsm8k_acc_pct"),
            ("CoT presence %", "cot_presence_pct"),
        ]
        print(f"{'Metric':<24}{'Base':>10}{'SFT':>10}{'DPO':>10}")
        for label, key in rows:
            print(f"{label:<24}{baseline[key]:>10.2f}{sft[key]:>10.2f}{results[key]:>10.2f}")
    except FileNotFoundError:
        print("\n--- DPO Metrics (Baseline/SFT comparison metrics not uploaded) ---")
        print(f"Perplexity: {results['perplexity']:.2f}")
        print(f"Instr-follow acc %: {results['instruction_following_acc_pct']:.2f}")
        print(f"GSM8K acc %: {results['gsm8k_acc_pct']:.2f}")
        print(f"CoT presence %: {results['cot_presence_pct']:.2f}")
        print("\nNote: Upload baseline_metrics.json and sft_metrics.json to /kaggle/working/metrics/ to see the comparative table.")
