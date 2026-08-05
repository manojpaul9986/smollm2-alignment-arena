# =====================================================================
# 01_baseline_eval.ipynb  (paste each ### CELL ### block as its own cell)
# Goal: establish "before" metrics on raw SmolLM2-1.7B before any tuning.
# Runtime: ~5-10 min on 1xT4
# =====================================================================

### CELL 1: install deps ###
# !pip install -q -U transformers trl datasets accelerate peft evaluate bitsandbytes

if __name__ == "__main__":
    ### CELL 2: config ###
    from config import set_seed, save_state, save_metrics, MODEL_ID
    from eval_utils import run_full_eval
    
    set_seed()
    save_state(phase="baseline_eval_start")
    
    ### CELL 3: run eval ###
    # Base model has no chat template -> chat=False, we prompt it raw.
    results = run_full_eval(
        model_path=MODEL_ID,
        chat=False,
        ppl_n=200,
        gsm8k_n=100,
    )
    
    ### CELL 4: save + inspect ###
    save_metrics("baseline", results)
    save_state(phase="baseline_eval_done")
    
    print("\n--- Baseline snapshot ---")
    print(f"Perplexity:              {results['perplexity']:.2f}")
    print(f"Instruction-follow acc:  {results['instruction_following_acc_pct']:.1f}%")
    print(f"GSM8K acc:               {results['gsm8k_acc_pct']:.1f}%")
    print(f"CoT presence:            {results['cot_presence_pct']:.1f}%")
