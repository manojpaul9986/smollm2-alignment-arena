# =====================================================================
# 06_compare_and_report.ipynb
# Goal: consolidate all metrics, run qualitative evaluation,
# and query the Gemini API Judge to compute model Win Rates.
# Runtime: ~3-5 min
# =====================================================================

### CELL 1: load all metrics ###
# !pip install -q -U transformers trl datasets accelerate peft evaluate bitsandbytes matplotlib pandas

if __name__ == "__main__":
    import os
    from config import load_metrics, save_state, BASE_DIR
    import pandas as pd
    import matplotlib.pyplot as plt
    
    save_state(phase="report_start")
    
    baseline = load_metrics("baseline")
    sft = load_metrics("sft")
    dpo = load_metrics("dpo")
    
    def pct_identical(samples_a, samples_b):
        if not samples_a or not samples_b or len(samples_a) != len(samples_b):
            return 0.0
        matches = sum(a["response"] == b["response"] for a, b in zip(samples_a, samples_b))
        return 100.0 * matches / len(samples_a)
        
    gsm8k_base_sft_overlap = pct_identical(baseline.get("gsm8k_samples", []), sft.get("gsm8k_samples", []))
    gsm8k_sft_dpo_overlap = pct_identical(sft.get("gsm8k_samples", []), dpo.get("gsm8k_samples", []))
    instr_base_sft_overlap = pct_identical(baseline.get("instr_samples", []), sft.get("instr_samples", []))
    instr_sft_dpo_overlap = pct_identical(sft.get("instr_samples", []), dpo.get("instr_samples", []))
    
    df = pd.DataFrame({
        "Metric": ["Perplexity", "Instruction-Follow Acc (%)", "GSM8K Acc (%)", "CoT Presence (%)"],
        "Base": [baseline["perplexity"], baseline["instruction_following_acc_pct"],
                 baseline["gsm8k_acc_pct"], baseline["cot_presence_pct"]],
        "SFT": [sft["perplexity"], sft["instruction_following_acc_pct"],
                sft["gsm8k_acc_pct"], sft["cot_presence_pct"]],
        "DPO": [dpo["perplexity"], dpo["instruction_following_acc_pct"],
                dpo["gsm8k_acc_pct"], dpo["cot_presence_pct"]],
    })
    print("\n--- Quantitative Summary ---")
    print(df.to_string(index=False))
    print(f"\n--- Output Overlap (Identity Checks) ---")
    print(f"Base vs SFT GSM8K Overlap:       {gsm8k_base_sft_overlap:.1f}%")
    print(f"SFT vs DPO GSM8K Overlap:        {gsm8k_sft_dpo_overlap:.1f}%")
    print(f"Base vs SFT Instruction Overlap: {instr_base_sft_overlap:.1f}%")
    print(f"SFT vs DPO Instruction Overlap:  {instr_sft_dpo_overlap:.1f}%")
    df.to_csv(os.path.join(BASE_DIR, "final_comparison.csv"), index=False)
    
    ### CELL 2: bar chart of the progression ###
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    metrics = ["Instruction-Follow Acc (%)", "GSM8K Acc (%)", "CoT Presence (%)", "Perplexity"]
    for ax, metric in zip(axes, metrics):
        row = df[df["Metric"] == metric].iloc[0]
        ax.bar(["Base", "SFT", "DPO"], [row["Base"], row["SFT"], row["DPO"]],
               color=["#94a3b8", "#60a5fa", "#34d399"])
        ax.set_title(metric, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "progression_chart.png"), dpi=150)
    plt.close()
    
    ### CELL 3: generate outputs for qualitative showcase ###
    from eval_utils import load_model_and_tokenizer, generate, get_gemini_judge_score
    from config import MODEL_ID, SFT_OUT_DIR, DPO_OUT_DIR, GEMINI_API_KEY, GROQ_API_KEY
    import torch
    
    showcase_prompts = [
        "If a train travels 60 miles in 1.5 hours, what is its average speed?",
        "Explain in simple terms why the sky is blue.",
        "A store has 120 apples. They sell 45 in the morning and 30 in the afternoon. How many are left?",
        "What are three tips for staying focused while studying?",
        "If I have $50 and spend 40% of it, how much do I have left?",
        "Write a polite email to a customer explaining that their shipment will be delayed by 2 days due to weather conditions.",
        "What is the difference between a software library and a framework? Explain with a simple analogy."
    ]
    
    responses = {p: {} for p in showcase_prompts}
    
    for label, path, chat in [
        ("Base", MODEL_ID, False),
        ("SFT", SFT_OUT_DIR, True),
        ("DPO", DPO_OUT_DIR, True),
    ]:
        print(f"Generating answers from {label} model...")
        model, tok = load_model_and_tokenizer(path, is_chat=chat)
        for p in showcase_prompts:
            resp = generate(model, tok, p, chat=chat, max_new_tokens=250)
            responses[p][label] = resp
        del model
        torch.cuda.empty_cache()
    
    ### CELL 4: LLM-as-a-Judge Evaluation (Win Rate) ###
    judge_results = []
    win_counts = {"Base": 0, "SFT": 0, "DPO": 0, "Tie": 0, "Error": 0}
    
    if GROQ_API_KEY or GEMINI_API_KEY:
        if GROQ_API_KEY:
            print("\nQuerying Groq API Judge (LLM-as-a-Judge)...")
        else:
            print("\nQuerying Gemini API Judge (LLM-as-a-Judge)...")
        for idx, prompt in enumerate(showcase_prompts):
            print(f"  Evaluating Prompt {idx+1}/{len(showcase_prompts)}...")
            base_ans = responses[prompt]["Base"]
            sft_ans = responses[prompt]["SFT"]
            dpo_ans = responses[prompt]["DPO"]
            
            judge_opinion = get_gemini_judge_score(prompt, base_ans, sft_ans, dpo_ans)
            winner = judge_opinion.get("winner", "Tie")
            reason = judge_opinion.get("reason", "No reason provided.")
            
            win_counts[winner] = win_counts.get(winner, 0) + 1
            judge_results.append({
                "prompt": prompt,
                "Base": base_ans,
                "SFT": sft_ans,
                "DPO": dpo_ans,
                "winner": winner,
                "reason": reason
            })
            
        judge_df = pd.DataFrame(judge_results)
        judge_df.to_csv(os.path.join(BASE_DIR, "llm_judge_results.csv"), index=False)
        
        print("\n--- LLM-as-a-Judge Win Rates ---")
        total_valid = sum(win_counts[k] for k in ["Base", "SFT", "DPO", "Tie"])
        if total_valid > 0:
            for model_name in ["Base", "SFT", "DPO", "Tie"]:
                pct = 100 * win_counts[model_name] / total_valid
                print(f"Model {model_name:<5} | Wins: {win_counts[model_name]} | Win Rate: {pct:.1f}%")
        
        # Display side-by-side results
        for item in judge_results:
            print(f"\n{'='*80}\nPROMPT: {item['prompt']}\n{'='*80}")
            print(f"[Base]: {item['Base']}\n")
            print(f"[SFT]: {item['SFT']}\n")
            print(f"[DPO]: {item['DPO']}\n")
            print(f"👉 JUDGE WINNER: {item['winner']}")
            print(f"👉 REASON: {item['reason']}")
    else:
        print("\n[Notice] No active API keys (Groq/Gemini) found. Skipping LLM-as-a-Judge evaluation.")
        # Just save regular outputs without judge evaluation
        showcase_rows = []
        for prompt, resps in responses.items():
            showcase_rows.append({
                "prompt": prompt,
                "Base": resps["Base"],
                "SFT": resps["SFT"],
                "DPO": resps["DPO"]
            })
        pd.DataFrame(showcase_rows).to_csv(os.path.join(BASE_DIR, "qualitative_showcase.csv"), index=False)
        for prompt in showcase_prompts:
            print(f"\n{'='*80}\nPROMPT: {prompt}\n{'='*80}")
            for label in ["Base", "SFT", "DPO"]:
                print(f"\n[{label}]\n{responses[prompt][label]}")
                
    # ---------------------------------------------------------------------
    # OPTION 2: Compute Offline Validation DPO Margin and Accuracy
    # ---------------------------------------------------------------------
    print("\n--- Running Offline Validation DPO Margin & Accuracy ---")
    try:
        from datasets import load_dataset
        from eval_utils import compute_dataset_logps
        import numpy as np
        import json
        
        print("Loading HuggingFaceH4/ultrafeedback_binarized (test_prefs)...")
        raw_eval = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
        
        MAX_CHARS = 1200
        def is_high_quality_and_short(ex):
            chosen = ex["chosen"]
            rejected = ex["rejected"]
            chosen_text = chosen[-1]["content"] if isinstance(chosen, list) else str(chosen)
            rejected_text = rejected[-1]["content"] if isinstance(rejected, list) else str(rejected)
            if len(chosen_text) > MAX_CHARS or len(rejected_text) > MAX_CHARS:
                return False
            score_c = ex.get("score_chosen", 0.0)
            score_r = ex.get("score_rejected", 0.0)
            if score_c is not None and score_r is not None:
                if (score_c - score_r) < 1.0:
                    return False
            return True
            
        filtered_eval = raw_eval.filter(is_high_quality_and_short, num_proc=4).shuffle(seed=42)
        dpo_eval_samples = filtered_eval.select(range(min(100, len(filtered_eval))))
        
        print("Computing log probabilities under SFT Model...")
        sft_chosen_lps, sft_rejected_lps = compute_dataset_logps(SFT_OUT_DIR, dpo_eval_samples)
        
        print("Computing log probabilities under DPO Model...")
        dpo_chosen_lps, dpo_rejected_lps = compute_dataset_logps(DPO_OUT_DIR, dpo_eval_samples)
        
        # Calculate rewards and margins
        beta = 0.05
        sft_chosen_lps = np.array(sft_chosen_lps)
        sft_rejected_lps = np.array(sft_rejected_lps)
        dpo_chosen_lps = np.array(dpo_chosen_lps)
        dpo_rejected_lps = np.array(dpo_rejected_lps)
        
        dpo_chosen_rewards = beta * (dpo_chosen_lps - sft_chosen_lps)
        dpo_rejected_rewards = beta * (dpo_rejected_lps - sft_rejected_lps)
        
        margins = dpo_chosen_rewards - dpo_rejected_rewards
        dpo_acc = float((margins > 0).mean()) * 100.0
        avg_margin = float(margins.mean())
        
        print("\n==================================================")
        print("📊 OFFLINE DPO VALIDATION RESULTS (Orca Domain)")
        print("==================================================")
        print(f"Validation Preference Accuracy: {dpo_acc:.2f}%")
        print(f"Average Reward Margin:           {avg_margin:.6f}")
        print("==================================================")
        
        # Save metrics
        dpo_val_metrics = {
            "validation_preference_accuracy": dpo_acc,
            "average_reward_margin": avg_margin
        }
        with open(os.path.join(BASE_DIR, "dpo_validation_analysis.json"), "w") as f:
            json.dump(dpo_val_metrics, f, indent=2)
            
    except Exception as e:
        print(f"Skipping or failed DPO Validation Analysis: {e}")
        
    save_state(phase="report_done", notes="progression_chart.png, final_comparison.csv, outputs saved.")
    print("\nReport generation complete.")
