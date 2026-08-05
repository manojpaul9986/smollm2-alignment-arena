# =====================================================================
# 06_compare_and_report.ipynb
# Goal: consolidate all metrics, run qualitative evaluation,
# and query the Gemini API Judge to compute model Win Rates.
# Runtime: ~3-5 min
# =====================================================================

### CELL 1: load all metrics ###
# !pip install -q -U transformers trl datasets accelerate peft evaluate bitsandbytes matplotlib pandas

if __name__ == "__main__":
    from config import load_metrics, save_state
    import pandas as pd
    import matplotlib.pyplot as plt
    
    save_state(phase="report_start")
    
    baseline = load_metrics("baseline")
    sft = load_metrics("sft")
    dpo = load_metrics("dpo")
    
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
    df.to_csv("/kaggle/working/final_comparison.csv", index=False)
    
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
    plt.savefig("/kaggle/working/progression_chart.png", dpi=150)
    plt.close()
    
    ### CELL 3: generate outputs for qualitative showcase ###
    from eval_utils import load_model_and_tokenizer, generate, get_gemini_judge_score
    from config import MODEL_ID, SFT_OUT_DIR, DPO_OUT_DIR, GEMINI_API_KEY
    import torch
    
    showcase_prompts = [
        "If a train travels 60 miles in 1.5 hours, what is its average speed?",
        "Explain in simple terms why the sky is blue.",
        "A store has 120 apples. They sell 45 in the morning and 30 in the afternoon. How many are left?",
        "What are three tips for staying focused while studying?",
        "If I have $50 and spend 40% of it, how much do I have left?",
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
            resp = generate(model, tok, p, chat=chat, max_new_tokens=150)
            responses[p][label] = resp
        del model
        torch.cuda.empty_cache()
    
    ### CELL 4: LLM-as-a-Judge Evaluation (Win Rate) ###
    judge_results = []
    win_counts = {"Base": 0, "SFT": 0, "DPO": 0, "Tie": 0, "Error": 0}
    
    if GEMINI_API_KEY:
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
        judge_df.to_csv("/kaggle/working/llm_judge_results.csv", index=False)
        
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
        print("\n[Notice] GEMINI_API_KEY environment variable not found. Skipping LLM-as-a-Judge evaluation.")
        # Just save regular outputs without judge evaluation
        showcase_rows = []
        for prompt, resps in responses.items():
            showcase_rows.append({
                "prompt": prompt,
                "Base": resps["Base"],
                "SFT": resps["SFT"],
                "DPO": resps["DPO"]
            })
        pd.DataFrame(showcase_rows).to_csv("/kaggle/working/qualitative_showcase.csv", index=False)
        for prompt in showcase_prompts:
            print(f"\n{'='*80}\nPROMPT: {prompt}\n{'='*80}")
            for label in ["Base", "SFT", "DPO"]:
                print(f"\n[{label}]\n{responses[prompt][label]}")
                
    save_state(phase="report_done", notes="progression_chart.png, final_comparison.csv, outputs saved.")
    print("\nReport generation complete.")
