# =====================================================================
# 04_dpo_train.ipynb
# Goal: bias the SFT model toward step-by-step CoT over terse answers using LoRA.
# Data: argilla/distilabel-intel-orca-dpo-pairs, ~2,000 pairs
# Runtime: ~15-20 min on 1xT4
# =====================================================================

### CELL 1: install deps ###
# !pip install -q -U transformers trl datasets accelerate peft bitsandbytes

if __name__ == "__main__":
    ### CELL 2: config ###
    from config import (
        set_seed, save_state, find_latest_checkpoint, MODEL_ID, SFT_OUT_DIR, DPO_OUT_DIR, PUSH_TO_HUB, HUB_DPO_REPO,
        LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES, BASE_DIR
    )
    set_seed()
    save_state(phase="dpo_train_start")
    
    ### CELL 3: load + filter + subsample preference pairs ###
    from datasets import load_dataset
    
    print("Loading HuggingFaceH4/ultrafeedback_binarized...")
    raw = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    
    TARGET_N = 1500
    MAX_CHARS = 1200
    
    def is_high_quality_and_short(ex):
        chosen = ex["chosen"]
        rejected = ex["rejected"]
        chosen_text = chosen[-1]["content"] if isinstance(chosen, list) else str(chosen)
        rejected_text = rejected[-1]["content"] if isinstance(rejected, list) else str(rejected)
        
        if len(chosen_text) > MAX_CHARS or len(rejected_text) > MAX_CHARS:
            return False
            
        # Ensure a strong score margin (quality gap >= 1.0) between chosen & rejected
        score_c = ex.get("score_chosen", 0.0)
        score_r = ex.get("score_rejected", 0.0)
        if score_c is not None and score_r is not None:
            if (score_c - score_r) < 1.0:
                return False
                
        return True
    
    filtered = raw.filter(is_high_quality_and_short, num_proc=4)
    print(f"Filtered high-contrast UltraFeedback pool: {len(filtered)} (from {len(raw)})")
    
    filtered = filtered.shuffle(seed=42)
    dpo_train = filtered.select(range(min(TARGET_N, len(filtered))))
    
    # Load test_prefs for evaluation
    raw_test = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
    filtered_test = raw_test.filter(is_high_quality_and_short, num_proc=4).shuffle(seed=42)
    dpo_eval = filtered_test.select(range(min(100, len(filtered_test))))
    
    def to_dpo_format(ex):
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]
        
        if isinstance(prompt, list):
            prompt_msgs = prompt
        else:
            prompt_msgs = [{"role": "user", "content": str(prompt)}]
            
        if isinstance(chosen, list):
            if len(chosen) > 1 and chosen[-1]["role"] == "assistant":
                chosen_msgs = [{"role": "assistant", "content": chosen[-1]["content"]}]
            else:
                chosen_msgs = chosen
        else:
            chosen_msgs = [{"role": "assistant", "content": str(chosen)}]
            
        if isinstance(rejected, list):
            if len(rejected) > 1 and rejected[-1]["role"] == "assistant":
                rejected_msgs = [{"role": "assistant", "content": rejected[-1]["content"]}]
            else:
                rejected_msgs = rejected
        else:
            rejected_msgs = [{"role": "assistant", "content": str(rejected)}]
            
        return {
            "prompt": prompt_msgs,
            "chosen": chosen_msgs,
            "rejected": rejected_msgs,
        }
    
    dpo_train = dpo_train.map(to_dpo_format, remove_columns=dpo_train.column_names)
    dpo_eval = dpo_eval.map(to_dpo_format, remove_columns=dpo_eval.column_names)
    
    print(f"Train: {len(dpo_train)} | Eval: {len(dpo_eval)}")
    
    ### CELL 4: load SFT checkpoint as starting policy and merge weights ###
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig
    import torch
    import os
    
    # Ensure SFT checkpoint exists locally, or download it from Hugging Face Hub
    if not os.path.exists(SFT_OUT_DIR) or not os.listdir(SFT_OUT_DIR):
        print(f"SFT directory {SFT_OUT_DIR} not found locally. Downloading from Hugging Face Hub: manojpaul9986/smollm2-1.7b-sft-lora...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="manojpaul9986/smollm2-1.7b-sft-lora", local_dir=SFT_OUT_DIR)
        print("SFT adapter downloaded successfully.")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(SFT_OUT_DIR)
    except Exception:
        print(f"Loading tokenizer directly from base model {MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
        )
    
    # Load base model, load SFT LoRA adapters, and merge them
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    sft_model = PeftModel.from_pretrained(base_model, SFT_OUT_DIR)
    model = sft_model.merge_and_unload() # Clean unadapted starting policy for DPO
    
    ### CELL 5: DPOTrainer config with LoRA (resume-ready) ###
    from trl import DPOTrainer, DPOConfig
    
    resume_ckpt = find_latest_checkpoint(DPO_OUT_DIR)
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
        
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    dpo_config = DPOConfig(
        output_dir=DPO_OUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,       # smaller batch size for 1.7B DPO (requires 2 forward passes per pair)
        gradient_accumulation_steps=8,       # effective batch size = 8
        learning_rate=2e-5,                  # increased to 2e-5 to allow proper policy shift
        beta=0.05,                           # lowered from 0.1 to amplify DPO gradient signal
        max_length=512,                      # reduced from 1024 for faster execution & lower memory
        logging_steps=20,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=50,
        fp16=True,
        max_grad_norm=0.5,                   # gradient clipping to prevent exploding gradients in float16
        warmup_steps=20,                     # warmup steps compatible across all TRL versions
        report_to="none",
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=HUB_DPO_REPO if PUSH_TO_HUB else None,
        seed=42,
    )
    
    trainer = DPOTrainer(
        model=model,
        ref_model=None,                      # DPOTrainer will clone the unadapted merged model internally
        args=dpo_config,
        train_dataset=dpo_train,
        eval_dataset=dpo_eval,
        peft_config=peft_config,             # Pass the new DPO LoRA config
        processing_class=tokenizer,
    )
    
    ### CELL 6: train (resumes automatically if a checkpoint exists) ###
    trainer.train(resume_from_checkpoint=resume_ckpt)
    
    ### CELL 7: save final model ###
    trainer.save_model(DPO_OUT_DIR)
    
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(DPO_OUT_DIR)
        
        # Create a zip archive of the DPO adapter folder for easy download in Kaggle
        import shutil
        zip_path = os.path.join(BASE_DIR, "dpo_lora_adapter")
        print(f"Zipping DPO adapter folder {DPO_OUT_DIR} to {zip_path}.zip...")
        try:
            shutil.make_archive(zip_path, 'zip', DPO_OUT_DIR)
            print("Zipping complete. You can download dpo_lora_adapter.zip directly from Kaggle outputs.")
        except Exception as e:
            print(f"Error zipping adapter folder: {e}")
            
        if PUSH_TO_HUB:
            trainer.push_to_hub()
        save_state(phase="dpo_train_done", notes=f"saved adapter to {DPO_OUT_DIR}")
        print("DPO training complete.")
