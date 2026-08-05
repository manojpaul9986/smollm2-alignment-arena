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
        LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES
    )
    set_seed()
    save_state(phase="dpo_train_start")
    
    ### CELL 3: load + filter + subsample preference pairs ###
    from datasets import load_dataset
    
    raw = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train")
    
    TARGET_N = 1000
    MAX_CHARS = 2500
    
    def is_cot_signal_and_short(ex):
        chosen, rejected = ex["chosen"], ex["rejected"]
        if len(chosen) > MAX_CHARS or len(rejected) > MAX_CHARS:
            return False
        chosen_is_longer = len(chosen) > len(rejected) * 1.1
        return chosen_is_longer
    
    filtered = raw.filter(is_cot_signal_and_short, num_proc=4)
    print(f"Filtered pool: {len(filtered)} (from {len(raw)})")
    
    filtered = filtered.shuffle(seed=42)
    dpo_train = filtered.select(range(min(TARGET_N, len(filtered))))
    dpo_eval = filtered.select(range(TARGET_N, TARGET_N + 100))
    
    def to_dpo_format(ex):
        system_prompt = ex.get("system", "")
        user_prompt = ex.get("input", "") or ex.get("question", "")
        
        prompt_msgs = []
        if system_prompt:
            prompt_msgs.append({"role": "system", "content": system_prompt})
        prompt_msgs.append({"role": "user", "content": user_prompt})
        
        chosen_msgs = [{"role": "assistant", "content": ex["chosen"]}]
        rejected_msgs = [{"role": "assistant", "content": ex["rejected"]}]
        
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
    
    tokenizer = AutoTokenizer.from_pretrained(SFT_OUT_DIR)
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
        num_train_epochs=1,
        per_device_train_batch_size=1,       # smaller batch size for 1.7B DPO (requires 2 forward passes per pair)
        gradient_accumulation_steps=16,      # effective batch size = 16
        learning_rate=1e-6,                  # lowered from 5e-6 to prevent policy collapse
        beta=0.1,
        max_length=1024,
        logging_steps=20,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=50,
        fp16=True,
        max_grad_norm=0.5,                   # gradient clipping to prevent exploding gradients in float16
        warmup_ratio=0.1,                    # longer warmup for early stability
        disable_dropout=True,                # standard DPO practice to prevent policy divergence
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
        if PUSH_TO_HUB:
            trainer.push_to_hub()
        save_state(phase="dpo_train_done", notes=f"saved adapter to {DPO_OUT_DIR}")
        print("DPO training complete.")
