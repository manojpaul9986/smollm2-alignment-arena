# =====================================================================
# 02_sft_train.ipynb
# Goal: teach ChatML structure + instruction following using LoRA on SmolLM2-1.7B.
# Data: HuggingFaceTB/smoltalk, 4,000 samples (filtered to short convos)
# Runtime: ~15-20 min on 1xT4
# =====================================================================

### CELL 1: install deps ###
# !pip install -q -U transformers trl datasets accelerate peft bitsandbytes

if __name__ == "__main__":
    ### CELL 2: config ###
    from config import (
        set_seed, save_state, find_latest_checkpoint, MODEL_ID, SFT_OUT_DIR, PUSH_TO_HUB, HUB_SFT_REPO,
        LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES
    )
    set_seed()
    save_state(phase="sft_train_start")
    
    ### CELL 3: load + filter + subsample dataset ###
    from datasets import load_dataset
    
    raw = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
    
    TARGET_N = 4000          
    MAX_TURNS = 6             
    MAX_CHARS = 2000          
    
    def is_short_and_clean(ex):
        msgs = ex["messages"]
        if len(msgs) == 0 or len(msgs) > MAX_TURNS:
            return False
        total_chars = sum(len(m["content"]) for m in msgs)
        return total_chars <= MAX_CHARS
    
    filtered = raw.filter(is_short_and_clean, num_proc=4)
    print(f"Filtered pool: {len(filtered)} (from {len(raw)})")
    
    filtered = filtered.shuffle(seed=42)
    sft_train = filtered.select(range(min(TARGET_N, len(filtered))))
    sft_eval = filtered.select(range(TARGET_N, TARGET_N + 200))  
    
    print(f"Train: {len(sft_train)} | Eval: {len(sft_eval)}")
    
    ### CELL 4: tokenizer + model ###
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    
    ### CELL 5: SFTTrainer config with LoRA (resume-ready) ###
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig
    
    resume_ckpt = find_latest_checkpoint(SFT_OUT_DIR)
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
    
    sft_config = SFTConfig(
        output_dir=SFT_OUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,       
        gradient_accumulation_steps=8,       
        learning_rate=1e-4,                  # LoRA typically needs a higher learning rate (e.g., 1e-4 to 2e-4) than full tuning
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_length=1024,
        logging_steps=20,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,                  
        eval_strategy="steps",
        eval_steps=100,
        fp16=True,
        report_to="none",
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=HUB_SFT_REPO if PUSH_TO_HUB else None,
        dataset_text_field=None,             
        packing=False,
        seed=42,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=sft_train,
        eval_dataset=sft_eval,
        peft_config=peft_config,             # Pass the LoRA config
        processing_class=tokenizer,
    )
    
    ### CELL 6: train (resumes automatically if a checkpoint exists) ###
    trainer.train(resume_from_checkpoint=resume_ckpt)
    
    ### CELL 7: save final model + tokenizer ###
    trainer.save_model(SFT_OUT_DIR)
    
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(SFT_OUT_DIR)
        if PUSH_TO_HUB:
            trainer.push_to_hub()
        save_state(phase="sft_train_done", notes=f"saved adapter to {SFT_OUT_DIR}")
        print("SFT training complete.")
