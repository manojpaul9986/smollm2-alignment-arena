"""
app.py
Gradio side-by-side Chatbot Arena for SmolLM2-1.7B (Base vs SFT vs DPO).
Loads each model path cleanly to resolve correct adapter base dependencies.
"""

import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B"
BASE_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "./work"
SFT_OUT_DIR = os.path.join(BASE_DIR, "checkpoints/sft")
DPO_OUT_DIR = os.path.join(BASE_DIR, "checkpoints/dpo")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer...")
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

print("Loading Base model...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)

print("Loading SFT model...")
has_sft = os.path.exists(os.path.join(SFT_OUT_DIR, "adapter_config.json"))
if has_sft:
    sft_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    sft_model = PeftModel.from_pretrained(sft_base, SFT_OUT_DIR).to(DEVICE)
else:
    sft_model = None
    print("Warning: SFT adapter not found.")

print("Loading DPO model...")
has_dpo = os.path.exists(os.path.join(DPO_OUT_DIR, "adapter_config.json"))
if has_dpo and has_sft:
    # DPO base model must be the SFT merged model
    dpo_base_raw = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    dpo_sft = PeftModel.from_pretrained(dpo_base_raw, SFT_OUT_DIR).to(DEVICE)
    dpo_base = dpo_sft.merge_and_unload()
    
    dpo_model = PeftModel.from_pretrained(dpo_base, DPO_OUT_DIR).to(DEVICE)
else:
    dpo_model = None
    print("Warning: DPO adapter not found.")


def generate_response(model_type, prompt, max_tokens, temperature):
    use_chat = model_type in ["SFT", "DPO"]
    
    if use_chat:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
        
    # Ultra-defensive extraction of input_ids tensor
    if isinstance(inputs, dict) or hasattr(inputs, "keys") or hasattr(inputs, "data"):
        input_ids = inputs["input_ids"]
    else:
        input_ids = inputs

    # Ensure input_ids is a torch.Tensor
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)

    input_ids = input_ids.to(DEVICE)

    # Ensure 2D tensor (batch_size, sequence_length)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    
    # Active model selection
    if model_type == "Base":
        active_model = base_model
    elif model_type == "SFT":
        active_model = sft_model if sft_model is not None else base_model
    elif model_type == "DPO":
        active_model = dpo_model if dpo_model is not None else (sft_model if sft_model is not None else base_model)
    else:
        active_model = base_model

    with torch.no_grad():
        out = active_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 0.01),
            pad_token_id=tokenizer.pad_token_id
        )
        
    response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def chatbot_arena(prompt, max_tokens, temperature):
    base_out = generate_response("Base", prompt, max_tokens, temperature)
    sft_out = generate_response("SFT", prompt, max_tokens, temperature)
    dpo_out = generate_response("DPO", prompt, max_tokens, temperature)
    return base_out, sft_out, dpo_out


# Build UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚔️ SmolLM2-1.7B Alignment Arena")
    gr.Markdown("Compare the progression of the **SmolLM2-1.7B** model through **Base**, **Supervised Fine-Tuning (SFT)**, and **Direct Preference Optimization (DPO)** side-by-side.")
    
    with gr.Row():
        with gr.Column(scale=4):
            prompt_input = gr.Textbox(
                label="Enter your prompt",
                placeholder="e.g., Explain why the sky is blue step by step.",
                lines=3
            )
            submit_btn = gr.Button("Generate Responses", variant="primary")
            
        with gr.Column(scale=1):
            max_tokens_slider = gr.Slider(minimum=32, maximum=512, value=150, step=1, label="Max New Tokens")
            temp_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.05, label="Temperature (0 = Greedy)")
            
    with gr.Row():
        base_output = gr.Textbox(label="1. Base Model (Raw Completion)", lines=10)
        sft_output = gr.Textbox(label="2. SFT Model (Instruction Aligned)", lines=10)
        dpo_output = gr.Textbox(label="3. DPO Model (Reasoning Preferred)", lines=10)
        
    submit_btn.click(
        fn=chatbot_arena,
        inputs=[prompt_input, max_tokens_slider, temp_slider],
        outputs=[base_output, sft_output, dpo_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
