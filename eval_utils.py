"""
eval_utils.py
One evaluation harness reused identically for base / SFT / DPO models so the
comparison in 06_compare_and_report is apples-to-apples. Supports PEFT loading and Gemini Judge.
"""

import os
import re
import json
import torch
import urllib.request
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(model_path: str, is_chat: bool = True):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        tok.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
        )
        
    # Check if model_path is a PEFT/LoRA adapter (has adapter_config.json)
    is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    if is_peft:
        # Load the base model first, then the PEFT wrapper
        from config import MODEL_ID, SFT_OUT_DIR, DPO_OUT_DIR
        from peft import PeftModel
        
        # If loading DPO adapter, the base model MUST be the merged SFT model
        if os.path.abspath(model_path) == os.path.abspath(DPO_OUT_DIR):
            print("Loading DPO adapter on top of SFT-merged base model...")
            raw_base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.float16
            ).to(DEVICE)
            sft_model = PeftModel.from_pretrained(raw_base, SFT_OUT_DIR).to(DEVICE)
            base_model = sft_model.merge_and_unload()
        else:
            # For SFT, the base is the raw base model
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.float16
            ).to(DEVICE)
            
        model = PeftModel.from_pretrained(base_model, model_path).to(DEVICE)
    else:
        # Load standard model
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to(DEVICE)
        
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, prompt: str, chat: bool = True, max_new_tokens: int = 200):
    if chat and tok.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        inputs = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    else:
        inputs = tok(prompt, return_tensors="pt")

    # Extract input_ids tensor
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

    # Identify EOS tokens (standard eos + ChatML end token)
    eos_token_ids = [tok.eos_token_id]
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tok.unk_token_id:
        eos_token_ids.append(im_end_id)

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tok.pad_token_id,
        eos_token_id=eos_token_ids,
    )
    text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip()


# ---------------------------------------------------------------------
# 1. Perplexity on a small held-out split (catastrophic-forgetting check)
# ---------------------------------------------------------------------
@torch.no_grad()
def compute_perplexity(model, tok, n_samples: int = 200, max_len: int = 512):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ds = ds.filter(lambda x: len(x["text"].strip()) > 20).select(range(n_samples))

    nlls = []
    for idx, row in enumerate(ds):
        if idx > 0 and idx % 50 == 0:
            print(f"  [Perplexity] Processed {idx}/{n_samples} samples...")
        enc = tok(row["text"], return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = enc.input_ids.to(DEVICE)
        if input_ids.shape[1] < 2:
            continue
        out = model(input_ids, labels=input_ids)
        nlls.append(out.loss.item())
    ppl = float(torch.exp(torch.tensor(nlls).mean()))
    return ppl


# ---------------------------------------------------------------------
# 2. Instruction-following
# ---------------------------------------------------------------------
INSTR_PROMPTS = [
    "List three benefits of regular exercise.",
    "Translate 'good morning' into French.",
    "Write a one-sentence summary of what photosynthesis is.",
    "Give me two synonyms for 'happy'.",
    "Explain what a variable is in programming, in one sentence.",
    "Name the capital of Japan.",
    "Convert 10 kilometers to miles.",
    "Write a short greeting for a birthday card.",
    "What is the boiling point of water in Celsius?",
    "Give a one-line definition of gravity.",
]


def score_instruction_following(model, tok, chat: bool = True):
    hits = 0
    outputs = []
    for idx, p in enumerate(INSTR_PROMPTS):
        resp = generate(model, tok, p, chat=chat, max_new_tokens=100)
        outputs.append({"prompt": p, "response": resp})
        ok = (
            len(resp.strip()) > 3
            and resp.strip().lower() not in p.lower()
            and len(set(resp.split())) > max(3, len(resp.split()) * 0.3)
        )
        if ok:
            hits += 1
    return {"accuracy_pct": 100 * hits / len(INSTR_PROMPTS), "samples": outputs}


# ---------------------------------------------------------------------
# 3. GSM8K subset: reasoning accuracy via exact-match on final number
# ---------------------------------------------------------------------
def extract_final_number(text: str):
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return matches[-1] if matches else None


def score_gsm8k(model, tok, n_samples: int = 120, chat: bool = True):
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n_samples))
    correct = 0
    cot_markers = 0
    outputs = []
    for idx, row in enumerate(ds):
        if idx > 0 and idx % 10 == 0:
            print(f"  [GSM8K] Solved {idx}/{n_samples} questions (Accuracy so far: {100 * correct / idx:.1f}%)...")
        q = row["question"]
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        resp = generate(
            model, tok,
            f"{q}\nSolve step by step and give the final numeric answer.",
            chat=chat, max_new_tokens=384,
        )
        pred = extract_final_number(resp)
        is_correct = pred is not None and pred == gold
        has_cot = bool(re.search(r"(step\s*\d|first,|then,|next,|\n\d\.)", resp, re.I))
        correct += int(is_correct)
        cot_markers += int(has_cot)
        outputs.append({"question": q, "gold": gold, "pred": pred, "response": resp})
    return {
        "accuracy_pct": 100 * correct / n_samples,
        "cot_presence_pct": 100 * cot_markers / n_samples,
        "samples": outputs[:10],
    }


# ---------------------------------------------------------------------
# 4. Full eval bundle for one checkpoint
# ---------------------------------------------------------------------
def run_full_eval(model_path: str, chat: bool = True, ppl_n=200, instr_n=None, gsm8k_n=120):
    model, tok = load_model_and_tokenizer(model_path, is_chat=chat)
    print(f"Computing perplexity on {ppl_n} wikitext-2 samples...")
    ppl = compute_perplexity(model, tok, n_samples=ppl_n)
    print(f"Scoring instruction following ({len(INSTR_PROMPTS)} prompts)...")
    instr = score_instruction_following(model, tok, chat=chat)
    print(f"Scoring GSM8K reasoning on {gsm8k_n} problems...")
    gsm8k = score_gsm8k(model, tok, n_samples=gsm8k_n, chat=chat)

    del model
    torch.cuda.empty_cache()

    return {
        "perplexity": ppl,
        "instruction_following_acc_pct": instr["accuracy_pct"],
        "gsm8k_acc_pct": gsm8k["accuracy_pct"],
        "cot_presence_pct": gsm8k["cot_presence_pct"],
        "instr_samples": instr["samples"],
        "gsm8k_samples": gsm8k["samples"],
    }


# ---------------------------------------------------------------------
# 5. Gemini API Judge (LLM-as-a-Judge)
# ---------------------------------------------------------------------
def get_gemini_judge_score(prompt: str, base_resp: str, sft_resp: str, dpo_resp: str) -> dict:
    from config import GEMINI_API_KEY
    if not GEMINI_API_KEY:
        return {"winner": "Tie", "reason": "No Gemini API key configured."}
    
    judge_prompt = f"""
    You are an expert AI evaluator. Compare three responses generated by different models (Base, SFT, DPO) to the user's prompt.
    Evaluate their quality based on helpfulness, accuracy, structure, instruction following, and step-by-step reasoning quality.
    
    [USER PROMPT]:
    {prompt}
    
    [BASE RESPONSE]:
    {base_resp}
    
    [SFT RESPONSE]:
    {sft_resp}
    
    [DPO RESPONSE]:
    {dpo_resp}
    
    Respond ONLY with a JSON object in this format:
    {{
      "winner": "Base" | "SFT" | "DPO" | "Tie",
      "reason": "<one sentence explanation of why the winner was chosen>"
    }}
    """
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    data = {
        "contents": [{
            "parts": [{"text": judge_prompt}]
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            res = json.loads(response.read().decode("utf-8"))
            text = res["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text.strip())
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return {"winner": "Error", "reason": str(e)}
