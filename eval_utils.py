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
    try:
        tok = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        from config import MODEL_ID
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        
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
    # Multi-constraint and harder prompts
    "Answer the following question in exactly two sentences: What is the main cause of ocean tides?",
    "List exactly 4 items you would need to build a wooden birdhouse, no more and no less.",
    "Explain the concept of inflation to a 5-year-old in one paragraph of exactly three sentences.",
    "Respond ONLY with a JSON object containing the keys 'capital' and 'population' for the country France.",
    "Write a haiku about artificial intelligence.",
    "Give me a list of three distinct colors, but do not include the color blue or red.",
    "Tell me the name of the largest ocean on Earth, and write it in all capital letters.",
    "Provide two antonyms for the word 'cold', separated by a semicolon.",
    "In exactly five words, describe what a book is.",
    "What is the result of 15 multiplied by 4, plus 12? Provide only the final number.",
    "List three different programming languages that are statically typed.",
    "Write a short sentence where every word starts with the letter 'S'.",
    "Who wrote the play Romeo and Juliet? Answer with only the author's name.",
    "What are the three physical states of matter? List them separated by commas.",
    "State the name of the chemical symbol for Gold in exactly one word."
]


def score_instruction_following(model, tok, chat: bool = True):
    hits = 0
    outputs = []
    for idx, p in enumerate(INSTR_PROMPTS):
        resp = generate(model, tok, p, chat=chat, max_new_tokens=100)
        outputs.append({"prompt": p, "response": resp})
        
        # Check for repetition loops or ChatML collapses
        has_chat_collapse = any(marker in resp.lower() for marker in ["user\n", "assistant\n", "<|im_start|>", "<|im_end|>"])
        
        # Check for repeated lines of substantial length
        lines = [line.strip().lower() for line in resp.split("\n") if len(line.strip()) > 3]
        has_repetition = len(lines) != len(set(lines))
        
        resp_lower = resp.lower().strip()
        is_correct = False
        
        # Simple heuristic check for prompt-specific accuracy
        if "exercise" in p.lower():
            is_correct = any(w in resp_lower for w in ["benefit", "health", "weight", "heart", "cardio", "mental", "sleep", "energy"]) and len(resp_lower) > 20
        elif "french" in p.lower():
            is_correct = "bonjour" in resp_lower
        elif "photosynthesis" in p.lower():
            is_correct = "photosynthesis" in resp_lower or ("plant" in resp_lower and "light" in resp_lower)
        elif "synonyms for 'happy'" in p.lower():
            is_correct = any(w in resp_lower for w in ["joyful", "content", "cheerful", "glad", "delighted", "pleased", "merry", "elated"])
        elif "variable" in p.lower():
            is_correct = "variable" in resp_lower or "container" in resp_lower or "value" in resp_lower
        elif "japan" in p.lower():
            is_correct = "tokyo" in resp_lower
        elif "kilometers" in p.lower():
            is_correct = "6.2" in resp_lower or "6" in resp_lower
        elif "birthday" in p.lower():
            is_correct = "birthday" in resp_lower or "happy" in resp_lower or "greet" in resp_lower or "wish" in resp_lower
        elif "boiling point" in p.lower():
            is_correct = "100" in resp_lower
        elif "gravity" in p.lower():
            is_correct = "force" in resp_lower or "attract" in resp_lower or "gravity" in resp_lower
        elif "ocean tides" in p.lower():
            sentences = [s.strip() for s in re.split(r'[.!?]', resp.strip()) if s.strip()]
            is_correct = len(sentences) == 2 and any(w in resp_lower for w in ["moon", "gravit", "tide", "pull"])
        elif "birdhouse" in p.lower():
            is_correct = any(w in resp_lower for w in ["wood", "nail", "screw", "glue", "hammer", "saw", "paint"]) and ("1" in resp_lower or "2" in resp_lower or "3" in resp_lower or "4" in resp_lower or "-" in resp_lower or "\n" in resp_lower)
        elif "inflation" in p.lower():
            sentences = [s.strip() for s in re.split(r'[.!?]', resp.strip()) if s.strip()]
            is_correct = len(sentences) == 3 and any(w in resp_lower for w in ["price", "money", "buy", "rise", "cost"])
        elif "json" in p.lower():
            is_correct = "capital" in resp_lower and "population" in resp_lower and "paris" in resp_lower
        elif "haiku" in p.lower():
            lines_count = len([l for l in resp.split("\n") if l.strip()])
            is_correct = lines_count >= 2 and any(w in resp_lower for w in ["ai", "mind", "computer", "machine", "think", "code"])
        elif "distinct colors" in p.lower():
            is_correct = not any(w in resp_lower for w in ["blue", "red"]) and any(w in resp_lower for w in ["green", "yellow", "orange", "purple", "black", "white", "pink", "brown"])
        elif "largest ocean" in p.lower():
            is_correct = "PACIFIC" in resp
        elif "antonyms for the word 'cold'" in p.lower():
            is_correct = ";" in resp and any(w in resp_lower for w in ["hot", "warm", "heat"])
        elif "five words" in p.lower():
            words = [w for w in resp.split() if w.strip()]
            is_correct = len(words) == 5 and any(w in resp_lower for w in ["book", "read", "page", "story", "paper"])
        elif "15 multiplied by 4" in p.lower():
            is_correct = "72" in resp_lower
        elif "statically typed" in p.lower():
            is_correct = any(w in resp_lower for w in ["java", "c++", "c#", "rust", "go", "typescript", "swift", "kotlin", "scala"])
        elif "starts with the letter 's'" in p.lower():
            words = [w.strip(".,!?\"'") for w in resp_lower.split() if w.strip()]
            is_correct = len(words) >= 3 and all(w.startswith("s") for w in words)
        elif "romeo and juliet" in p.lower():
            is_correct = "shakespeare" in resp_lower
        elif "states of matter" in p.lower():
            is_correct = "solid" in resp_lower and "liquid" in resp_lower and "gas" in resp_lower
        elif "symbol for gold" in p.lower():
            is_correct = "au" in resp_lower
            
        ok = is_correct and not has_chat_collapse and not has_repetition
        if ok:
            hits += 1
    return {"accuracy_pct": 100 * hits / len(INSTR_PROMPTS), "samples": outputs}


# ---------------------------------------------------------------------
# 3. GSM8K subset: reasoning accuracy via exact-match on final number
# ---------------------------------------------------------------------
def extract_final_number(text: str):
    # Anchor to #### match first to prevent trailing text pollution
    ans_match = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if ans_match:
        return ans_match.group(1).replace(",", "")
    # Fallback to standard regex match for raw/chat completion outputs
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
        
        # Check for repetition loops
        lines = [line.strip().lower() for line in resp.split("\n") if len(line.strip()) > 3]
        has_repetition = len(lines) != len(set(lines))
        
        has_cot = bool(re.search(r"(step\s*\d|first,|then,|next,|\n\d\.)", resp, re.I)) and not has_repetition
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
    from config import GEMINI_API_KEY, GROQ_API_KEY
    if not GEMINI_API_KEY and not GROQ_API_KEY:
        return {"winner": "Tie", "reason": "No Gemini or Groq API key configured."}
    
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
    
    # 1. Try Groq first if available
    if GROQ_API_KEY:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "User-Agent": "Mozilla/5.0"
        }
        models_to_try = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
        for groq_model in models_to_try:
            data = {
                "model": groq_model,
                "messages": [
                    {"role": "user", "content": judge_prompt}
                ],
                "response_format": {"type": "json_object"}
            }
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode("utf-8"),
                    headers=headers
                )
                with urllib.request.urlopen(req, timeout=15) as response:
                    res = json.loads(response.read().decode("utf-8"))
                    text = res["choices"][0]["message"]["content"]
                    return json.loads(text.strip())
            except Exception:
                continue
            
    # 2. Try Gemini as fallback if available
    if GEMINI_API_KEY:
        gemini_models = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-1.5-flash"]
        for gem_m in gemini_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{gem_m}:generateContent?key={GEMINI_API_KEY}"
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
                    headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, timeout=15) as response:
                    res = json.loads(response.read().decode("utf-8"))
                    text = res["candidates"][0]["content"]["parts"][0]["text"]
                    return json.loads(text.strip())
            except Exception:
                continue
            
    return {"winner": "Error", "reason": "All API judge model calls failed or timed out."}


# ---------------------------------------------------------------------
# 6. Offline DPO Metrics Calculator (Option 2)
# ---------------------------------------------------------------------
@torch.no_grad()
def compute_model_logps_single(model, tok, prompt_msgs, completion_msgs):
    # Retrieve token ids for the prompt
    prompt_res = tok.apply_chat_template(prompt_msgs, add_generation_prompt=True)
    if isinstance(prompt_res, dict) or hasattr(prompt_res, "input_ids") or hasattr(prompt_res, "data"):
        prompt_ids = prompt_res["input_ids"]
    else:
        prompt_ids = prompt_res
        
    # Convert prompt_ids to a plain python list of integers
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    else:
        prompt_ids = list(prompt_ids)
        
    if len(prompt_ids) > 0 and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    
    # Retrieve completion text and tokenize it directly
    completion_text = completion_msgs[0]["content"]
    comp_res = tok.encode(completion_text, add_special_tokens=False)
    if isinstance(comp_res, dict) or hasattr(comp_res, "input_ids") or hasattr(comp_res, "data"):
        completion_ids = comp_res["input_ids"]
    else:
        completion_ids = comp_res
        
    # Convert completion_ids to a plain python list of integers
    if hasattr(completion_ids, "tolist"):
        completion_ids = completion_ids.tolist()
    else:
        completion_ids = list(completion_ids)
        
    if len(completion_ids) > 0 and isinstance(completion_ids[0], list):
        completion_ids = completion_ids[0]
    
    # Concatenate prompt + completion
    full_ids = prompt_ids + completion_ids
    
    # Ensure standard termination tag (im_end or eos) is present at the end
    eos_id = tok.eos_token_id
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tok.unk_token_id:
        if not full_ids or full_ids[-1] != im_end_id:
            full_ids.append(im_end_id)
    else:
        if not full_ids or full_ids[-1] != eos_id:
            full_ids.append(eos_id)
            
    # Convert list to PyTorch tensor
    full_ids_tensor = torch.tensor(full_ids)
    
    # Labels: set prompt tokens to -100 (ignored in loss computation)
    labels = full_ids_tensor.clone()
    labels[:len(prompt_ids)] = -100
    
    # Input batch formatting
    input_ids = full_ids_tensor.unsqueeze(0).to(DEVICE)
    labels = labels.unsqueeze(0).to(DEVICE)
    
    outputs = model(input_ids)
    logits = outputs.logits
    
    # Compute log probabilities
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_mask = shift_labels != -100
    shift_labels[shift_labels == -100] = 0
    
    log_probs = shift_logits.log_softmax(-1)
    per_token_logps = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Zero out ignored prompt tokens
    per_token_logps = per_token_logps * loss_mask
    return per_token_logps.sum(-1).item()


def compute_dataset_logps(model_path, dataset_samples, is_chat=True):
    model, tok = load_model_and_tokenizer(model_path, is_chat=is_chat)
    chosen_logps = []
    rejected_logps = []
    
    for idx, ex in enumerate(dataset_samples):
        if idx > 0 and idx % 20 == 0:
            print(f"  [Logps] Computed {idx}/{len(dataset_samples)} samples...")
        
        # Handle prompt
        if "prompt" in ex:
            prompt_val = ex["prompt"]
            if isinstance(prompt_val, list):
                prompt_msgs = prompt_val
            else:
                prompt_msgs = [{"role": "user", "content": str(prompt_val)}]
        else:
            system_prompt = ex.get("system", "")
            user_prompt = ex.get("input", "") or ex.get("question", "")
            prompt_msgs = []
            if system_prompt:
                prompt_msgs.append({"role": "system", "content": system_prompt})
            prompt_msgs.append({"role": "user", "content": user_prompt})
            
        # Handle chosen and rejected
        chosen_val = ex["chosen"]
        if isinstance(chosen_val, list):
            chosen_content = chosen_val[-1]["content"] if (len(chosen_val) > 0 and isinstance(chosen_val[-1], dict)) else str(chosen_val)
        else:
            chosen_content = str(chosen_val)
            
        rejected_val = ex["rejected"]
        if isinstance(rejected_val, list):
            rejected_content = rejected_val[-1]["content"] if (len(rejected_val) > 0 and isinstance(rejected_val[-1], dict)) else str(rejected_val)
        else:
            rejected_content = str(rejected_val)
            
        chosen_msgs = [{"role": "assistant", "content": chosen_content}]
        rejected_msgs = [{"role": "assistant", "content": rejected_content}]
        
        try:
            chosen_lp = compute_model_logps_single(model, tok, prompt_msgs, chosen_msgs)
            rejected_lp = compute_model_logps_single(model, tok, prompt_msgs, rejected_msgs)
            chosen_logps.append(chosen_lp)
            rejected_logps.append(rejected_lp)
        except Exception as e:
            # Handle template errors safely
            print(f"Error computing log-prob for index {idx}: {e}")
            chosen_logps.append(0.0)
            rejected_logps.append(0.0)
            
    del model
    torch.cuda.empty_cache()
    return chosen_logps, rejected_logps
