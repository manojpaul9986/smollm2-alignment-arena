# Action Plan: Diagnose & Fix Near-Identical SFT/DPO Results

## Context (paste this for the agent)

Evaluation of three checkpoints (base, SFT, DPO — SmolLM2-1.7B, LoRA adapters) shows SFT and DPO
have nearly identical metrics:

| Metric | Base | SFT | DPO |
|---|---|---|---|
| Perplexity | 42.50 | 47.08 | 47.05 |
| Instruction-follow acc % | 40.0 | 100.0 | 100.0 |
| GSM8K acc % | 8.0 | 37.0 | 37.0 |
| CoT presence % | 55.0 | 15.0 | 15.0 |

Diagnostic: of 10 saved GSM8K samples, **7/10 responses are byte-identical text** between SFT and
DPO under greedy decoding, and perplexity differs by only 0.036. This means DPO training *did*
change the weights slightly (not a total no-op / not a pipeline bug reusing the same checkpoint),
but the shift is too small to show up as a real capability change. Most likely: undertrained DPO
(too few steps/pairs, beta too high, or LR too low), not a broken pipeline.

Goal: confirm the root cause, fix it, retrain DPO, and re-evaluate — while also fixing two
eval-methodology issues found in the same pass.

---

## Task 1 — Confirm root cause from existing artifacts (no GPU / no retraining needed)

- [ ] **1.1** Locate and print the DPO training logs (loss, reward/margin accuracy, reward margin
  over steps). If TRL's `DPOTrainer` was used with `report_to="none"` and no logs were saved,
  check for a `trainer_state.json` in the DPO checkpoint dir — it contains the loss history.
  - Pass/fail check: if reward margin stayed near 0 and reward accuracy stayed near 50% the whole
    run, this **confirms undertraining** as the cause.
- [ ] **1.2** Diff the SFT and DPO LoRA adapter weights directly:
  ```python
  from safetensors.torch import load_file
  import torch

  sft_w = load_file("checkpoints/sft/adapter_model.safetensors")
  dpo_w = load_file("checkpoints/dpo/adapter_model.safetensors")

  total_norm_diff = 0.0
  for k in sft_w:
      if k in dpo_w:
          diff = (sft_w[k].float() - dpo_w[k].float()).norm().item()
          total_norm_diff += diff
          print(f"{k:60s} diff_norm={diff:.6f}")
  print(f"\nTOTAL adapter weight diff norm: {total_norm_diff:.6f}")
  ```
  - If this total is very small relative to the weight norms themselves, that's independent
    confirmation of undertraining (not just an eval-output artifact).
- [ ] **1.3** Check the DPO training config actually used (not just `config.py` defaults — the
  actual run args/logged hyperparameters): `beta`, `learning_rate`, `num_train_epochs` or
  `max_steps`, number of preference pairs in the training dataset, `per_device_train_batch_size` ×
  `gradient_accumulation_steps` (effective batch size).
- [ ] **1.4** Inspect the preference dataset itself: sample 10–15 `(prompt, chosen, rejected)`
  pairs and eyeball whether chosen/rejected are meaningfully different, or near-duplicates with
  trivial differences (low signal → weak gradient).

**Output of Task 1:** a short note stating which of the following applies:
(a) too few steps/pairs, (b) beta too high, (c) LR too low, (d) low-signal preference pairs,
or some combination — with the supporting numbers from 1.1–1.4.

---

## Task 2 — Fix and retrain DPO

- [ ] **2.1** Based on Task 1 findings, adjust config (`config.py` or DPO training notebook):
  - If beta was ≥0.3 → lower to **0.1** (0.05–0.1 is typical for LoRA DPO on small models).
  - If LR was <5e-6 or unset/default → set to **5e-6 to 1e-5**.
  - If training was 1 epoch on <1,000 pairs → increase epochs to **2–3**, and/or increase the
    preference dataset size if feasible.
  - If preference pairs were low-signal (near-duplicate chosen/rejected) → regenerate the dataset
    with a clearer quality gap (e.g., sample completions at higher temperature to get more
    divergent candidates before scoring/labeling chosen vs rejected).
- [ ] **2.2** Re-run DPO training with the updated config. Save `trainer_state.json` / log the loss
  and reward-margin curve this time (do not skip logging — needed to confirm the fix worked).
- [ ] **2.3** Sanity-check immediately after training, before re-running full eval: repeat the
  Task 1.2 weight-diff check on the new adapter vs SFT. Confirm the diff norm increased
  meaningfully vs the old DPO adapter.

---

## Task 3 — Fix eval methodology issues (independent of retraining, do in parallel)

- [ ] **3.1** Fix `cot_presence_pct` in `eval_utils.py` — it currently fires on repeated `"Step N:"`
  patterns from degenerate repetition loops (this is why base model scored 55% "CoT presence" vs
  15% for SFT/DPO, despite base being far less accurate and mostly just looping). Add the same
  `has_repetition` check already used in `score_instruction_following` before counting a CoT
  marker as valid:
  ```python
  # in score_gsm8k, before counting cot_markers:
  lines = [l.strip().lower() for l in resp.split("\n") if len(l.strip()) > 3]
  has_repetition = len(lines) != len(set(lines))
  has_cot = bool(re.search(r"(step\s*\d|first,|then,|next,|\n\d\.)", resp, re.I)) and not has_repetition
  ```
- [ ] **3.2** Expand `INSTR_PROMPTS` beyond 10 easy prompts — it's ceiling-saturated at 100% for
  both SFT and DPO, so it currently has zero ability to detect whether DPO helped or hurt
  instruction-following. Add 15–20 more prompts, including harder/multi-constraint ones (e.g.
  "Answer in exactly two sentences," "List exactly 4 items, no more no less," "Respond only in
  JSON with keys x and y").
- [ ] **3.3** Add a permanent diagnostic metric to the eval pipeline: **% of outputs identical to
  the previous stage** (SFT vs DPO, and base vs SFT), computed over the full sample set, not just
  the 10 saved ones. Add this to `run_full_eval` / the comparison report so a near-zero DPO effect
  is visible immediately in future runs without manual JSON diffing:
  ```python
  def pct_identical(samples_a, samples_b):
      matches = sum(a["response"] == b["response"] for a, b in zip(samples_a, samples_b))
      return 100 * matches / len(samples_a)
  ```
- [ ] **3.4** Improve `extract_final_number` in `eval_utils.py` — it currently regex-grabs the
  *last* number in the full response, which can pick up trailing/unrelated numbers if the model
  rambles past its stated answer. Replace with either (a) a stricter regex anchored to
  `"####"` / `"final answer"` / `"answer is"` patterns, or (b) an LLM-judge re-extraction call
  (cheap, you already have the Gemini judge wired up).

---

## Task 4 — Re-run full evaluation and compare

- [ ] **4.1** Re-run `run_full_eval` for base / SFT / new DPO checkpoint using the fixed
  `eval_utils.py` from Task 3.
- [ ] **4.2** Regenerate `final_comparison.csv` and `progression_chart.png` via the Step 7
  comparison cell.
- [ ] **4.3** Confirm the outcome: DPO metrics should now differ from SFT (in either direction —
  improvement is the hoped-for outcome, but even a *worse* DPO score is a legitimate, reportable
  result now that we know it reflects real training signal rather than an undertrained adapter).
- [ ] **4.4** Re-check the "% identical outputs" diagnostic (Task 3.3) — expect a meaningfully
  lower overlap between SFT and DPO than the ~70% seen before.

---

## Task 5 — Only after Task 4 confirms a real DPO effect: process-vs-outcome GSM8K scoring

(Deferred — no point analyzing reasoning quality on a DPO run that barely moved the policy.)

- [ ] **5.1** Add LLM-judge scoring that separates `process_correct` (is the reasoning chain
  logically valid) from `outcome_correct` (does the final number match gold), using Gemini judge
  already integrated in `06_compare_and_report.ipynb`.
- [ ] **5.2** Report the 2×2 breakdown (process✓/outcome✓, process✓/outcome✗, process✗/outcome✓,
  process✗/outcome✗) for SFT vs DPO to characterize *how* DPO changed reasoning, not just whether
  final-answer accuracy moved.

---

## Priority order summary

1. Task 1 (diagnose) — cheapest, no GPU, do first.
2. Task 3 (fix eval bugs) — no GPU, can run in parallel with Task 1.
3. Task 2 (fix + retrain DPO) — needs GPU, do once Task 1 identifies the likely cause.
4. Task 4 (re-evaluate) — needs GPU, after Task 2 + 3 complete.
5. Task 5 (process-vs-outcome analysis) — only after Task 4 confirms DPO has a real, non-trivial
   effect on the policy.
