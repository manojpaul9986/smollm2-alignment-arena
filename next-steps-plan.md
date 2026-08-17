# Next Steps: smollm2-alignment-arena

Priority-ordered plan to take this from "working pipeline" to "portfolio-strong project with a real finding."

---

## 1. Fix the trust-breakers first (30–60 min, no GPU needed)

These are the things a reviewer checks in the first minute — fix before anything else.

- [ ] **Recompute the headline % improvement.** 7→37 is ~429% relative; 15→37 is ~147%. Neither is 246%. Figure out what the 246% actually referred to (old run? different metric?) and correct the README, or drop the % framing and just show the table.
- [ ] **State GSM8K eval sample size.** If you evaluated on a subset (common on Kaggle time limits) instead of the full 1,319-question test set, say so — "37% on a 200-question subsample" reads very differently than "37% on full GSM8K."
- [ ] **Report eval seed / number of runs.** On a 1.7B model, single-run accuracy can swing several points. Even 2–3 seeds with a mean ± std bar makes the numbers far more credible.

---

## 2. Add the missing "why" — preference dataset description (1–2 hrs)

This is the single biggest gap. Right now the README never explains what your DPO chosen/rejected pairs actually were.

- [ ] Add a section to the README: dataset source (existing, e.g. UltraFeedback/distilabel-math-preference, vs self-generated), how pairs were built, how many pairs, and what "chosen" was scored on (correctness? style? both?).
- [ ] If pairs were **math-only**, say so explicitly — this directly explains the instruction-following regression (see #3) and turns a weakness into an analyzed finding.

---

## 3. Investigate and write up the instruction-following regression (2–4 hrs)

Instruction-following dropped 80% → 70% after DPO while GSM8K jumped. This is your most interesting result — don't bury it.

- [ ] Pull 10–15 failing instruction-following examples from the DPO model and read them. Common causes to check for:
  - Model now biases toward "show reasoning steps" even on non-math prompts (overfit to CoT style from math-heavy pairs)
  - Output length/format drift (DPO pushing toward longer or more rigid completions)
  - Reward hacking on whatever heuristic scored "chosen" answers
- [ ] Add a **"Limitations & Findings"** section to the README naming this trade-off directly: *"DPO improved mathematical reasoning substantially but degraded general instruction-following, consistent with narrow-domain preference data causing capability trade-offs rather than general improvement."*
- [ ] If time allows: mix in a small % of non-math instruction-following preference pairs and re-run DPO to see if the trade-off narrows. This single experiment would meaningfully strengthen the project.

---

## 4. Fill in missing technical transparency (1 hr)

- [ ] Surface key hyperparameters in the README (not just in `config.py`): DPO `beta`, learning rate, LoRA rank/alpha/target modules, batch size, epochs, number of preference pairs.
- [ ] Add training curves — DPO loss, reward margin (chosen vs rejected reward gap) over steps, and reward accuracy. These are usually the first thing people ask for to sanity-check that DPO training was stable and didn't collapse.
- [ ] Explain the perplexity numbers briefly — SFT/DPO perplexity being close (47.05 vs 47.04) and higher than base (42.53) is expected (DPO isn't optimizing likelihood against a reference corpus), but say so or it looks like an unexplained regression.

---

## 5. Optional but high-value additions (if you have more GPU time)

- [ ] **Add GRPO as a comparison arm.** Since you already have a Kaggle T4 setup, running GRPO with a rule-based reward (e.g. answer-correctness on GSM8K, no reward model needed) alongside DPO on the same base would let you directly compare online vs offline preference optimization on the same task — a strong differentiator vs typical DPO-only projects.
- [ ] **KL-divergence from reference model**, reported alongside the DPO reward margin, to show you're monitoring policy drift (not just accuracy).
- [ ] **Held-out preference-pair win-rate** (not just GSM8K) evaluated by an LLM judge, since you already built this in `06_compare_and_report.ipynb` — surface those numbers in the README table too, not just accuracy/perplexity.

---

## Suggested README restructure

```
1. Overview (keep as-is)
2. Results table (fix % claim, add sample size + seeds)
3. NEW: Preference Data (what pairs, how built, how many)
4. NEW: Key Finding — the math/instruction-following trade-off
5. Hugging Face Models (keep)
6. Repository Structure (keep)
7. Key Technical Challenges Solved (keep — this section is genuinely good)
8. NEW: Limitations & Future Work
9. How to Run (keep)
```

---

## Rough time budget

| Task | Time | GPU needed? |
|---|---|---|
| Fix numbers/claims | 30–60 min | No |
| Preference dataset writeup | 1–2 hrs | No |
| Regression analysis + Limitations section | 2–4 hrs | No (uses existing outputs) |
| Hyperparams + training curves | 1 hr | No (if logs saved) — else re-run |
| Non-math pairs re-run experiment | 3–6 hrs | Yes |
| GRPO comparison arm | 6–10 hrs | Yes |

The first four rows need zero additional compute — just analysis and writing — and will do more for the project's credibility than any additional training run.
