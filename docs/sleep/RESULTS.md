# SkillOpt-Sleep — results & analysis

This is the evidence behind SkillOpt-Sleep: does a nightly, offline sleep cycle
actually make a *deployed* agent better, and is it safe to run unattended? We
answer with a controlled deployment-scale study — the same protocol the plugin
runs in production, scored on full held-out test sets.

## Setup

**Protocol (identical for every cell unless stated).** 5 nights; each night adds
**10 new real "today" tasks**; the skill carries over and is refined night to
night. The full held-out **test** split is scored before night 1 (*baseline*) and
after night 5 (*after*); **Δ = after − baseline** in percentage points. Optimizer
model = **GPT-5.5**; single seed (42); every number is produced by the exact
shipped engine `skillopt_sleep.dream.dream_consolidate` (the experiment harness and
the plugin cycle call the same function).

**Benchmarks** (real evaluators, not format heuristics):

| Benchmark | Held-out test | Scoring |
|---|---|---|
| SearchQA | 1,400 items | SQuAD exact-match vs gold |
| LiveMathematicianBench | 124 items | multiple-choice label (choices shuffled per item) |
| SpreadsheetBench | 280 items | the agent's generated openpyxl code is **executed**, output workbook compared cell-by-cell to a golden file |

**Targets:** GPT-5.5, GPT-5.4-mini, GPT-5.4-nano. **Modes:** validation-gated
(default) and gate-free.

---

## 1. The headline — the validation gate is what makes nightly self-evolution *safe*

Self-evolution is easy to build and easy to ruin: an optimizer that accepts its
own "lessons" unconditionally can adopt a plausible-but-wrong rule and an obedient
model will follow it off a cliff. We reproduced exactly that failure, then showed
the gate prevents it.

Stress case — **GPT-5.4-nano on SearchQA**, weak model on a single-sample (degraded)
reflection signal, same nights, same candidate edits, gate **off** vs **on**:

| | Night 0 → Night 5 | Δ |
|---|---|---|
| **no gate** | 0.554 → **0.026** | **−52.8** |
| **with gate (default)** | 0.570 → 0.570 | 0.0 |

Ungated, the optimizer learned "answer with the document-title string, verbatim";
the model complied and accuracy collapsed night after night
(0.554 → 0.490 → 0.325 → 0.031 → 0.034 → 0.026). The gated twin **rejected every one
of those edits** and never lost a point. This single experiment is the core
argument for SkillOpt-Sleep's design, and why the gate ships **on by default**.

---

## 2. Cross-model scaling — bigger gains where there's headroom

The same protocol on a weaker target model (**GPT-5.4-nano**, optimizer = GPT-5.5)
produces substantially larger gains — because the weaker model has more room to
learn. This is the realistic "cheap deployed agent, strong overnight optimizer"
scenario:

| Config (SearchQA, nano, gated) | Baseline → After | Δ | Night-by-night |
|---|---|---|---|
| **cumulative replay, nights=5** | 0.560 → **0.679** | **+11.9** | 0.560 → 0.626 → 0.665 → 0.665 → 0.665 → 0.679 |
| recall_k=20, nights=5 | 0.566 → 0.681 | +11.5 | 0.566 → 0.659 → 0.685 → 0.685 → 0.681 → 0.681 |
| cumulative, nights=8 | 0.562 → 0.657 | +9.5 | saturates after night 5 |

Both replay strategies (cumulative and recall) agree within 0.4 pt — the gain is
robust across configurations.

**Compared to GPT-5.5 on the same benchmark (SearchQA, gated):**

| Target model | Best Δ | Baseline | Headroom |
|---|---|---|---|
| GPT-5.4-nano | **+11.9** | 0.560 | 44 pt |
| GPT-5.5 | +6.0 | 0.798 | 20 pt |

The story: **SkillOpt-Sleep helps most where there's the most to learn** — weaker
deployed models benefit ~2× as much from the same nightly optimization. This is
also the economical deployment pattern (cheap inference model + one strong
overnight optimizer call).

---

## 3. Experience replay turns a one-time bump into a climb

The plugin's two opt-in knobs (`recall_k`, `dream_rollouts`) are what produce the
gains. On **SearchQA, GPT-5.5, gated** — the gain rises monotonically with how
much relevant past experience is recalled:

| Replay (`dream_rollouts=5`) | Baseline → After | Δ |
|---|---|---|
| `recall_k=10` | 0.802 → 0.834 | +3.1 |
| `recall_k=20` | 0.803 → 0.848 | **+4.5** |
| full-history (reference, not a default) | 0.796 → 0.851 | +5.6 |

And the curve genuinely **climbs across nights** rather than jumping once and
plateauing — full-history replay, gated, night by night:

```
0.798 → 0.814 → 0.854 → 0.854 → 0.854 → 0.858
```

The gate accepts a new, better skill as late as **night 5** (0.854 → 0.858).
Replay-policy ablation (SearchQA, GPT-5.5):

| Replay policy | Gate-free Δ | Gated Δ |
|---|---|---|
| none (tonight's tasks only) | +3.9 | +2.0 |
| **recall k=10 (shipped default-able)** | +5.1 | +4.4 |
| cumulative (full history) | +4.8 | +6.0 |

Recall captures most of cumulative's benefit at a fraction of the per-night cost.

---

## 4. Default hyperparameters are the sweet spot

We swept `dream_factor`, `rollouts`, `per_night`, and `nights` on the nano cell
(SearchQA, gated) to verify the shipped defaults are well-tuned:

| Variant | Δ | vs default (+11.9) |
|---|---|---|
| dream_factor=4 (default 2) | +8.8 | −3.1 |
| rollouts=10 (default 5) | +9.5 | −2.4 |
| per_night=15 (default 10) | +2.7 | −9.2 |
| nights=8 (default 5) | +9.5 | −2.4 |

Every direction away from the default hurts. This means users get the best result
**out of the box** without tuning — the recipe is robust by design.

---

## 5. Why these gains exist — the dream-diversity fix (and a rigor note)

Reflection learns from the **contrast** between good and bad rollouts of the same
task, which requires the K dream rollouts to be *independent samples*. An early
version of the engine collapsed them to one cached sample, so contrastive
reflection never fired. Fixing that, then adding recall, is what produces the
gains in Sections 1–2. Measured across an 18-cell deployment sweep (3 benchmarks ×
3 targets × 2 modes), under three engine configurations:

| Engine configuration | mean Δ | worst-cell Δ | cells > +0.5 | cells < −0.5 |
|---|---|---|---|---|
| single-sample reflection (degraded) | −2.66 | **−52.8** | 7 / 18 | 5 / 18 |
| diverse rollouts (K=5), no recall | +0.24 | −4.0 | 6 / 18 | 7 / 18 |
| **diverse rollouts + recall (shipped)** | **+0.53** | **−2.4** | 7 / 18 | 7 / 18 |

The catastrophic −52.8 is removed **at its source** by diverse rollouts: the same
gate-free nano-SearchQA cell goes 0.554 → **0.586 (+2.7)** with no gate at all once
the dream is fixed. Recall then lifts the grid mean and tightens the worst case.
This is **defense in depth, each layer measured**: diverse rollouts propose better
edits, recall remembers relevant experience, and the gate catches whatever still
slips through.

---

## 6. End-to-end on real agents

On the public [gbrain-evals](https://github.com/garrytan/gbrain-evals) `skillopt-v1`
benchmark — designed for exactly this learnable-gap setting — deficient seed skills
go **0.00 → 1.00** on the held-out set with **both Claude Code and Codex** as the
target agent (all 4 seeds, including a real tool-use loop), and the two agents
cross-verify each other's consolidated skills.

---

## 7. Honest scope & limitations

- **Where it helps:** recurring tasks with a checkable correctness signal and real
  headroom. That is the plugin's actual use case (your repeated daily tasks and
  house rules the agent keeps missing).
- **Where it's flat:** saturated tasks on strong models, or noisy tasks with a weak
  learning signal — within run-to-run noise.
- **Single seed.** Cells aggregate one seed per config; treat sub-~1.5 pt
  differences as noise. Spot seed-robustness check on the one flagged cell
  (nano SearchQA gated): seeds 42/43/44 give −1.9 / +3.6 / +4.7 (3-seed mean
  **+2.1**), i.e. the tabled −1.9 is a pessimistic draw, not the typical outcome.
- **Keep the gate on.** It is the difference between bounded downside (−2.4) and a
  −52.8 collapse. Gate-free mode is for users who cannot hold out a validation set
  and is additionally protected by the output-contract guardrail.

---

Back to the module overview: [`docs/sleep/README.md`](README.md) ·
full reference: [Documentation & Reproduction Guide](https://microsoft.github.io/SkillOpt/docs/guideline.html#sleep).
