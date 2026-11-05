# The Training Loop

SkillOpt's core insight: **optimizing natural-language skill documents follows the same structure as training neural networks**.

## Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Training Loop                         │
│                                                         │
│  for epoch in epochs:                                   │
│    for step in steps:                                   │
│      1. Rollout   — Target executes tasks              │
│      2. Reflect   — Optimizer analyzes trajectories       │
│      3. Aggregate — Hierarchical merge of patches       │
│      4. Select    — Rank & clip edits (learning rate)   │
│      5. Update    — Apply patches to skill doc          │
│      6. Gate      — Validate & accept/reject            │
│                                                         │
│    Epoch Boundary:                                       │
│      • Slow Update (longitudinal comparison & guidance) │
│      • Meta Skill  (cross-epoch strategy memory)        │
└─────────────────────────────────────────────────────────┘
```

## Stage Details

### 1. Rollout (Forward Pass)

The **target** model executes tasks using the current skill document as its prompt. Each task produces a trajectory and a score.

```python
# Analogy: forward pass through the network
predictions = model(input, skill_document)
scores = evaluate(predictions, ground_truth)
```

### 2. Reflect (Backward Pass)

The **optimizer** model analyzes trajectory minibatches and produces **edit
patches** — structured suggestions for improving the skill document. Failure
minibatches are always eligible for analysis; successful trajectories are also
analyzed unless `gradient.failure_only` is enabled. Independent minibatches can
run concurrently according to `gradient.analyst_workers`.

```python
# Analogy: computing gradients
gradients = loss.backward()  # → edit patches
```

### 3. Aggregate

Semantically similar edit patches are merged to avoid redundant edits.

### 4. Select (Gradient Clipping)

Edits are ranked by relevance score. The `learning_rate` parameter caps how many edits are applied per step — just like gradient clipping prevents overshooting.

```python
# Analogy: gradient clipping + optimizer step size
selected = top_k(edits, k=learning_rate)
```

The `lr_scheduler` adjusts this over training:

- **cosine**: Start aggressive, taper smoothly
- **linear**: Linear decay
- **constant**: Fixed rate

### 5. Update (Parameter Update)

Selected edits are applied to the skill document, producing a new version.

### 6. Gate (Validation)

The updated skill is evaluated on a **selection split** (analogous to a
validation set). With the gate enabled, the candidate is accepted only when its
configured gate score (`hard`, `soft`, or `mixed`) is strictly higher than the
current skill's score. With `evaluation.use_gate: false`, validation is still
recorded but candidates are force-accepted.

#### Why current and best are separate

On the strictly gated path, the current and best skills are initialized to the
same document and score. Every accepted candidate must beat the current score,
which is also the best score, so both states advance together. Thus
`current_skill == best_skill` remains an invariant for the procedure described
by the paper; the nested "new best" check is redundant on that path.

The implementation keeps separate current and best snapshots for optional
extensions where they can diverge. Disabling the gate can force-accept a new
current skill without replacing the validation-best snapshot, and an ungated
slow update can likewise modify only the current skill. The final evaluation
reports the validation-best snapshot separately from the last current skill.

## Epoch Boundary Mechanisms

### Slow Update

At the end of each epoch (starting from epoch 2), the system performs a
**longitudinal comparison**: it rolls out both the previous epoch's skill and
the current skill on the same samples, categorizes items as
improved/regressed/persistent-fail/stable-success, then generates high-level
**guidance** for the skill document. Depending on
`optimizer.slow_update_gate_with_selection`, that guidance is either checked on
the selection split or applied unconditionally. Its purpose is to counter
cross-epoch forgetting.

### Meta Skill

A **meta-skill memory** accumulates high-level strategy notes across the training
run. Starting at the end of epoch 2, the optimizer compares the previous and
current epoch, writes a compact memory, and provides the prior epoch's memory as
additional context during later reflection and update stages.

## Next Steps

- [Understand Skill Documents](skill-document.md)
- [DL ↔ SkillOpt analogy table](dl-analogy.md)
