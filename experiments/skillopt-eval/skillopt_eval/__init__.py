"""Controlled baseline-vs-candidate experiment harness for SkillOpt.

Implements the reproducible experiment requested in
https://github.com/microsoft/SkillOpt/issues/132 :

* identical harness/model/settings for baseline and candidate runs
* deterministic train / validation / test scenario splits
* raw trajectories plus quality score, tokens, tool calls, turns,
  latency and failure counts per run
* accepted / rejected optimization-attempt bookkeeping behind a
  held-out validation gate that fails closed
* paired statistics via skillopt_sleep.evalkit (McNemar + bootstrap CI)
* artifacts: raw_results.csv, results.json, RESULTS.md

The runner is pluggable: ``mock`` runs the full pipeline deterministically
offline; ``superpowers`` delegates to
``skillopt_sleep.adapters.superpowers`` for the real Claude Code +
Superpowers bootstrap path (POSIX + authenticated ``claude`` CLI).
"""
