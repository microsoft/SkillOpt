"""SkillOpt-Sleep — the per-night evidentiary chain (``evidence.jsonl``).

The existing report/diagnostics answer *what* the cycle decided; they do not
answer *why*. This module records the full causal chain, per night:

    transcript session  ->  miner exchange (prompt + raw reply)
                        ->  mined task (+ its checks, + source session ids)
                        ->  split assignment (train / val)
                        ->  every replay attempt (phase-tagged, full prompt
                            and response, cache hits marked)
                        ->  per-task scores with the failing checks named
                        ->  the reflect exchange (prompt, raw reply, parsed
                            edits) and every gate trial
                        ->  the final gate decision with the score arithmetic
                        ->  what was staged

Design constraints (matching the sleep engine's contract):
  * pure stdlib, thread-safe (replay batches run in a thread pool);
  * every persisted string passes through ``redact_secrets`` and a length
    cap, so the log can never leak more than diagnostics already could;
  * append-only JSONL so a crashed night still leaves its partial chain;
  * zero behavior change when disabled (``evidence_log: false``).

Events share the shape::

    {"ts": <iso8601>, "seq": <int>, "stage": <str>, "event": <str>, ...}
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional

from skillopt_sleep.staging import redact_secrets


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


class EvidenceLog:
    """Append-only, thread-safe JSONL logger for one sleep night."""

    def __init__(self, path: str, *, max_chars: int = 4000, redact: bool = True) -> None:
        self.path = path
        self.max_chars = max(200, int(max_chars))
        self.redact = redact
        self._lock = threading.Lock()
        self._seq = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # ── sanitization ──────────────────────────────────────────────────────
    def _clean(self, value: Any) -> Any:
        if isinstance(value, str):
            if len(value) > self.max_chars:
                dropped = len(value) - self.max_chars
                value = value[: self.max_chars] + f"…[truncated {dropped} chars]"
            return redact_secrets(value) if self.redact else value
        if isinstance(value, dict):
            return {k: self._clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._clean(v) for v in value]
        return value

    # ── the one write path ────────────────────────────────────────────────
    def log(self, stage: str, event: str, **data: Any) -> None:
        record = {"ts": _now_iso(), "stage": stage, "event": event}
        record.update(self._clean(data))
        with self._lock:
            self._seq += 1
            record["seq"] = self._seq
            try:
                line = json.dumps(record, ensure_ascii=False, default=str)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                # Evidence must never break a night; drop the record instead.
                pass


def attach(backend, ev: Optional[EvidenceLog]) -> None:
    """Attach ``ev`` to a backend — and, for DualBackend, to both halves —
    so every layer that wants to log can find it via ``backend.evidence``."""
    if backend is None:
        return
    backend.evidence = ev
    for half in ("target", "optimizer"):
        sub = getattr(backend, half, None)
        if sub is not None:
            sub.evidence = ev


def get(backend) -> Optional[EvidenceLog]:
    return getattr(backend, "evidence", None)


def set_phase(backend, phase: str) -> None:
    """Tag subsequent replay calls with a phase label (baseline_val,
    train, gate_trial:skill, final_val, ...). Phases are sequential in the
    consolidation loop, so a plain attribute is safe; parallelism only ever
    happens *within* one phase."""
    if backend is None:
        return
    backend.evidence_phase = phase
    for half in ("target", "optimizer"):
        sub = getattr(backend, half, None)
        if sub is not None:
            sub.evidence_phase = phase


def phase(backend) -> str:
    return getattr(backend, "evidence_phase", "") or ""


def read_events(path: str) -> list:
    """Best-effort reader for the dashboard: skips corrupt lines."""
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except OSError:
        return []
    out.sort(key=lambda r: r.get("seq", 0))
    return out
