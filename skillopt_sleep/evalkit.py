"""Paired A/B evaluation kit for SkillOpt-Sleep.

Sleep reports (and many PRs) quote single-run success rates with no
uncertainty and no guarantee that the two conditions saw the same tasks.
This module is the shared instrument for those comparisons:

  * one fixed task manifest, paired by task id
  * McNemar's test on per-task binary outcomes
  * percentile-bootstrap confidence intervals on the success-rate delta
  * optional multi-seed repeats (per-seed deltas + a pooled pair test)

It does not change the nightly gate. It standardizes the evidence that
reports and PRs cite. Pure stdlib; no numpy / scipy.

Refuse comparisons whose task-id sets differ. Graded (non-binary) scores
are bootstrap-only: McNemar is not defined for them.

CLI::

    python -m skillopt_sleep.evalkit --manifest M.json --a A.json --b B.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# ── errors ────────────────────────────────────────────────────────────────────

class EvalkitError(ValueError):
    """User-facing contract failure (mismatched ids, empty, etc.)."""


# ── results ───────────────────────────────────────────────────────────────────

@dataclass
class McNemarResult:
    both_success: int
    a_only: int          # A success, B fail  (c in the usual 2x2)
    b_only: int          # A fail, B success  (b)
    both_fail: int
    n: int
    chi2: float          # uncorrected (b-c)^2 / (b+c); nan if no discordants
    p_chi2: float
    p_exact: float       # two-sided exact binomial on discordants
    significant: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BootstrapCI:
    n_boot: int
    seed: int
    alpha: float
    low: float
    high: float
    mean: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalReport:
    n_tasks: int
    rate_a: float
    rate_b: float
    delta: float
    mcnemar: Optional[McNemarResult]
    bootstrap: BootstrapCI
    per_seed: List[Dict[str, float]] = field(default_factory=list)
    seed_mean_delta: Optional[float] = None
    seed_sd_delta: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    refused: bool = False
    refuse_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.mcnemar is not None:
            d["mcnemar"] = self.mcnemar.to_dict()
        d["bootstrap"] = self.bootstrap.to_dict()
        return d


# ── statistics ────────────────────────────────────────────────────────────────

def _chi2_sf_df1(x: float) -> float:
    """Survival function of chi-square with 1 df: P(X > x) = erfc(sqrt(x/2))."""
    if x < 0.0 or math.isnan(x):
        return float("nan")
    if x == 0.0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def _binom_pmf(k: int, n: int, p: float = 0.5) -> float:
    if k < 0 or k > n:
        return 0.0
    # nCk * p^k * (1-p)^(n-k). For p=0.5 this is nCk / 2^n.
    if p == 0.5:
        return math.comb(n, k) / float(1 << n) if n < 1024 else math.comb(n, k) * (0.5 ** n)
    return math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial test of discordants, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(_binom_pmf(i, n, 0.5) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail)


def mcnemar_from_counts(
    both_success: int,
    a_only: int,
    b_only: int,
    both_fail: int,
    *,
    alpha: float = 0.05,
) -> McNemarResult:
    n = both_success + a_only + b_only + both_fail
    disc = a_only + b_only
    if disc == 0:
        chi2 = 0.0
        p_chi2 = 1.0
    else:
        chi2 = (b_only - a_only) ** 2 / float(disc)
        p_chi2 = _chi2_sf_df1(chi2)
    p_exact = exact_mcnemar_p(b_only, a_only)
    return McNemarResult(
        both_success=both_success,
        a_only=a_only,
        b_only=b_only,
        both_fail=both_fail,
        n=n,
        chi2=chi2,
        p_chi2=p_chi2,
        p_exact=p_exact,
        significant=p_exact < alpha,
    )


def mcnemar_paired(a: Sequence[int], b: Sequence[int], *, alpha: float = 0.05) -> McNemarResult:
    if len(a) != len(b):
        raise EvalkitError("McNemar requires equal-length paired outcomes")
    bs = ao = bo = bf = 0
    for x, y in zip(a, b):
        if x and y:
            bs += 1
        elif x and not y:
            ao += 1
        elif (not x) and y:
            bo += 1
        else:
            bf += 1
    return mcnemar_from_counts(bs, ao, bo, bf, alpha=alpha)


def bootstrap_delta_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> BootstrapCI:
    if len(a) != len(b) or not a:
        raise EvalkitError("bootstrap requires a non-empty paired sample")
    if n_boot < 1:
        raise EvalkitError("n_boot must be >= 1")
    rng = random.Random(seed)
    n = len(a)
    deltas: List[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(a[i] for i in idx) / n
        db = sum(b[i] for i in idx) / n
        deltas.append(db - da)
    deltas.sort()
    # Inclusive percentile on the sorted sample.
    lo_i = int(math.floor((alpha / 2.0) * (n_boot - 1)))
    hi_i = int(math.ceil((1.0 - alpha / 2.0) * (n_boot - 1)))
    lo_i = max(0, min(n_boot - 1, lo_i))
    hi_i = max(0, min(n_boot - 1, hi_i))
    return BootstrapCI(
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
        low=deltas[lo_i],
        high=deltas[hi_i],
        mean=sum(deltas) / n_boot,
    )


# ── pairing / loading ─────────────────────────────────────────────────────────

def _as_binary(value: Any) -> Optional[int]:
    if value is True or value == 1 or value == 1.0:
        return 1
    if value is False or value == 0 or value == 0.0:
        return 0
    return None


def _normalize_outcomes(raw: Mapping[str, Any]) -> Dict[str, List[float]]:
    """Map task id -> list of per-seed scores (length 1 if unseeded)."""
    out: Dict[str, List[float]] = {}
    for tid, val in raw.items():
        key = str(tid)
        if isinstance(val, Mapping) and "seeds" in val:
            val = val["seeds"]
        if isinstance(val, (list, tuple)):
            out[key] = [float(x) for x in val]
        else:
            out[key] = [float(val)]
    return out


def align_pairs(
    manifest_ids: Sequence[str],
    outcomes_a: Mapping[str, Any],
    outcomes_b: Mapping[str, Any],
) -> Tuple[List[str], List[List[float]], List[List[float]]]:
    """Align A and B onto the manifest. Refuse any id-set mismatch."""
    ids = [str(i) for i in manifest_ids]
    if not ids:
        raise EvalkitError("manifest is empty")
    if len(ids) != len(set(ids)):
        raise EvalkitError("manifest has duplicate task ids")
    a = _normalize_outcomes(outcomes_a)
    b = _normalize_outcomes(outcomes_b)
    a_ids, b_ids = set(a), set(b)
    want = set(ids)
    if a_ids != want or b_ids != want:
        missing_a = sorted(want - a_ids)
        missing_b = sorted(want - b_ids)
        extra_a = sorted(a_ids - want)
        extra_b = sorted(b_ids - want)
        raise EvalkitError(
            "outcome task ids must equal the manifest "
            f"(missing_a={missing_a[:8]}, missing_b={missing_b[:8]}, "
            f"extra_a={extra_a[:8]}, extra_b={extra_b[:8]})"
        )
    n_seed_a = {len(a[i]) for i in ids}
    n_seed_b = {len(b[i]) for i in ids}
    if len(n_seed_a) != 1 or n_seed_a != n_seed_b:
        raise EvalkitError("every task must have the same number of seed repeats in A and B")
    return ids, [a[i] for i in ids], [b[i] for i in ids]


def _is_binary_matrix(rows: Sequence[Sequence[float]]) -> bool:
    for row in rows:
        for x in row:
            if _as_binary(x) is None:
                return False
    return True


def _mean(xs: Iterable[float]) -> float:
    seq = list(xs)
    return sum(seq) / len(seq) if seq else float("nan")


def _sd(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def reconstruct_paired_from_rates(n: int, rate_a: float, rate_b: float) -> Tuple[List[int], List[int]]:
    """Deterministic maximum-concordance reconstruction of paired binaries.

    First ``round(n * rate)`` tasks succeed in each condition, same id order.
    This is a published-rate replay convention, not original microdata.
    """
    if n < 1:
        raise EvalkitError("n must be >= 1")
    ka = int(round(n * rate_a))
    kb = int(round(n * rate_b))
    a = [1 if i < ka else 0 for i in range(n)]
    b = [1 if i < kb else 0 for i in range(n)]
    return a, b


def compare(
    manifest_ids: Sequence[str],
    outcomes_a: Mapping[str, Any],
    outcomes_b: Mapping[str, Any],
    *,
    alpha: float = 0.05,
    n_boot: int = 10000,
    seed: int = 42,
    allow_graded: bool = False,
) -> EvalReport:
    ids, a_rows, b_rows = align_pairs(manifest_ids, outcomes_a, outcomes_b)
    n_seed = len(a_rows[0])
    notes: List[str] = []

    # Per-task mean across seeds (the headline paired sample).
    a_mean = [_mean(row) for row in a_rows]
    b_mean = [_mean(row) for row in b_rows]
    rate_a = _mean(a_mean)
    rate_b = _mean(b_mean)
    delta = rate_b - rate_a
    boot = bootstrap_delta_ci(a_mean, b_mean, n_boot=n_boot, seed=seed, alpha=alpha)

    binary = _is_binary_matrix(a_rows) and _is_binary_matrix(b_rows)
    mcnemar: Optional[McNemarResult] = None
    if binary:
        # Pool (task, seed) as paired observations when seeds align.
        flat_a = [int(_as_binary(x) or 0) for row in a_rows for x in row]
        flat_b = [int(_as_binary(x) or 0) for row in b_rows for x in row]
        mcnemar = mcnemar_paired(flat_a, flat_b, alpha=alpha)
    elif allow_graded:
        notes.append("graded scores: McNemar omitted; bootstrap CI only")
    else:
        raise EvalkitError(
            "non-binary scores require --allow-graded (McNemar is undefined)"
        )

    per_seed: List[Dict[str, float]] = []
    seed_mean = seed_sd = None
    if n_seed > 1:
        for s in range(n_seed):
            da = _mean(row[s] for row in a_rows)
            db = _mean(row[s] for row in b_rows)
            per_seed.append({"seed": float(s), "rate_a": da, "rate_b": db, "delta": db - da})
        deltas = [row["delta"] for row in per_seed]
        seed_mean = _mean(deltas)
        seed_sd = _sd(deltas)
        notes.append(
            f"multi-seed: {n_seed} repeats; seed-mean delta={seed_mean:.6f} "
            f"sd={seed_sd:.6f}"
        )

    return EvalReport(
        n_tasks=len(ids),
        rate_a=rate_a,
        rate_b=rate_b,
        delta=delta,
        mcnemar=mcnemar,
        bootstrap=boot,
        per_seed=per_seed,
        seed_mean_delta=seed_mean,
        seed_sd_delta=seed_sd,
        notes=notes,
    )


def compare_aa(
    manifest_ids: Sequence[str],
    outcomes: Mapping[str, Any],
    **kwargs: Any,
) -> EvalReport:
    """A/A calibration: identical conditions must not reject at alpha."""
    report = compare(manifest_ids, outcomes, outcomes, **kwargs)
    report.notes.append("A/A calibration (identical conditions)")
    return report


# ── I/O ───────────────────────────────────────────────────────────────────────

def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _manifest_ids(obj: Any) -> List[str]:
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, Mapping):
        if "ids" in obj:
            return [str(x) for x in obj["ids"]]
        if "tasks" in obj:
            return [str(t["id"] if isinstance(t, Mapping) else t) for t in obj["tasks"]]
        if "outcomes" in obj:
            return [str(k) for k in obj["outcomes"]]
    raise EvalkitError("manifest must be a list of ids or an object with ids/tasks")


def _outcomes(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, Mapping) and "outcomes" in obj:
        return dict(obj["outcomes"])
    if isinstance(obj, Mapping):
        return dict(obj)
    raise EvalkitError("outcomes file must be an object mapping task id to score")


def format_markdown(report: EvalReport) -> str:
    lines = [
        "# Paired A/B evalkit report",
        "",
        f"- n_tasks: {report.n_tasks}",
        f"- rate_a: {report.rate_a:.6f}",
        f"- rate_b: {report.rate_b:.6f}",
        f"- delta (B-A): {report.delta:+.6f}",
        (
            f"- bootstrap {int((1 - report.bootstrap.alpha) * 100)}% CI: "
            f"[{report.bootstrap.low:+.6f}, {report.bootstrap.high:+.6f}] "
            f"(n_boot={report.bootstrap.n_boot}, seed={report.bootstrap.seed})"
        ),
    ]
    if report.mcnemar is not None:
        m = report.mcnemar
        lines.append(
            f"- McNemar 2x2: both+={m.both_success} a_only={m.a_only} "
            f"b_only={m.b_only} both-={m.both_fail}"
        )
        lines.append(
            f"- McNemar chi2={m.chi2:.4f} p_chi2={m.p_chi2:.6g} "
            f"p_exact={m.p_exact:.6g} significant={m.significant}"
        )
    if report.seed_mean_delta is not None:
        lines.append(
            f"- multi-seed mean delta: {report.seed_mean_delta:+.6f} "
            f"(sd {report.seed_sd_delta:.6f}, k={len(report.per_seed)})"
        )
    for note in report.notes:
        lines.append(f"- note: {note}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="skillopt_sleep.evalkit",
        description="Paired A/B comparison with McNemar and bootstrap CIs",
    )
    p.add_argument("--manifest", required=True, help="JSON list of task ids (or {ids,tasks})")
    p.add_argument("--a", required=True, help="JSON outcomes for condition A")
    p.add_argument("--b", default="", help="JSON outcomes for condition B (omit for A/A)")
    p.add_argument("--aa", action="store_true", help="A/A calibration (ignore --b, reuse --a)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-graded", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    try:
        ids = _manifest_ids(_load_json(args.manifest))
        a = _outcomes(_load_json(args.a))
        if args.aa or not args.b:
            report = compare_aa(
                ids, a, alpha=args.alpha, n_boot=args.boot,
                seed=args.seed, allow_graded=args.allow_graded,
            )
        else:
            b = _outcomes(_load_json(args.b))
            report = compare(
                ids, a, b, alpha=args.alpha, n_boot=args.boot,
                seed=args.seed, allow_graded=args.allow_graded,
            )
    except EvalkitError as exc:
        print(f"ERR_EVALKIT {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"ERR_EVALKIT {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_markdown(report), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
