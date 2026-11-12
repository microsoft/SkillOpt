"""SkillOpt-Sleep: measure the nightly gate on a public benchmark.

Two experiments, both through the shipped ``dream_consolidate``:

``faults``  A fixed rule whose held-out effect is known is proposed in place of
            the optimizer's reflection, in many nightly contexts, and the gate
            decides. Reports how often each rule is accepted.
``nights``  The multi-night deployment protocol of docs/sleep/RESULTS.md with
            the held-out test items scored after every night. Reports the
            learning curve.

Run an arm with a feature off and again with it on (``--feature`` is passed to
``dream_consolidate``). Arms share every model draw they have in common
through the response store, so they differ only by the feature, and a run can
be resumed.

Protocol: each seed shuffles the benchmark's training pool; night k gets items
10(k-1)..10k-1, split by ``assign_splits(val_fraction=0.34, seed)``; the skill
starts as the benchmark's shipped initial skill; gate metric ``mixed``.
``faults`` uses one rollout, no dream variants, ``recall_k=10`` and an archive
made of the seed's earlier nights. ``nights`` uses ``dream_rollouts=5``,
``dream_factor=2`` and ``recall_k=10``. Held-out scores use the test split.

Benchmarks: ``searchqa`` (400 training items, exact match) and ``livemath``
(LiveMathematicianBench, 52 training items from the released train and val
manifests, multiple-choice label). ``fixture`` is a tiny offline set.

Offline check, no key and no network (this is what the test suite runs):
    python -m skillopt_sleep.experiments.gate_study faults --offline --arm off
    python -m skillopt_sleep.experiments.gate_study faults --offline --arm check \
        --feature '{"gate_train_check": true}'
    python -m skillopt_sleep.experiments.gate_study report --offline

Live runs spend money on your own key, so they need ``--live`` and print the
number of model calls they expect before the first one:
    python -m pip install -e ".[searchqa]" openai
    python scripts/materialize_searchqa.py
    export OPENAI_API_KEY=...        # any OpenAI-compatible endpoint
    export OPENAI_BASE_URL=...       # optional
    python -m skillopt_sleep.experiments.gate_study effects --live
    python -m skillopt_sleep.experiments.gate_study faults --live --arm off
    python -m skillopt_sleep.experiments.gate_study faults --live --arm check \
        --feature '{"gate_train_check": true}'
    python -m skillopt_sleep.experiments.gate_study nights --live --arm base
    python -m skillopt_sleep.experiments.gate_study report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import sys
import threading
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "gate_study" / "items.json"
LIVEMATH_REPO = "LiveMathematicianBench/LiveMathematicianBench"

RULES: Dict[str, List[Tuple[str, str]]] = {
    "searchqa": [
        ("H1 title verbatim", "Answer with the title of the first document in the context, verbatim."),
        ("H2 full sentence", "Always answer in one complete sentence that restates the question."),
        ("H3 list alternatives", "When several names could apply, list every plausible alternative separated by commas."),
        ("H4 category suffix", "Always add the category of the answer in parentheses after it, for example: Paris (city)."),
        ("H5 formal name", "Prefer the most formal, complete official name of the answer, for example United States of America rather than USA."),
        ("H6 placebo", "Be careful and precise."),
    ],
    "livemath": [
        ("M1 last option", "Always choose the last option listed."),
        ("M2 weakest statement", "Always choose the weakest, most conservative statement among the options."),
        ("M3 no reasoning", "Do not reason step by step. Output the answer tag immediately."),
        ("M4 longest option", "When two options look similar, choose the longer one."),
        ("M5 restate hypotheses", "Before answering, restate the hypotheses of the question in one line."),
        ("M6 placebo", "Be careful and precise."),
    ],
    "fixture": [
        ("F1 harmful", "FIXTURE-RULE-HARMFUL"),
        ("F2 helpful", "FIXTURE-RULE-HELPFUL"),
        ("F3 placebo", "FIXTURE-RULE-PLACEBO"),
    ],
}


class Store:
    """SQLite response store and result tables. First writer wins, so a
    parallel run stays replayable and a second arm reuses the first arm's draws."""

    def __init__(self, path: str) -> None:
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=120)
        self.db.execute("PRAGMA journal_mode=WAL")
        for ddl in (
            "calls (k TEXT PRIMARY KEY, model TEXT, response TEXT, tin INTEGER, tout INTEGER, ts REAL)",
            "decisions (arm TEXT, model TEXT, seed INTEGER, night INTEGER, rule TEXT, accepted INTEGER, "
            "calls INTEGER, PRIMARY KEY (arm, model, seed, night, rule))",
            "effects (model TEXT, rule TEXT, em REAL, PRIMARY KEY (model, rule))",
            "curves (arm TEXT, model TEXT, seed INTEGER, night INTEGER, accepted INTEGER, em REAL, "
            "calls INTEGER, PRIMARY KEY (arm, model, seed, night))",
        ):
            self.db.execute(f"CREATE TABLE IF NOT EXISTS {ddl}")
        self.db.commit()

    def get(self, key: str):
        with self.lock:
            return self.db.execute("SELECT response FROM calls WHERE k=?", (key,)).fetchone()

    def put(self, key: str, model: str, text: str, tin: int, tout: int) -> str:
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO calls VALUES (?,?,?,?,?,?)",
                            (key, model, text, tin, tout, time.time()))
            self.db.commit()
            return self.db.execute("SELECT response FROM calls WHERE k=?", (key,)).fetchone()[0]

    def run(self, sql: str, args: Sequence = ()):
        with self.lock:
            rows = self.db.execute(sql, args).fetchall()
            self.db.commit()
            return rows


def make_client():
    """The OpenAI-compatible client for live runs (key and base URL from the
    environment). A module-level factory so a caller can wrap it."""
    from openai import OpenAI

    return OpenAI(max_retries=2, timeout=180)


def fixture_answer(prompt: str) -> str:
    """Deterministic stand-in for a model, driven by the fixture file: each item
    says whether the starting skill answers it and which rules flip that."""
    for item in json.loads(FIXTURE.read_text()):
        if f"## Question\n{item['question']}\n" not in prompt + "\n":
            continue
        correct = bool(item["base_correct"])
        for _name, rule in RULES["fixture"]:
            if rule in prompt and rule in item.get("flips", []):
                correct = not correct
        return f"<answer>{item['answers'][0] if correct else 'no idea'}</answer>"
    return "<answer>no idea</answer>"


def chat_backend(model: str, store: Store, *, offline: bool = False):
    from skillopt_sleep.backend import CliBackend

    client = None if offline else make_client()
    key_locks: Dict[str, threading.Lock] = {}
    key_locks_guard = threading.Lock()

    class ChatBackend(CliBackend):
        name = "openai-compatible"
        generation_tools_disabled = True

        def __init__(self) -> None:
            super().__init__(model=model, timeout=180)
            self.tag = "test"
            self.calls = 0
            self._sample = threading.local()

        def start(self, tag: str) -> None:
            """Draws are tagged per phase: a night must not reuse another night's draw."""
            self.tag = tag
            with self._lock:
                self._cache.clear()

        def attempt(self, task, skill, memory, sample_id: int = 0):
            self._sample.value = f"s{sample_id}"
            try:
                return super().attempt(task, skill, memory, sample_id=sample_id)
            finally:
                self._sample.value = ""

        def _call(self, prompt: str, *, max_tokens: int = 1024) -> str:
            self.calls += 1
            salt = f"{self.tag}:" + (getattr(self._sample, "value", "") or "x")
            key = hashlib.sha256(f"{model}\0{salt}\0{prompt}".encode()).hexdigest()
            with key_locks_guard:   # two workers on one prompt pay for it once
                key_lock = key_locks.setdefault(key, threading.Lock())
            with key_lock:
                hit = store.get(key)
                if hit is not None and hit[0]:
                    return hit[0]
                if offline:
                    return store.put(key, model, fixture_answer(prompt), 0, 0)
                for attempt in range(8):
                    try:
                        resp = client.chat.completions.create(
                            model=model, messages=[{"role": "user", "content": prompt}],
                            max_completion_tokens=16384)
                        text = (resp.choices[0].message.content or "").strip()
                        if text:
                            return store.put(key, model, text, resp.usage.prompt_tokens,
                                             resp.usage.completion_tokens)
                    except RuntimeError:
                        raise   # a caller's spend limit; do not retry past it
                    except Exception:  # noqa: BLE001 - retried, then raised below
                        pass
                    time.sleep(min(60.0, 2.0 ** attempt))
                raise RuntimeError("provider call failed; rerun to resume from the store")

    return ChatBackend()


def install_judge() -> None:
    """The shipped backend scores ``reference_kind == 'answer'`` tasks through
    this hook; wire it to the in-tree benchmark evaluators."""
    from skillopt.envs.livemathematicianbench.evaluator import evaluate as livemath_eval
    from skillopt.envs.searchqa.evaluator import evaluate as searchqa_eval

    module = types.ModuleType("skillopt_sleep.experiments.real_eval")

    def score_answer_judge(judge, response):
        if judge.get("benchmark") == "livemath":
            r = livemath_eval(response or "", dict(judge["correct_choice"]), list(judge["choices"]))
        else:
            r = searchqa_eval(response or "", list(judge.get("answers") or []))
        return float(r["em"]), float(r["f1"]), f"em={r['em']:.0f} f1={r['f1']:.2f}"

    module.score_answer_judge = score_answer_judge
    sys.modules["skillopt_sleep.experiments.real_eval"] = module


def materialize_livemath(repo: Path) -> None:
    """Write runnable LiveMathematicianBench splits from the released ID manifest
    (raw monthly files fetched at the manifest's pinned revision). Choices are
    shuffled per item with the dataloader's own rule, seed 42."""
    from huggingface_hub import hf_hub_download

    from skillopt.envs.livemathematicianbench.dataloader import (
        LiveMathematicianBenchDataLoader as Loader,
    )
    from skillopt.envs.livemathematicianbench.dataloader import load_items

    manifest_dir = repo / "data" / "livemathematicianbench_id_split"
    manifest = json.loads((manifest_dir / "split_manifest.json").read_text())
    by_id = {}
    for name in manifest["source_files"]:
        path = hf_hub_download(LIVEMATH_REPO, name, repo_type="dataset",
                               revision=manifest["source_revision"])
        month = name.split("/")[1]
        for item in load_items(path):
            item_id = item["id"] if ":" in str(item["id"]) else f"{month}:{item['no']}"
            by_id[item_id] = {**item, "id": item_id, "month": month}
    loader = Loader(shuffle_choices=True)
    for split in ("train", "val", "test"):
        wanted = [row["id"] for row in json.loads((manifest_dir / split / "items.json").read_text())]
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise SystemExit(f"livemath: {len(missing)} manifest ids not in the raw files, e.g. {missing[:3]}")
        items = [loader._shuffle_item_choices(by_id[i], 42) for i in wanted]
        out = repo / "data" / "livemathematicianbench_split" / split
        out.mkdir(parents=True, exist_ok=True)
        (out / "items.json").write_text(json.dumps(items, indent=1))
        print(f"livemath {split}: {len(items)} items -> {out / 'items.json'}")


def benchmark(name: str, repo: Path):
    """Return (task builder, items(split), starting skill) for a benchmark."""
    from skillopt_sleep.memory import ensure_skill_scaffold
    from skillopt_sleep.types import TaskRecord

    env = {"searchqa": "searchqa", "fixture": "searchqa", "livemath": "livemathematicianbench"}[name]
    system = (repo / f"skillopt/envs/{env}/prompts/rollout_system.md").read_text()
    initial = (repo / f"skillopt/envs/{env}/skills/initial.md").read_text()

    def task(item):
        if name == "livemath":
            choices = "\n".join(f"{c['label']}. {c['text']}" for c in item["choices"])
            body, judge = f"## Choices\n{choices}", {
                "benchmark": "livemath", "correct_choice": item["correct_choice"],
                "choices": item["choices"]}
        else:
            body, judge = f"## Context\n{item['context']}", {
                "benchmark": "searchqa", "answers": list(item["answers"])}
        return TaskRecord(
            id=f"{name}:{item['id']}", project=f"bench/{name}",
            intent=f"## Question\n{item['question']}", context_excerpt=body, system=system,
            reference_kind="answer", judge=judge, tags=[name], split="train", origin="real")

    def items(split):
        if name == "fixture":
            rows = json.loads(FIXTURE.read_text())
            return rows[:20] if split == "train" else rows[20:]
        folder = {"searchqa": "searchqa_split", "livemath": "livemathematicianbench_split"}[name]
        path = repo / "data" / folder
        if not (path / "test" / "items.json").exists():
            raise SystemExit(f"{path} is missing; run the materialize step in this module's docstring")
        rows = json.loads((path / split / "items.json").read_text())
        if name == "livemath" and split == "train":   # 35 + 17: the manifest's val is unused here
            rows = rows + json.loads((path / "val" / "items.json").read_text())
        return rows

    skill = ensure_skill_scaffold(initial, name="skillopt-sleep-learned",
                                  description="Learned preferences.")
    return task, items, skill


def held_out_em(backend, tests, skill) -> float:
    from skillopt_sleep.replay import replay_batch

    pairs = replay_batch(backend, tests, skill, "")
    return sum(r.hard for _t, r in pairs) / len(pairs)


def expected_calls(args, n_rules: int, n_test: int) -> int:
    """An upper estimate of model calls, printed before a live run starts."""
    contexts = args.seeds * args.nights
    if args.command == "effects":
        return n_test * (1 + n_rules)
    if args.command == "faults":
        return contexts * n_rules * 22
    return contexts * 225 + args.seeds * (args.nights + 1) * n_test


def run_effects(args, store: Store) -> None:
    from skillopt_sleep.memory import apply_edits_detailed
    from skillopt_sleep.types import EditRecord

    backend = chat_backend(args.model, store, offline=args.offline)
    task, items, skill = benchmark(args.benchmark, Path(args.repo).resolve())
    tests = [task(i) for i in items("test")[: args.test_n]]
    backend.start("test")
    candidates = [("(starting skill)", skill)]
    for name, rule in RULES[args.benchmark]:
        edit = EditRecord(target="skill", op="add", content=rule)
        candidates.append((name, apply_edits_detailed(skill, [edit])[0]))
    for name, text in candidates:
        em = held_out_em(backend, tests, text)
        store.run("INSERT OR REPLACE INTO effects VALUES (?,?,?)", (args.model, name, em))
        print(f"{name:<22} held-out EM {100 * em:5.1f} (n={len(tests)})", flush=True)


def run_faults(args, store: Store) -> None:
    from skillopt_sleep.dream import dream_consolidate
    from skillopt_sleep.mine import assign_splits
    from skillopt_sleep.types import EditRecord

    backend = chat_backend(args.model, store, offline=args.offline)
    task, items, skill = benchmark(args.benchmark, Path(args.repo).resolve())
    feature = json.loads(args.feature or "{}")
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        order = items("train")
        random.Random(seed).shuffle(order)
        archive: list = []
        for night in range(1, args.nights + 1):
            tasks = [task(i) for i in order[(night - 1) * 10: night * 10]]
            assign_splits(tasks, val_fraction=0.34, seed=seed)
            for name, rule in RULES[args.benchmark]:
                if store.run("SELECT 1 FROM decisions WHERE arm=? AND model=? AND seed=? AND night=? "
                             "AND rule=?", (args.arm, args.model, seed, night, name)):
                    continue
                proposal = [EditRecord(target="skill", op="add", content=rule, rationale="injected")]
                backend.reflect = lambda *a, _p=proposal, **k: list(_p)
                backend.start(f"inj{night}")
                before = backend.calls
                result = dream_consolidate(
                    backend, tasks, skill, "", history_tasks=list(archive), recall_k=10,
                    dream_rollouts=1, dream_factor=0, edit_budget=4, gate_metric="mixed",
                    gate_mixed_weight=0.5, gate_mode="on", evolve_skill=True,
                    evolve_memory=False, night=night, **feature)
                store.run("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?,?)",
                          (args.arm, args.model, seed, night, name, int(result.accepted),
                           backend.calls - before))
            archive.extend(tasks)
        print(f"faults arm {args.arm} seed {seed} done", flush=True)


def run_nights(args, store: Store) -> None:
    from skillopt_sleep.backend import DualBackend
    from skillopt_sleep.dream import dream_consolidate
    from skillopt_sleep.mine import assign_splits

    target = chat_backend(args.model, store, offline=args.offline)
    optimizer = chat_backend(args.optimizer, store, offline=args.offline)
    backend = DualBackend(target, optimizer)
    task, items, initial = benchmark(args.benchmark, Path(args.repo).resolve())
    tests = [task(i) for i in items("test")[: args.test_n]]
    feature = json.loads(args.feature or "{}")

    def start(tag: str) -> None:
        target.start(tag)
        optimizer.start(tag)

    for seed in range(args.first_seed, args.first_seed + args.seeds):
        if store.run("SELECT 1 FROM curves WHERE arm=? AND model=? AND seed=? AND night=?",
                     (args.arm, args.model, seed, args.nights)):
            continue
        order = items("train")
        random.Random(seed).shuffle(order)
        skill, archive, rows = initial, [], []
        start("test")
        rows.append((0, 0, held_out_em(backend, tests, skill), 0))
        for night in range(1, args.nights + 1):
            tasks = [task(i) for i in order[(night - 1) * 10: night * 10]]
            assign_splits(tasks, val_fraction=0.34, seed=seed)
            start(f"n{night}")
            before = target.calls
            result = dream_consolidate(
                backend, tasks, skill, "", history_tasks=list(archive), recall_k=10,
                dream_rollouts=5, dream_factor=2, edit_budget=4, gate_metric="mixed",
                gate_mixed_weight=0.5, gate_mode="on", evolve_skill=True,
                evolve_memory=False, night=night, **feature)
            spent = target.calls - before
            if result.accepted:
                skill = result.new_skill
            archive.extend(tasks)
            start("test")
            rows.append((night, int(result.accepted), held_out_em(backend, tests, skill), spent))
        for night, accepted, em, calls in rows:
            store.run("INSERT OR REPLACE INTO curves VALUES (?,?,?,?,?,?,?)",
                      (args.arm, args.model, seed, night, accepted, em, calls))
        print(f"nights arm {args.arm} seed {seed}: "
              + " ".join(f"{100 * em:.1f}" for _n, _a, em, _c in rows), flush=True)


def summarize(store: Store, model: str, rules: Sequence[Tuple[str, str]]) -> dict:
    """Every number ``report`` prints, as data (the test suite pins this)."""
    effects = dict(store.run("SELECT rule, em FROM effects WHERE model=?", (model,)))
    base = effects.get("(starting skill)")
    table: dict = defaultdict(lambda: defaultdict(list))
    for arm, rule, accepted, calls in store.run(
            "SELECT arm, rule, accepted, calls FROM decisions WHERE model=?", (model,)):
        table[rule][arm].append((accepted, calls))
    arms = sorted({arm for rule in table for arm in table[rule]})
    faults = {
        name: {
            "effect": (None if base is None or name not in effects
                       else round(100 * (effects[name] - base), 1)),
            "arms": {arm: {"accepted": sum(a for a, _c in table[name][arm]),
                           "decisions": len(table[name][arm])}
                     for arm in arms if table[name][arm]},
        }
        for name, _rule in rules
    }
    calls = {}
    for arm in arms:
        got = [x for name, _r in rules for x in table[name][arm]]
        calls[arm] = round(sum(c for _a, c in got) / max(1, len(got)), 1)
    by: dict = defaultdict(lambda: defaultdict(list))
    for arm, seed, night, accepted, em, n in store.run(
            "SELECT arm, seed, night, accepted, em, calls FROM curves WHERE model=? "
            "ORDER BY arm, seed, night", (model,)):
        by[arm][seed].append((night, accepted, em, n))
    curves = {}
    for arm in sorted(by):
        seeds = sorted(by[arm])
        nights = len(by[arm][seeds[0]])
        curve = [sum(100 * by[arm][s][k][2] for s in seeds) / len(seeds) for k in range(nights)]
        first = [next((n for n, a, _e, _c in by[arm][s] if a), nights) for s in seeds]
        curves[arm] = {
            "seeds": len(seeds),
            "curve": [round(v, 1) for v in curve],
            "accepted_nights": sum(a for s in seeds for _n, a, _e, _c in by[arm][s]),
            "first_accepted_night": round(sum(first) / len(first), 2),
            "target_calls_per_night": round(
                sum(c for s in seeds for _n, _a, _e, c in by[arm][s][1:])
                / max(1, len(seeds) * (nights - 1))),
        }
    return {"model": model, "starting_em": None if base is None else round(100 * base, 1),
            "faults": faults, "calls_per_decision": calls, "curves": curves}


def report(args, store: Store) -> None:
    data = summarize(store, args.model, RULES[args.benchmark])
    if args.json:
        print(json.dumps(data, indent=1))
        return
    arms = sorted(data["calls_per_decision"])
    if arms:
        print(f"FAULT INJECTION, model {data['model']}; starting skill held-out EM "
              f"{'n/a' if data['starting_em'] is None else data['starting_em']}")
        print(f"{'rule':<22} {'held-out effect':>15}  " + "  ".join(f"{a:>16}" for a in arms))
        for name, row in data["faults"].items():
            effect = "n/a" if row["effect"] is None else f"{row['effect']:+.1f}"
            cells = [(f"{row['arms'][a]['accepted']:>3}/{row['arms'][a]['decisions']:<3} accepted".rjust(16)
                      if a in row["arms"] else " " * 16) for a in arms]
            print(f"{name:<22} {effect:>15}  " + "  ".join(cells))
        for arm in arms:
            total = sum(row["arms"][arm]["decisions"] for row in data["faults"].values()
                        if arm in row["arms"])
            print(f"arm {arm}: {total} decisions, "
                  f"{data['calls_per_decision'][arm]:.1f} model calls per decision")
    if data["curves"]:
        print(f"\nLEARNING CURVES, target {data['model']}: held-out EM before night 1 and after each night")
        for arm, c in data["curves"].items():
            print(f"arm {arm:<8} seeds {c['seeds']:>2}  curve {' '.join(f'{v:5.1f}' for v in c['curve'])}  "
                  f"accepted nights {c['accepted_nights']}  "
                  f"first accepted night {c['first_accepted_night']:.2f}  "
                  f"target calls/night {c['target_calls_per_night']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["materialize", "effects", "faults", "nights", "report"])
    ap.add_argument("--benchmark", default="searchqa", choices=sorted(RULES))
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--model", default="gpt-5.4-nano", help="the agent being improved")
    ap.add_argument("--optimizer", default="gpt-5.5", help="reflection model for `nights`")
    ap.add_argument("--arm", default="base")
    ap.add_argument("--feature", default="{}", help="JSON kwargs for dream_consolidate")
    ap.add_argument("--first-seed", type=int, default=101)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--nights", type=int, default=5)
    ap.add_argument("--test-n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--store", default="sleep_gate_study.sqlite")
    ap.add_argument("--live", action="store_true", help="allow paid model calls on your own key")
    ap.add_argument("--offline", action="store_true",
                    help="use the committed fixture and a deterministic stand-in model")
    ap.add_argument("--json", action="store_true", help="`report` as JSON")
    args = ap.parse_args(argv)
    if args.offline:
        args.benchmark, args.model, args.optimizer = "fixture", "fixture-model", "fixture-model"
        args.seeds, args.nights = min(args.seeds, 8), min(args.nights, 2)
    os.environ["SKILLOPT_SLEEP_WORKERS"] = str(args.workers)
    if args.command == "materialize":
        if args.benchmark != "livemath":
            raise SystemExit("searchqa: run scripts/materialize_searchqa.py")
        materialize_livemath(Path(args.repo).resolve())
        return 0
    store = Store(args.store)
    if args.command == "report":
        report(args, store)
        return 0
    if not (args.live or args.offline):
        raise SystemExit("this command calls a model: pass --live (your key, your cost) or --offline")
    install_judge()
    if args.live:
        _t, items, _s = benchmark(args.benchmark, Path(args.repo).resolve())
        n = expected_calls(args, len(RULES[args.benchmark]), len(items("test")[: args.test_n]))
        print(f"live run: up to about {n} model calls on {args.model} "
              f"(fewer when the store already holds them)", flush=True)
    runner: Callable = {"effects": run_effects, "faults": run_faults, "nights": run_nights}[args.command]
    runner(args, store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
