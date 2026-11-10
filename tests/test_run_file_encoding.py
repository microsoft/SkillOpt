"""Run files are read and written as UTF-8, whatever the locale codec is.

``open()`` without an encoding uses the locale codec, which on Windows is the
ANSI code page (cp1252, GBK, ...). Released skills, retrieved passages, model
replies and optimizer edits routinely hold characters outside it, so a run
crashed loading a skill, scored a correct answer 0 when writing its trajectory
failed inside a broad ``except``, or showed the target a garbled skill.

On a UTF-8 runner the locale codec is UTF-8, so behaviour alone cannot show the
defect there. The pipeline tests therefore run in a subprocess under
``-X warn_default_encoding`` with ``EncodingWarning`` raised as an error for
SkillOpt's own modules: every ``open()`` on the exercised path that leaves the
codec to the locale fails, on every platform.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RELEASED_SKILL = ROOT / "ckpt" / "searchqa" / "gpt5.5_skill.md"

ARROW = "\N{RIGHTWARDS ARROW}"
OMEGA = "\N{GREEK CAPITAL LETTER OMEGA}"
QUOTED = "\N{LEFT DOUBLE QUOTATION MARK}Paris\N{RIGHT DOUBLE QUOTATION MARK}"
PRODUCT = "\N{N-ARY PRODUCT}"


# Scoped to SkillOpt's modules so that a dependency leaving the codec to the
# locale cannot fail these tests.
_STRICT = (
    "import warnings\n"
    "warnings.filterwarnings('error', category=EncodingWarning, module='skillopt|scripts')\n"
)


def _run_strict(script: str, *args: str) -> str:
    """Run ``script`` in a subprocess where a locale-codec ``open()`` raises."""
    proc = subprocess.run(
        [sys.executable, "-X", "warn_default_encoding", "-c", _STRICT + script, *args],
        cwd=ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert proc.returncode == 0, f"{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    return proc.stdout


def _searchqa_split(root: Path) -> Path:
    for split, size in (("train", 2), ("val", 2), ("test", 1)):
        items = [
            {
                "id": f"{split}_{i}",
                "question": "Which city is the capital of France?",
                "context": "[DOC] Paris is the capital of France.",
                "answers": ["Paris"],
            }
            for i in range(size)
        ]
        (root / split).mkdir(parents=True)
        (root / split / "items.json").write_text(json.dumps(items), encoding="utf-8")
    return root


SEARCHQA_ROLLOUT = """
import json, sys
import skillopt.envs.searchqa.rollout as rollout

reply = "The passage names the city " + chr(0x2192) + " <answer>Paris</answer>"
rollout.chat_target = lambda **kwargs: (reply, {})
item = {
    "id": "q1",
    "question": "Which city is the capital of France?",
    "context": "[DOC] Paris (" + chr(0x3A9) + ") is the capital of France.",
    "answers": ["Paris"],
}
skill = "Quote the passage " + chr(0x2192) + " then answer."
first = rollout.run_batch([item], sys.argv[1], skill, workers=1)

def finished_items_are_not_rerun(**kwargs):
    raise AssertionError("resume rolled out a finished item again")

rollout.chat_target = finished_items_are_not_rerun
resumed = rollout.run_batch([item], sys.argv[1], skill, workers=1)
print(json.dumps({"first": first, "resumed": resumed}))
"""


def test_searchqa_rollout_scores_and_keeps_non_ascii_text(tmp_path):
    """A correct reply scores 1 whatever characters the context, skill or reply hold.

    The prompts and the trajectory are written before scoring, inside a broad
    ``except``: on cp1252 an ``Omega`` in a retrieved passage made the write
    raise, so the correct answer scored 0 while ``agent_ok`` stayed true and
    the systemic-failure check never fired. An arrow in the reply then crashed
    the run writing ``results.jsonl``.
    """
    out = json.loads(_run_strict(SEARCHQA_ROLLOUT, str(tmp_path)).splitlines()[-1])

    (row,) = out["first"]
    assert (row["hard"], row["fail_reason"]) == (1, "")
    assert out["resumed"] == out["first"]
    pred = tmp_path / "predictions" / "q1"
    assert ARROW in (pred / "target_system_prompt.txt").read_text(encoding="utf-8")
    assert OMEGA in (pred / "target_user_prompt.txt").read_text(encoding="utf-8")
    conversation = json.loads((pred / "conversation.json").read_text(encoding="utf-8"))
    assert ARROW in conversation[0]["content"]
    (record,) = (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert ARROW in json.loads(record)["response"]


EVAL_ONLY = """
import sys
import skillopt.envs.searchqa.rollout as rollout

rollout.chat_target = lambda **kwargs: ("<answer>Paris</answer>", {})
from scripts import eval_only

sys.argv = ["eval_only.py", *sys.argv[1:]]
eval_only.main()
"""


def test_eval_only_loads_a_released_skill(tmp_path):
    """``ckpt/README.md`` documents ``eval_only.py --skill ckpt/<bench>/gpt5.5_skill.md``.

    Five of the six released skills hold a right double quotation mark, whose
    UTF-8 bytes end in 0x9D, a byte cp1252 does not define, so on Windows the
    documented command crashed before any model call.
    """
    out = tmp_path / "eval"
    _run_strict(
        EVAL_ONLY,
        "--config", "configs/searchqa/default.yaml",
        "--skill", str(RELEASED_SKILL),
        "--split", "valid_unseen",
        "--split_dir", str(_searchqa_split(tmp_path / "split")),
        "--backend", "openai_compatible",
        "--target_model", "fake",
        "--out_root", str(out),
    )

    assert json.loads((out / "eval_summary.json").read_text(encoding="utf-8"))["hard"] == 1.0
    prompt = (out / "predictions" / "test_0" / "target_system_prompt.txt").read_text(encoding="utf-8")
    assert RELEASED_SKILL.read_text(encoding="utf-8").strip() in prompt


TRAINING_RUN = """
import json, sys
import skillopt.model.openai_compatible_backend as backend

edit = {"op": "append", "content": "- Copy the entity name " + chr(0x2192) + " never paraphrase it."}
patch = json.dumps({"reasoning": "exact names", "patch": {"edits": [edit]}, "edits": [edit]})
backend.chat_target = lambda **kwargs: ("<answer>Paris</answer>", {})
backend.chat_optimizer = lambda **kwargs: (patch, {})
from scripts import train

sys.argv = ["train.py", *sys.argv[1:]]
train.main()
"""


def test_training_step_keeps_a_released_skill_and_non_ascii_edits(tmp_path):
    """A released skill as ``skill_init`` and an optimizer edit with an arrow.

    On Windows, reading the skill crashed before the first rollout, and an
    arrow in an edit crashed the step when the analyst's patch was saved. The
    trainer writes the same edit into the merged patch, the ranked edits, the
    edit-apply report and the candidate skill.
    """
    out = tmp_path / "run"
    _run_strict(
        TRAINING_RUN,
        "--config", "configs/searchqa/default.yaml",
        "--skill_init", str(RELEASED_SKILL),
        "--split_dir", str(_searchqa_split(tmp_path / "split")),
        "--backend", "openai_compatible",
        "--optimizer_model", "fake",
        "--target_model", "fake",
        "--train_size", "2",
        "--batch_size", "2",
        "--num_epochs", "1",
        "--workers", "1",
        "--analyst_workers", "1",
        "--use_slow_update", "false",
        "--use_meta_skill", "false",
        "--out_root", str(out),
    )

    released = RELEASED_SKILL.read_text(encoding="utf-8")
    assert (out / "skills" / "skill_v0000.md").read_text(encoding="utf-8") == released
    step = out / "steps" / "step_0001"
    patch = json.loads((step / "patches" / "minibatch_succ_000.json").read_text(encoding="utf-8"))
    assert ARROW in patch["edits"][0]["content"]
    assert f"- Copy the entity name {ARROW} never paraphrase it." in (
        step / "candidate_skill.md"
    ).read_text(encoding="utf-8")
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["test_hard"] == 1.0


REFLECT_TRAJECTORIES = """
import sys
from skillopt.gradient.reflect import fmt_minibatch_trajectories

print(fmt_minibatch_trajectories([{"id": "t1", "hard": 0}], sys.argv[1]))
"""


def test_reflection_reads_utf8_trajectories(tmp_path):
    """DocVQA and OfficeQA already write trajectories as UTF-8.

    Reflection read them with the locale codec, before the analyst's ``try``,
    so a closing curly quote in one crashed the reflect stage on Windows.
    """
    task = tmp_path / "t1"
    task.mkdir()
    conversation = [{"type": "message", "turn": 1, "content": f"The passage says {QUOTED}."}]
    (task / "conversation.json").write_text(json.dumps(conversation, ensure_ascii=False), encoding="utf-8")
    (task / "target_system_prompt.txt").write_text(f"Quote the passage {ARROW} then answer.", encoding="utf-8")
    (task / "target_user_prompt.txt").write_text(f"[DOC] Paris ({OMEGA})", encoding="utf-8")

    text = _run_strict(REFLECT_TRAJECTORIES, str(tmp_path))

    assert QUOTED in text
    assert ARROW in text
    assert OMEGA in text


DATASET_LOADERS = """
import json, sys
from skillopt.envs.livemathematicianbench.dataloader import load_items as load_livemath
from skillopt.envs.searchqa.dataloader import SearchQADataLoader
from skillopt.envs.spreadsheetbench.rollout import load_items as load_spreadsheetbench

root = sys.argv[1]
print(json.dumps({
    "livemath": load_livemath(root + "/livemath")[0]["question"],
    "searchqa": SearchQADataLoader().load_raw_items(root + "/searchqa.json")[0]["question"],
    "spreadsheetbench": load_spreadsheetbench(root + "/spreadsheetbench.json")[0]["instruction"],
}))
"""


def test_benchmark_data_files_are_read_as_utf8(tmp_path):
    """Raw LiveMathematicianBench files failed to load on Windows (#117)."""
    question = f"Evaluate {PRODUCT} (1 + 1/k^2) over k >= 1."
    (tmp_path / "livemath").mkdir()
    (tmp_path / "livemath" / "qa_smoke_final.json").write_text(
        json.dumps(
            [{"no": 1, "question": question, "choices": {"A": "2", "B": "e"}, "correct_choice": "A"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "searchqa.json").write_text(
        json.dumps([{"id": "q1", "question": question, "answers": ["2"]}], ensure_ascii=False), encoding="utf-8",
    )
    (tmp_path / "spreadsheetbench.json").write_text(
        json.dumps([{"id": "1-1", "instruction": question}], ensure_ascii=False), encoding="utf-8",
    )

    loaded = json.loads(_run_strict(DATASET_LOADERS, str(tmp_path)).splitlines()[-1])

    assert loaded == {"livemath": question, "searchqa": question, "spreadsheetbench": question}


SPREADSHEETBENCH_CODEGEN = """
import json, sys
import skillopt.envs.spreadsheetbench.codegen_agent as codegen_agent
from skillopt.envs.spreadsheetbench import rollout

data_root, out_root, skill_path = sys.argv[1:4]
with open(skill_path, encoding="utf-8") as f:
    skill = f.read()
codegen_agent.run_single = lambda **kwargs: {"code": "", "raw": "No code " + chr(0x2192), "n_turns": 1}
item = {
    "id": "1-1",
    "instruction": "put 1 in A1",
    "instruction_type": "cell",
    "answer_position": "A1",
    "answer_sheet": "Sheet1",
    "spreadsheet_path": "1-1",
}
print(json.dumps(rollout.process_one_codegen(item, data_root, out_root, skill)))
"""


def test_spreadsheetbench_prompts_keep_the_seed_skill(tmp_path):
    """The SpreadsheetBench seed skill holds an em dash and an arrow.

    Read with cp1252 they reached every prompt as mojibake. Read correctly,
    they made the rollout's own prompt writes fail inside its broad
    ``except``, so the item scored 0 before the model was called; the skill
    read and the rollout writes have to change together.
    """
    seed = ROOT / "skillopt" / "envs" / "spreadsheetbench" / "skills" / "initial.md"
    case = tmp_path / "data" / "1-1"
    case.mkdir(parents=True)
    (case / "1_input.xlsx").write_bytes(b"")
    (case / "1_answer.xlsx").write_bytes(b"")
    out = tmp_path / "out"

    result = json.loads(
        _run_strict(SPREADSHEETBENCH_CODEGEN, str(tmp_path / "data"), str(out), str(seed)).splitlines()[-1]
    )

    assert result["fail_reason"] == "empty-code-block", result
    system_prompt = (out / "predictions" / "1-1" / "target_system_prompt.txt").read_text(encoding="utf-8")
    assert "\N{EM DASH}" in system_prompt
    assert ARROW in system_prompt
    assert ARROW in (out / "predictions" / "1-1" / "raw.txt").read_text(encoding="utf-8")


def test_resume_reads_files_an_earlier_release_wrote_with_the_locale_codec(tmp_path, monkeypatch):
    """Runs started before this change hold locale-codec bytes and still resume.

    Reading them only as UTF-8 would fail, and for ``runtime_state.json`` the
    failure is swallowed, so the resume falls back to ``history.json``, or
    starts over when no step has finished yet. A rollout
    resumed after the upgrade appends UTF-8 records to such a ``results.jsonl``,
    so its lines are decoded one by one. The locale codec is forced to cp1252
    so the fallback is exercised on a UTF-8 runner as well.
    """
    from skillopt.engine import trainer
    from skillopt.envs.searchqa import rollout
    from skillopt.utils import text_io

    monkeypatch.setattr(text_io, "locale_codec", lambda: "cp1252")
    run = tmp_path / "run"
    skill_path = run / "skills" / "skill_v0003.md"
    skill_path.parent.mkdir(parents=True)
    skill = f"# Skill\n- Quote the passage \N{EM DASH} word for word: {QUOTED}.\n"
    resume = "r\N{LATIN SMALL LETTER E WITH ACUTE}sum\N{LATIN SMALL LETTER E WITH ACUTE}"
    history = [{"step": 3, "best_step": 3, "best_score": 0.5, "note": f"Keep the {resume} short"}]
    state = {
        "last_completed_step": 3,
        "current_skill_path": str(skill_path),
        "current_origin": "step_0003 \N{EM DASH} accepted",
    }
    skill_path.write_bytes(skill.encode("cp1252"))
    (run / "history.json").write_bytes(json.dumps(history, ensure_ascii=False).encode("cp1252"))
    (run / "runtime_state.json").write_bytes(json.dumps(state, ensure_ascii=False).encode("cp1252"))
    for path in (skill_path, run / "history.json", run / "runtime_state.json"):
        with pytest.raises(UnicodeDecodeError):  # the fixtures must not be valid UTF-8
            path.read_bytes().decode("utf-8")

    assert trainer._load_runtime_state(str(run)) == state
    assert trainer._load_history(str(run)) == history
    assert trainer._load_skill(str(run), 3) == skill

    legacy = {"id": "q1", "hard": 1, "response": "Paris \N{EM DASH} <answer>Paris</answer>"}
    upgraded = {"id": "q2", "hard": 1, "response": f"{QUOTED} {ARROW} <answer>Paris</answer>"}
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    (rollout_dir / "results.jsonl").write_bytes(
        (json.dumps(legacy, ensure_ascii=False) + "\n").encode("cp1252")
        + (json.dumps(upgraded, ensure_ascii=False) + "\n").encode("utf-8")
    )

    def no_model_call(**kwargs):
        raise AssertionError("resume rolled out a finished item again")

    monkeypatch.setattr(rollout, "chat_target", no_model_call)
    items = [{"id": item_id, "question": "", "answers": ["Paris"]} for item_id in ("q1", "q2")]
    assert rollout.run_batch(items, str(rollout_dir), "", workers=1) == [legacy, upgraded]


def test_resume_rolls_out_a_record_cut_off_mid_write_again(tmp_path, monkeypatch):
    """A crash can leave the last ``results.jsonl`` record cut off mid-character.

    That line is valid in neither codec. It is treated like any other line that
    does not parse, so its item is rolled out again instead of the resume
    failing on a decode error.
    """
    from skillopt.envs.searchqa import rollout
    from skillopt.utils import text_io

    monkeypatch.setattr(text_io, "locale_codec", lambda: "cp1252")
    done = {"id": "q1", "hard": 1, "response": "<answer>Paris</answer>"}
    cut_off = json.dumps({"id": "q2", "hard": 1, "response": f"{QUOTED} {ARROW}"}, ensure_ascii=False)
    (tmp_path / "results.jsonl").write_bytes(
        (json.dumps(done) + "\n").encode("utf-8") + cut_off.encode("utf-8")[:-4]
    )
    monkeypatch.setattr(rollout, "chat_target", lambda **kwargs: ("<answer>Paris</answer>", {}))

    items = [{"id": item_id, "question": "", "answers": ["Paris"]} for item_id in ("q1", "q2")]
    rows = rollout.run_batch(items, str(tmp_path), "", workers=1)

    assert rows[0] == done
    assert (rows[1]["id"], rows[1]["hard"], rows[1]["fail_reason"]) == ("q2", 1, "")
