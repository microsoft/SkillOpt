"""LiveMathematicianBench choice shuffling: the correct answer must survive relabeling."""

from __future__ import annotations

import copy
import json

import pytest

from skillopt.envs.livemathematicianbench.dataloader import LiveMathematicianBenchDataLoader

LABELS = ["A", "B", "C", "D", "E"]


def _item(i: int) -> dict:
    return {
        "id": f"2026-01:{i}",
        "question": f"Question {i}",
        "theorem_type": ["algebra"],
        "choices": [{"label": label, "text": f"choice {i}{label}"} for label in LABELS],
        "correct_choice": {"label": "C", "text": f"choice {i}C"},
    }


ITEMS = [_item(i) for i in range(8)]


@pytest.fixture
def make_loader(tmp_path):
    for split in ("train", "val", "test"):
        (tmp_path / split).mkdir()
        (tmp_path / split / "items.json").write_text(json.dumps(ITEMS), encoding="utf-8")

    def make(shuffle_choices: bool = True) -> LiveMathematicianBenchDataLoader:
        loader = LiveMathematicianBenchDataLoader(
            split_dir=str(tmp_path), split_mode="split_dir", shuffle_choices=shuffle_choices
        )
        loader.setup({})
        return loader

    return make


def _correct_text(item: dict) -> str:
    by_label = {choice["label"]: choice["text"] for choice in item["choices"]}
    return by_label[item["correct_choice"]["label"]]


def _assert_well_formed(item: dict, source: dict) -> None:
    assert [choice["label"] for choice in item["choices"]] == LABELS
    assert sorted(choice["text"] for choice in item["choices"]) == sorted(c["text"] for c in source["choices"])
    assert _correct_text(item) == source["correct_choice"]["text"] == item["correct_choice"]["text"]


def test_eval_batch_keeps_the_correct_answer_after_shuffling(make_loader):
    batch = make_loader().build_eval_batch(0, "val", seed=42).payload

    assert [item["id"] for item in batch] == [item["id"] for item in ITEMS]
    for item, source in zip(batch, ITEMS):
        _assert_well_formed(item, source)
    # The correct answer is not always left at its original label.
    assert {item["correct_choice"]["label"] for item in batch} != {"C"}


def test_eval_batch_is_deterministic_for_a_seed(make_loader):
    loader = make_loader()

    first = loader.build_eval_batch(0, "val", seed=42).payload
    again = loader.build_eval_batch(0, "val", seed=42).payload
    other = loader.build_eval_batch(0, "val", seed=7).payload

    # The selection gate compares candidates on the same choice order.
    assert first == again
    assert first != other


def test_shuffling_does_not_mutate_the_loaded_items(make_loader):
    loader = make_loader()
    before = copy.deepcopy(loader.val_items)

    loader.build_eval_batch(0, "val", seed=42)
    loader.build_train_batch(4, seed=3)

    assert loader.val_items == before
    assert loader.train_items == before


def test_shuffle_disabled_keeps_order_and_returns_copies(make_loader):
    loader = make_loader(shuffle_choices=False)

    batch = loader.build_eval_batch(0, "val", seed=42).payload

    assert batch == ITEMS
    assert batch[0]["choices"] is not loader.val_items[0]["choices"]
    assert batch[0]["correct_choice"] is not loader.val_items[0]["correct_choice"]


def test_eval_batch_respects_env_num(make_loader):
    batch = make_loader().build_eval_batch(3, "val", seed=42).payload

    assert [item["id"] for item in batch] == [item["id"] for item in ITEMS[:3]]


def test_train_batches_keep_the_correct_answer(make_loader):
    loader = make_loader()
    by_id = {item["id"]: item for item in ITEMS}

    batch = loader.build_train_batch(4, seed=3).payload
    epoch = loader.plan_train_epoch(epoch=0, steps_per_epoch=2, accumulation=2, batch_size=2, seed=1)

    assert len(batch) == 4
    for item in batch + [item for spec in epoch for item in spec.payload]:
        _assert_well_formed(item, by_id[item["id"]])
    # One epoch covers every training item exactly once.
    assert sorted(item["id"] for spec in epoch for item in spec.payload) == sorted(by_id)
