"""Deterministic scenario splitting.

One seeded shuffle decides which scenario ids land in train / validation /
test. The split is a pure function of (ids, seed, counts) so a reported
experiment can be reproduced bit-for-bit, and the sets are disjoint by
construction - validation and test scenarios are never seen by the
optimizer's proposal step.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class ScenarioSplit:
    train: List[str]
    validation: List[str]
    test: List[str]

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }


def split_scenarios(
    scenario_ids: Sequence[str],
    n_train: int,
    n_validation: int,
    n_test: int,
    seed: int,
) -> ScenarioSplit:
    ids = list(scenario_ids)
    total = n_train + n_validation + n_test
    if total != len(ids):
        raise ValueError(
            f"split counts ({n_train}+{n_validation}+{n_test}={total}) "
            f"must equal the number of scenarios ({len(ids)})"
        )
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate scenario ids")
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    return ScenarioSplit(
        train=sorted(shuffled[:n_train]),
        validation=sorted(shuffled[n_train:n_train + n_validation]),
        test=sorted(shuffled[n_train + n_validation:]),
    )
