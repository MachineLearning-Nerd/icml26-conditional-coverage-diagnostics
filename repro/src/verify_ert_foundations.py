#!/usr/bin/env python3
"""Independent deterministic checks for anchored claims 1, 4, and 6.

The checks call the pinned covmetrics implementation directly, but use a
separate finite construction and split recorder.  They intentionally do not
reuse any historical experiment CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from sklearn.model_selection import KFold

from covmetrics.ERT import ERT, evaluate_with_predictions
from covmetrics.losses import (
    L1_miscoverage,
    L1_miscoverage_over,
    L1_miscoverage_under,
    brier_score,
    logloss,
)


TARGET = 0.9


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


class RecordingConstantClassifier:
    """A target-coverage classifier that exposes every source-CV split."""

    records: list[dict[str, list[int]]] = []

    def fit(self, x: np.ndarray, cover: np.ndarray) -> "RecordingConstantClassifier":
        self.train_ids = sorted(int(identifier) for identifier in x[:, 0])
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        test_ids = sorted(int(identifier) for identifier in x[:, 0])
        self.records.append({"train": self.train_ids, "test": test_ids})
        probability = np.full(len(x), TARGET)
        return np.column_stack([1.0 - probability, probability])


def verify_constant_predictor() -> dict:
    # Every one of the 20 feature cells has exactly nine covered and one
    # uncovered outcome, so P(cover | cell) is exactly 1-alpha.  Candidate
    # cellwise classifiers cannot beat the constant target in any listed loss.
    cover = np.tile(np.array([1] * 9 + [0], dtype=int), 20)
    candidate_probabilities = np.repeat(np.linspace(0.05, 0.95, 20), 10)
    losses = {"L1": L1_miscoverage, "L2": brier_score, "KL": logloss}
    candidate_ert = {
        name: float(evaluate_with_predictions(candidate_probabilities, cover, 0.1, loss=loss))
        for name, loss in losses.items()
    }
    target_ert = {
        name: float(evaluate_with_predictions(np.full(len(cover), TARGET), cover, 0.1, loss=loss))
        for name, loss in losses.items()
    }
    if any(abs(value) > 1e-12 for value in target_ert.values()):
        raise AssertionError(f"target predictor must have zero excess risk: {target_ert}")
    if any(value > 1e-12 for value in candidate_ert.values()):
        raise AssertionError(f"a candidate beat exact conditional target coverage: {candidate_ert}")
    return {"target_ert": target_ert, "candidate_ert": candidate_ert}


def verify_asymmetric_decomposition() -> dict:
    conservative_cover = np.array([1] * 95 + [0] * 5, dtype=int)
    aggressive_cover = np.array([1] * 85 + [0] * 15, dtype=int)
    conservative = {
        "over": float(evaluate_with_predictions(np.full(100, 0.95), conservative_cover, 0.1, loss=L1_miscoverage_over)),
        "under": float(evaluate_with_predictions(np.full(100, 0.95), conservative_cover, 0.1, loss=L1_miscoverage_under)),
    }
    aggressive = {
        "over": float(evaluate_with_predictions(np.full(100, 0.85), aggressive_cover, 0.1, loss=L1_miscoverage_over)),
        "under": float(evaluate_with_predictions(np.full(100, 0.85), aggressive_cover, 0.1, loss=L1_miscoverage_under)),
    }
    if not (conservative["over"] > 0 and abs(conservative["under"]) < 1e-12):
        raise AssertionError(f"conservative case did not isolate over coverage: {conservative}")
    if not (abs(aggressive["over"]) < 1e-12 and aggressive["under"] > 0):
        raise AssertionError(f"aggressive case did not isolate under coverage: {aggressive}")
    return {"conservative": conservative, "aggressive": aggressive}


def verify_cross_validation() -> dict:
    x = np.column_stack([np.arange(50), np.linspace(-1.0, 1.0, 50)])
    cover = np.tile(np.array([1] * 9 + [0], dtype=int), 5)
    RecordingConstantClassifier.records = []
    ert = ERT(RecordingConstantClassifier)
    value = float(ert.evaluate(x, cover, alpha=0.1, n_splits=5, random_state=42))
    observed = RecordingConstantClassifier.records
    expected = [
        {"train": sorted(train.tolist()), "test": sorted(test.tolist())}
        for train, test in KFold(n_splits=5, shuffle=True, random_state=42).split(x)
    ]
    if observed != expected:
        raise AssertionError("source ERT fit/predict partitions differ from independent KFold audit")
    if abs(value) > 1e-12:
        raise AssertionError(f"constant target predictor should give zero cross-validated ERT, got {value}")
    return {"folds": len(observed), "cross_validated_l1_ert": value, "test_rows_per_fold": len(observed[0]["test"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/foundations/ert_foundations.json"))
    args = parser.parse_args()
    result = {
        "source": {
            "covmetrics_commit": "a5205aada6a0f39e3812daf087753217ef66b159",
            "conditional_coverage_commit": "39a99dcad92205a15d93f2c5fec40c76540abf1c",
        },
        "claims": {
            "1_constant_target_principle": verify_constant_predictor(),
            "4_asymmetric_decomposition": verify_asymmetric_decomposition(),
            "6_kfold_cross_validation": verify_cross_validation(),
        },
    }
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
