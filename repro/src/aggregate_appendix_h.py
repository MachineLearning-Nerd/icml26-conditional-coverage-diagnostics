#!/usr/bin/env python3
"""Create a fail-closed integrity manifest for all eight Appendix-H datasets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile


DATASETS = (
    "ailerons",
    "diamonds",
    "winequality",
    "miami2016",
    "o11",
    "superconductivity",
    "deliverytime",
    "protein",
)
PROTOCOL = {"split": "40/10/50", "alpha": 0.1, "ert_folds": 5, "device": "cpu"}
SEEDS = tuple(range(10))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def dataset_manifest(dataset: str, output_dir: Path) -> dict:
    coverages: list[float] = []
    shape: dict | None = None
    for seed in SEEDS:
        path = output_dir / f"{dataset}_seed{seed}.json"
        if not path.exists():
            raise RuntimeError(f"missing required source checkpoint: {path}")
        record = json.loads(path.read_text())
        if record.get("dataset") != dataset or record.get("seed") != seed:
            raise RuntimeError(f"identity mismatch: {path}")
        if record.get("source_protocol") != PROTOCOL:
            raise RuntimeError(f"protocol mismatch: {path}")
        if not isinstance(record.get("samples"), dict) or len(record["samples"]) != 10:
            raise RuntimeError(f"incomplete test-size grid: {path}")
        current_shape = record.get("shape")
        if not isinstance(current_shape, dict) or not all(
            isinstance(current_shape.get(key), int) and current_shape[key] > 0
            for key in ("train", "calibration", "test")
        ):
            raise RuntimeError(f"invalid split shape: {path}")
        if shape is None:
            shape = current_shape
        elif shape != current_shape:
            raise RuntimeError(f"split shape differs between seeds: {path}")
        integrity = record.get("coverage_integrity")
        if not isinstance(integrity, dict) or integrity.get("count") != current_shape["test"]:
            raise RuntimeError(f"missing or invalid persisted coverage integrity: {path}")
        covered = integrity.get("covered")
        if not isinstance(covered, int) or not 0 <= covered <= integrity["count"] or not valid_sha256(integrity.get("sha256")):
            raise RuntimeError(f"invalid coverage total or SHA-256: {path}")
        coverage = record.get("average_test_coverage")
        if not isinstance(coverage, (int, float)) or not math.isfinite(coverage):
            raise RuntimeError(f"invalid coverage: {path}")
        if not math.isclose(float(coverage), covered / integrity["count"], rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(f"coverage/integrity disagreement: {path}")
        coverages.append(float(coverage))

    summary_path = output_dir / f"{dataset}_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"missing required strict aggregate: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("dataset") != dataset or summary.get("seed_count") != len(SEEDS):
        raise RuntimeError(f"invalid strict aggregate identity: {summary_path}")
    aggregate = summary.get("average_test_coverage", {})
    if not math.isclose(aggregate.get("mean", float("nan")), mean(coverages), rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(f"strict aggregate coverage mismatch: {summary_path}")
    return {
        "shape": shape,
        "seed_count": len(SEEDS),
        "average_test_coverage": {
            "per_seed": coverages,
            "mean": mean(coverages),
            "std": sample_std(coverages),
            "sem": sample_std(coverages) / math.sqrt(len(coverages)),
        },
    }


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    datasets = {dataset: dataset_manifest(dataset, args.output_dir) for dataset in DATASETS}
    atomic_json(
        args.result,
        {
            "dataset_count": len(DATASETS),
            "datasets": datasets,
            "protocol": PROTOCOL,
            "scope": "all eight Appendix-H datasets; source-protocol reproduction evidence, not a jury claim",
        },
    )
    print(args.result)


if __name__ == "__main__":
    main()
