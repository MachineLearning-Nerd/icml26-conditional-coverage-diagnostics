#!/usr/bin/env python3
"""Aggregate the released CPU LightGBM-versus-PartitionWise comparison.

The pinned-covmetrics CPU replay runs the two released CPU comparator classes
with the direct L1 loss. This utility requires every source-scale checkpoint
and reports the direct qualitative comparison only; it does not mislabel this
two-method result as the paper's GPU-inclusive Table-2 percentage-of-maximum
statistic.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile


DATASETS = (
    "ailerons", "diamonds", "winequality", "miami2016",
    "o11", "superconductivity", "deliverytime", "protein",
)
SEEDS = tuple(range(10))
METHODS = ("CheapBetterLGBMClassifier", "PartitionWise")
METRIC = "ERT_L1_miscoverage"
PROTOCOL = {
    "split": "40/10/50", "alpha": 0.1, "ert_folds": 5, "device": "cpu",
    "ert_backend": "pinned covmetrics L1_miscoverage",
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sem(values: list[float]) -> float:
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1) / len(values))


def finite(value: object, label: str) -> float:
    if not isinstance(value, (float, int)) or not math.isfinite(value):
        raise RuntimeError(f"invalid {label}")
    return float(value)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_dataset(dataset: str, output_dir: Path) -> dict[str, list[float]]:
    records = []
    for seed in SEEDS:
        path = output_dir / f"{dataset}_seed{seed}.json"
        if not path.exists():
            raise RuntimeError(f"missing source checkpoint: {path}")
        record = json.loads(path.read_text())
        if record.get("dataset") != dataset or record.get("seed") != seed or record.get("source_protocol") != PROTOCOL:
            raise RuntimeError(f"identity/protocol mismatch: {path}")
        if not isinstance(record.get("samples"), dict) or len(record["samples"]) != 10:
            raise RuntimeError(f"incomplete test-size grid: {path}")
        records.append(record)
    sizes = sorted(records[0]["samples"], key=int)
    values = {method: [] for method in METHODS}
    for record in records:
        if sorted(record["samples"], key=int) != sizes:
            raise RuntimeError(f"test-size grid mismatch for {dataset}")
        for size in sizes:
            for method in METHODS:
                metrics = record["samples"][size].get(method)
                if not isinstance(metrics, dict):
                    raise RuntimeError(f"missing {method}: {dataset} seed {record['seed']} n={size}")
                values[method].append(finite(metrics.get(METRIC), f"{dataset}/{method}/{size}"))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    per_dataset = {}
    overall = {method: [] for method in METHODS}
    paired_wins = {"lgbm": 0, "partitionwise": 0, "ties": 0}
    for dataset in DATASETS:
        values = load_dataset(dataset, args.output_dir)
        if len(values[METHODS[0]]) != 100 or len(values[METHODS[1]]) != 100:
            raise RuntimeError(f"unexpected comparison cell count for {dataset}")
        per_dataset[dataset] = {
            method: {"mean": mean(method_values), "sem": sem(method_values), "cell_count": len(method_values)}
            for method, method_values in values.items()
        }
        overall[METHODS[0]].extend(values[METHODS[0]])
        overall[METHODS[1]].extend(values[METHODS[1]])
        for lightgbm, partitionwise in zip(values[METHODS[0]], values[METHODS[1]]):
            if lightgbm > partitionwise:
                paired_wins["lgbm"] += 1
            elif partitionwise > lightgbm:
                paired_wins["partitionwise"] += 1
            else:
                paired_wins["ties"] += 1
    atomic_json(args.result, {
        "scope": "eight source-scale Appendix-H datasets, ten seeds, ten test sizes, two released CPU comparator blocks; not a GPU-inclusive Table-2 percentage replication",
        "datasets": list(DATASETS),
        "seed_count": len(SEEDS),
        "test_sizes_per_seed": 10,
        "metric": METRIC,
        "methods": list(METHODS),
        "per_dataset": per_dataset,
        "overall": {
            method: {"mean": mean(values), "sem": sem(values), "cell_count": len(values)}
            for method, values in overall.items()
        },
        "paired_cells": paired_wins,
    })
    print(args.result)


if __name__ == "__main__":
    main()
