#!/usr/bin/env python3
"""Strictly aggregate completed source-protocol CPU checkpoints.

This utility deliberately refuses partial seed sets.  Its output is evidence
for audit and comparison only; claim verification additionally requires every
Appendix-H dataset plus independent checks.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile


EXPECTED_SEEDS = tuple(range(10))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def load_completed(dataset: str, output_dir: Path) -> list[dict]:
    records = []
    for seed in EXPECTED_SEEDS:
        path = output_dir / f"{dataset}_seed{seed}.json"
        if not path.exists():
            raise RuntimeError(f"missing required seed checkpoint: {path}")
        record = json.loads(path.read_text())
        if record.get("dataset") != dataset or record.get("seed") != seed:
            raise RuntimeError(f"identity mismatch in {path}")
        if record.get("source_protocol") != {
            "split": "40/10/50", "alpha": 0.1, "ert_folds": 5, "device": "cpu"
        }:
            raise RuntimeError(f"source protocol mismatch in {path}")
        samples = record.get("samples", {})
        if len(samples) != 10:
            raise RuntimeError(f"incomplete test-size grid in {path}: {len(samples)}/10")
        coverage = record.get("average_test_coverage")
        if not isinstance(coverage, (float, int)) or not math.isfinite(coverage):
            raise RuntimeError(f"invalid coverage in {path}")
        records.append(record)
    return records


def aggregate(records: list[dict]) -> dict:
    sample_sizes = sorted(records[0]["samples"], key=int)
    if any(sorted(record["samples"], key=int) != sample_sizes for record in records):
        raise RuntimeError("test-size grids differ between seeds")
    methods = sorted(records[0]["samples"][sample_sizes[0]])
    result = {
        "seed_count": len(records),
        "seeds": [record["seed"] for record in records],
        "average_test_coverage": {
            "per_seed": [record["average_test_coverage"] for record in records],
        },
        "ert": {},
    }
    coverages = result["average_test_coverage"]["per_seed"]
    result["average_test_coverage"].update(
        mean=mean(coverages), std=sample_std(coverages), sem=sample_std(coverages) / math.sqrt(len(coverages))
    )
    for size in sample_sizes:
        result["ert"][size] = {}
        for method in methods:
            metric_names = sorted(records[0]["samples"][size][method])
            values_by_metric = {metric: [] for metric in metric_names}
            for record in records:
                metrics = record["samples"][size].get(method)
                if metrics is None or sorted(metrics) != metric_names:
                    raise RuntimeError(f"method/metric mismatch for {method} at n={size}")
                for metric in metric_names:
                    value = metrics[metric]
                    if not isinstance(value, (float, int)) or not math.isfinite(value):
                        raise RuntimeError(f"non-finite {method}/{metric} at n={size}")
                    values_by_metric[metric].append(float(value))
            result["ert"][size][method] = {
                metric: {"mean": mean(values), "std": sample_std(values), "sem": sample_std(values) / math.sqrt(len(values))}
                for metric, values in values_by_metric.items()
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    records = load_completed(args.dataset, args.output_dir)
    summary = aggregate(records)
    summary.update(dataset=args.dataset, scope="one Appendix-H dataset; not a paper claim")
    atomic_json(args.result, summary)
    print(args.result)


if __name__ == "__main__":
    main()
