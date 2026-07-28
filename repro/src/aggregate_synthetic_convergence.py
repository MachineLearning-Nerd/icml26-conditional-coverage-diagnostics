#!/usr/bin/env python3
"""Fail-closed aggregate for the ten-seed synthetic convergence sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile

from run_synthetic_convergence import ALPHA, SIZES, SOURCE_PROTOCOL


METHODS = ("MSE", "HR")
SEEDS = tuple(range(10))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sem(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("SEM requires at least two values")
    average = mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_records(output_dir: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    for seed in SEEDS:
        path = output_dir / f"synthetic_seed{seed}.json"
        if not path.exists():
            raise RuntimeError(f"missing synthetic checkpoint: {path}")
        record = json.loads(path.read_text())
        if record.get("seed") != seed or record.get("source_protocol") != SOURCE_PROTOCOL:
            raise RuntimeError(f"identity/protocol mismatch: {path}")
        for method in METHODS:
            method_record = record.get("methods", {}).get(method)
            if not isinstance(method_record, dict):
                raise RuntimeError(f"missing {method} result: {path}")
            integrity = method_record.get("full_coverage_integrity")
            if not isinstance(integrity, dict) or integrity.get("count") != 300_000:
                raise RuntimeError(f"invalid full coverage integrity: {path} {method}")
            if not isinstance(integrity.get("covered"), int) or not isinstance(integrity.get("sha256"), str):
                raise RuntimeError(f"invalid coverage digest: {path} {method}")
            samples = method_record.get("samples")
            if not isinstance(samples, dict) or set(samples) != {str(size) for size in SIZES}:
                raise RuntimeError(f"incomplete source test-size grid: {path} {method}")
            for size in SIZES:
                metrics = samples[str(size)]
                if not isinstance(metrics, dict) or not all(
                    isinstance(metrics.get(key), (int, float)) and math.isfinite(metrics[key])
                    for key in ("l1_ert", "covgap", "groups")
                ):
                    raise RuntimeError(f"invalid metric record: {path} {method} n={size}")
        records[seed] = record
    return records


def aggregate(records: dict[int, dict]) -> dict:
    methods: dict[str, dict] = {}
    for method in METHODS:
        truth = [float(records[seed]["methods"][method]["true_l1_miscoverage"]) for seed in SEEDS]
        sizes: dict[str, dict] = {}
        for size in SIZES:
            samples = [records[seed]["methods"][method]["samples"][str(size)] for seed in SEEDS]
            l1_ert = [float(sample["l1_ert"]) for sample in samples]
            covgap = [float(sample["covgap"]) for sample in samples]
            l1_error = [abs(estimate - target) for estimate, target in zip(l1_ert, truth)]
            covgap_error = [abs(estimate - target) for estimate, target in zip(covgap, truth)]
            sizes[str(size)] = {
                "l1_ert": {"mean": mean(l1_ert), "sem": sem(l1_ert), "absolute_error_mean": mean(l1_error), "absolute_error_sem": sem(l1_error)},
                "covgap": {"mean": mean(covgap), "sem": sem(covgap), "absolute_error_mean": mean(covgap_error), "absolute_error_sem": sem(covgap_error)},
                "groups": sorted({int(sample["groups"]) for sample in samples}),
            }
        methods[method] = {
            "true_l1_miscoverage": {"mean": mean(truth), "sem": sem(truth)},
            "sizes": sizes,
        }
    return methods


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    records = load_records(args.output_dir)
    result = {
        "scope": "ten seeds, two released synthetic conformal constructions, fifteen released test sizes; descriptive evidence only",
        "alpha": ALPHA,
        "seed_count": len(SEEDS),
        "source_protocol": SOURCE_PROTOCOL,
        "methods": aggregate(records),
    }
    atomic_json(args.result, result)
    print(args.result)


if __name__ == "__main__":
    main()
