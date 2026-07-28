#!/usr/bin/env python3
"""Resume-safe full-scale CPU execution of the released classifier protocol.

This is a disclosed release-repair runner: source data preparation is pinned in
``prepare_source_data.py``; the CPU ERT blocks are in ``repaired_cpu_ert.py``;
all other split, model, calibration and test-size choices follow the released
``_generate_simultaneous_experiments.py`` driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from prepare_source_data import SPECS, load_frame
from repaired_cpu_ert import evaluate_cpu_methods


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CODE = ROOT / "upstream" / "Conditional_Coverage_Estimation" / "experiments" / "experiments_classifier_benchmark" / "code"
sys.path.insert(0, str(SOURCE_CODE))
from conditional_coverage_metrics import seed_everything  # noqa: E402
from conformalizer import Conformalizer, lp_y_f_x  # noqa: E402
from pytabkit import RealMLP_TD_S_Regressor  # noqa: E402


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def prepare_frame(dataset: str, raw_dir: Path) -> pd.DataFrame:
    spec = SPECS[dataset]
    frame = load_frame(spec, raw_dir)
    target = spec["target"]
    features = frame.drop(columns=[target])
    if (len(frame), len(features.columns)) != (spec["rows"], spec["features"]):
        raise RuntimeError(f"{dataset} violates pinned Appendix-H shape")
    categorical = features.select_dtypes(include="object").columns
    if len(categorical):
        features[categorical] = OrdinalEncoder(dtype=float).fit_transform(features[categorical])
    return pd.concat([features, frame[[target]].rename(columns={target: "target"})], axis=1)


def run_one(dataset: str, seed: int, raw_dir: Path, output_dir: Path) -> Path:
    result_path = output_dir / f"{dataset}_seed{seed}.json"

    seed_everything(seed)
    frame = prepare_frame(dataset, raw_dir)
    features = frame.drop(columns=["target"])
    target = frame["target"]
    x_train, x_temp, y_train, y_temp = train_test_split(features, target, test_size=0.6, random_state=seed)
    x_calibration, x_test, y_calibration, y_test = train_test_split(
        x_temp, y_temp, test_size=5 / 6, random_state=seed
    )
    model = RealMLP_TD_S_Regressor(device="cpu", n_cv=5, val_metric_name="brier", verbosity=False)
    model.fit(x_train, y_train)
    conformalizer = Conformalizer(get_scores=lp_y_f_x(model))
    conformalizer.conformalize(x_calibration, y_calibration, alpha=0.1)
    coverage = np.asarray(conformalizer.get_cover(x_test, y_test), dtype=int)

    sizes = np.round(np.logspace(np.log10(300), np.log10(len(x_test)), 10)).astype(int)
    expected = {
        "dataset": dataset,
        "seed": seed,
        "source_protocol": {"split": "40/10/50", "alpha": 0.1, "ert_folds": 5, "device": "cpu"},
        "shape": {"train": len(x_train), "calibration": len(x_calibration), "test": len(x_test)},
        "average_test_coverage": float(coverage.mean()),
        "coverage_integrity": {
            "count": int(coverage.size),
            "covered": int(coverage.sum()),
            "sha256": hashlib.sha256(coverage.tobytes()).hexdigest(),
        },
        "samples": {},
    }
    if result_path.exists():
        result = json.loads(result_path.read_text())
        # A partial checkpoint is durable progress, not a completed run.  Only
        # reuse it after recomputing and matching the split/model coverage
        # contract, so an accidental path collision cannot mix protocols.
        for key in ("dataset", "seed", "source_protocol", "shape", "average_test_coverage", "coverage_integrity"):
            if result.get(key) != expected[key]:
                raise RuntimeError(f"incompatible checkpoint for {dataset} seed {seed}: {key}")
        if not isinstance(result.get("samples"), dict):
            raise RuntimeError(f"invalid sample checkpoint for {dataset} seed {seed}")
        print(f"resuming {dataset} seed={seed} with {len(result['samples'])} saved sizes", flush=True)
    else:
        result = expected
    for n_values in sizes:
        indices = np.random.choice(len(x_test), size=int(n_values), replace=False)
        if str(int(n_values)) in result["samples"]:
            continue
        metrics = evaluate_cpu_methods(x_test.iloc[indices], coverage[indices])
        result["samples"][str(int(n_values))] = metrics
        write_json_atomic(result_path, result)
        print(f"checkpointed {dataset} seed={seed} n={n_values}", flush=True)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(SPECS))
    parser.add_argument("seed", type=int, choices=range(10))
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run_one(args.dataset, args.seed, args.raw_dir, args.output_dir))


if __name__ == "__main__":
    main()
