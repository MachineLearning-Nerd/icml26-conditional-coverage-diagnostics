#!/usr/bin/env python3
"""Full-scale pinned-covmetrics L1-ERT CPU comparator replay.

The released legacy ERT module writes a degenerate zero-valued L1 field for
the two CPU comparator blocks.  This runner preserves the released data split,
RealMLP conformal construction, test-size grid, five folds, and CPU execution,
but evaluates the L1 loss through the pinned ``covmetrics`` package directly.
It is intentionally separate from the legacy Appendix-H integrity outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from sklearn.model_selection import train_test_split

from prepare_source_data import SPECS
from run_full_repaired_cpu import prepare_frame
from repaired_cpu_ert import CheapBetterLGBMClassifier, PartitionWisePredictor


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "upstream" / "covmetrics" / "src"
if str(METRICS) not in sys.path:
    sys.path.insert(0, str(METRICS))
from covmetrics.ERT import ERT as CovmetricsERT  # noqa: E402
from covmetrics.losses import L1_miscoverage  # noqa: E402

SOURCE_CODE = ROOT / "upstream" / "Conditional_Coverage_Estimation" / "experiments" / "experiments_classifier_benchmark" / "code"
if str(SOURCE_CODE) not in sys.path:
    sys.path.insert(0, str(SOURCE_CODE))
from conditional_coverage_metrics import seed_everything  # noqa: E402
from conformalizer import Conformalizer, lp_y_f_x  # noqa: E402
from pytabkit import RealMLP_TD_S_Regressor  # noqa: E402


PROTOCOL = {
    "split": "40/10/50",
    "alpha": 0.1,
    "ert_folds": 5,
    "device": "cpu",
    "ert_backend": "pinned covmetrics L1_miscoverage",
}
METHODS = {
    "CheapBetterLGBMClassifier": CheapBetterLGBMClassifier,
    "PartitionWise": PartitionWisePredictor,
}


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def evaluate_subset(features, coverage: np.ndarray) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, model_cls in METHODS.items():
        kwargs = {}
        if name == "PartitionWise":
            kwargs["n_clusters"] = max(1, int(len(coverage) ** 0.25))
        values = CovmetricsERT(model_cls, **kwargs).evaluate_multiple_losses(
            features, coverage, 0.1, n_splits=5, all_losses=[L1_miscoverage]
        )
        value = values.get("ERT_L1_miscoverage")
        if not isinstance(value, (int, float, np.floating)) or not np.isfinite(value):
            raise RuntimeError(f"invalid pinned L1 ERT for {name}: {values}")
        result[name] = {"ERT_L1_miscoverage": float(value)}
    return result


def run_one(dataset: str, seed: int, raw_dir: Path, output_dir: Path) -> Path:
    path = output_dir / f"{dataset}_seed{seed}.json"
    seed_everything(seed)
    frame = prepare_frame(dataset, raw_dir)
    features, target = frame.drop(columns=["target"]), frame["target"]
    x_train, x_temp, y_train, y_temp = train_test_split(features, target, test_size=0.6, random_state=seed)
    x_calibration, x_test, y_calibration, y_test = train_test_split(x_temp, y_temp, test_size=5 / 6, random_state=seed)
    model = RealMLP_TD_S_Regressor(device="cpu", n_cv=5, val_metric_name="brier", verbosity=False)
    model.fit(x_train, y_train)
    conformalizer = Conformalizer(get_scores=lp_y_f_x(model))
    conformalizer.conformalize(x_calibration, y_calibration, alpha=0.1)
    coverage = np.asarray(conformalizer.get_cover(x_test, y_test), dtype=int)
    expected = {
        "dataset": dataset,
        "seed": seed,
        "source_protocol": PROTOCOL,
        "shape": {"train": len(x_train), "calibration": len(x_calibration), "test": len(x_test)},
        "average_test_coverage": float(coverage.mean()),
        "coverage_integrity": {
            "count": int(coverage.size),
            "covered": int(coverage.sum()),
            "sha256": hashlib.sha256(coverage.tobytes()).hexdigest(),
        },
        "samples": {},
    }
    if path.exists():
        result = json.loads(path.read_text())
        for key in ("dataset", "seed", "source_protocol", "shape", "average_test_coverage", "coverage_integrity"):
            if result.get(key) != expected[key]:
                raise RuntimeError(f"incompatible checkpoint {path}: {key}")
        if not isinstance(result.get("samples"), dict):
            raise RuntimeError(f"invalid samples checkpoint: {path}")
    else:
        result = expected
    sizes = np.round(np.logspace(np.log10(300), np.log10(len(x_test)), 10)).astype(int)
    for size in sizes:
        indices = np.random.choice(len(x_test), size=int(size), replace=False)
        key = str(int(size))
        if key in result["samples"]:
            continue
        result["samples"][key] = evaluate_subset(x_test.iloc[indices], coverage[indices])
        write_json_atomic(path, result)
        print(f"checkpointed pinned-L1 {dataset} seed={seed} n={size}", flush=True)
    return path


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
