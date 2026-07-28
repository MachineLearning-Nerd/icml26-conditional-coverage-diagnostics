#!/usr/bin/env python3
"""Resume-safe synthetic convergence reproduction for anchored claim 3.

This is a disclosed release-repair wrapper around
``_generate_fig_evol.py``.  It retains that driver's 8-D heteroscedastic
generator, 30k/3k/3k/300k allocation, alpha=.1, oracle-mean and oracle-interval
conformal constructions, and fifteen released test sizes.  Unlike the released
driver it writes atomic per-seed checkpoints and evaluates the pinned modern
L1-ERT field directly, rather than labelling the legacy driver's Brier ERT
output as L1-ERT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.stats import norm
import torch


ROOT = Path(__file__).resolve().parents[2]
GENERAL = ROOT / "upstream" / "Conditional_Coverage_Estimation" / "experiments" / "experiments_general" / "code"
METRICS = ROOT / "upstream" / "covmetrics" / "src"
if str(GENERAL) not in sys.path:
    sys.path.insert(0, str(GENERAL))
if str(METRICS) not in sys.path:
    sys.path.insert(0, str(METRICS))

from conditional_coverage_metrics import CoverageGap, KMeansGrouping, seed_everything  # noqa: E402
from repaired_cpu_ert import CheapBetterLGBMClassifier  # noqa: E402
from covmetrics.ERT import ERT as CovmetricsERT  # noqa: E402
from covmetrics.losses import L1_miscoverage  # noqa: E402


ALPHA = 0.1
TARGET = 1.0 - ALPHA
SIZES = tuple(np.round(np.logspace(np.log10(500), np.log10(100_000), 15)).astype(int).tolist())
SOURCE_PROTOCOL = {
    "generator": "x1~U[-1,1], sigma=.5+abs(2*x1)+x1**2, seven U[-1,1] nuisance features",
    "allocation": {"train": 30_000, "stop": 3_000, "calibration": 3_000, "test": 300_000},
    "alpha": ALPHA,
    "test_sizes": list(SIZES),
    "ert_folds": 5,
    "ert_classifier": "source-derived CheapBetterLGBMClassifier",
    "selection": "deterministic per-method/per-size torch generator; equivalent uniform without replacement",
}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def generate(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = (torch.rand(n, 1) - 0.5) * 2
    sigma = 0.5 + torch.abs(2 * x) + x**2
    return x, sigma * torch.randn_like(x)


def synthetic_data(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replicate the source draw order, including its unused train/stop draws."""
    seed_everything(seed)
    x_train, _ = generate(30_000)
    x_stop, _ = generate(3_000)
    x_calibration, y_calibration = generate(3_000)
    x_test, y_test = generate(300_000)

    def add_noise_features(x: torch.Tensor) -> np.ndarray:
        return np.hstack([x, np.random.uniform(-1, 1, size=(len(x), 7))])

    # The first two are intentionally retained to preserve the release's numpy
    # RNG consumption before calibration/test nuisance features are generated.
    add_noise_features(x_train)
    add_noise_features(x_stop)
    return add_noise_features(x_calibration), y_calibration.numpy().reshape(-1), add_noise_features(x_test), y_test.numpy().reshape(-1)


def coverage_vectors(seed: int) -> dict[str, dict[str, np.ndarray | float]]:
    x_calibration, y_calibration, x_test, y_test = synthetic_data(seed)
    q_index = int(np.ceil((len(y_calibration) + 1) * TARGET))
    mse_radius = float(np.sort(np.abs(y_calibration))[q_index])
    sigma_test = 0.5 + np.abs(2 * x_test[:, 0]) + x_test[:, 0] ** 2
    mse_cover = (np.abs(y_test) < mse_radius).astype(int)
    mse_conditional = 2.0 * norm.cdf(mse_radius / sigma_test) - 1.0
    lower, upper = norm.ppf(ALPHA / 2), norm.ppf(1 - ALPHA / 2)
    standardized = y_test / sigma_test
    hr_cover = ((standardized > lower) & (standardized < upper)).astype(int)
    return {
        "MSE": {
            "features": x_test,
            "cover": mse_cover,
            "truth_l1": float(np.mean(np.abs(TARGET - mse_conditional))),
            "radius": mse_radius,
        },
        "HR": {
            "features": x_test,
            "cover": hr_cover,
            "truth_l1": 0.0,
            "radius": 0.0,
        },
    }


def digest(cover: np.ndarray) -> dict[str, int | str]:
    return {
        "count": int(cover.size),
        "covered": int(cover.sum()),
        "sha256": hashlib.sha256(np.asarray(cover, dtype=np.int8).tobytes()).hexdigest(),
    }


def evaluate_subset(features: np.ndarray, cover: np.ndarray) -> dict[str, float]:
    n_groups = max(1, int(len(cover) ** 0.25))
    feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    cover_tensor = torch.as_tensor(cover, dtype=torch.int64)
    groups = KMeansGrouping()
    groups.fit(feature_tensor, n_groups=n_groups)
    covgap = CoverageGap(alpha=ALPHA).evaluate(cover_tensor, groups(feature_tensor))
    ert = CovmetricsERT(CheapBetterLGBMClassifier).evaluate_multiple_losses(
        features, cover, ALPHA, n_splits=5, all_losses=[L1_miscoverage]
    )
    l1_ert = ert.get("ERT_L1_miscoverage")
    if not isinstance(l1_ert, (int, float)) or not np.isfinite(l1_ert):
        raise RuntimeError(f"invalid source L1-ERT result: {ert}")
    return {"l1_ert": float(l1_ert), "covgap": float(covgap), "groups": n_groups}


def stable_indices(seed: int, method_index: int, size_index: int, n: int, size: int) -> np.ndarray:
    # Source draws are uniform without replacement.  A dedicated generator makes
    # the same design resume-safe even when a prior metric call was interrupted.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 100_003 + method_index * 1_009 + size_index)
    return torch.randperm(n, generator=generator)[:size].numpy()


def expected_record(seed: int) -> dict:
    vectors = coverage_vectors(seed)
    return {
        "seed": seed,
        "source_protocol": SOURCE_PROTOCOL,
        "methods": {
            name: {
                "full_coverage_integrity": digest(value["cover"]),
                "true_l1_miscoverage": value["truth_l1"],
                "conformal_radius": value["radius"],
                "samples": {},
            }
            for name, value in vectors.items()
        },
    }


def run_one(seed: int, output_dir: Path) -> Path:
    path = output_dir / f"synthetic_seed{seed}.json"
    vectors = coverage_vectors(seed)
    expected = expected_record(seed)
    if path.exists():
        record = json.loads(path.read_text())
        for key in ("seed", "source_protocol"):
            if record.get(key) != expected[key]:
                raise RuntimeError(f"incompatible checkpoint {path}: {key}")
        for method in vectors:
            for key in ("full_coverage_integrity", "true_l1_miscoverage", "conformal_radius"):
                if record.get("methods", {}).get(method, {}).get(key) != expected["methods"][method][key]:
                    raise RuntimeError(f"incompatible checkpoint {path}: {method}.{key}")
    else:
        record = expected
    for method_index, (method, value) in enumerate(vectors.items()):
        for size_index, size in enumerate(SIZES):
            key = str(size)
            if key in record["methods"][method]["samples"]:
                continue
            indices = stable_indices(seed, method_index, size_index, len(value["cover"]), size)
            metrics = evaluate_subset(value["features"][indices], value["cover"][indices])
            record["methods"][method]["samples"][key] = metrics
            atomic_json(path, record)
            print(f"checkpointed synthetic seed={seed} method={method} n={size}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, choices=range(10))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run_one(args.seed, args.output_dir))


if __name__ == "__main__":
    main()
