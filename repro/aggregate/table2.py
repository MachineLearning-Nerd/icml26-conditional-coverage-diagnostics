"""Compute the Table-2 percentage statistic from collected rows.

The normalisation is taken jointly over methods and test sizes within each
(dataset, experiment), so it can only be computed once every dataset shard has
landed — which is why this is a separate step rather than part of a run.

    uv run --frozen python -m repro.aggregate.table2

It also re-applies the identical function to the release's own committed
`results.csv`.  That reproduces the authors' published row and is how the
statistic's definition is pinned down; it is calibration of the code, using the
authors' numbers, and is reported separately from the regenerated result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from ..emit_paths import ARTIFACTS
from ..pipeline.stage_table2 import PAPER_TABLE2_L1, table2_statistic

RELEASE_CSV = Path("repro/reference/release_classifier_benchmark_results.csv")
# Table 5's four classifier-comparison datasets, under the release CSV's names.
RELEASE_NAMES = {"physiochemical_protein": "CASP1D", "Food_Delivery_Time": "deliverytime",
                 "diamonds": "diamonds", "superconductivity": "superconductivity"}
RELEASE_METHOD_NAMES = {"CheapBetterLGBMClassifier": "CheapBetterLGBMClassifier",
                        "BetterCatBoost": "BetterCatBoost", "RF": "RF", "XT": "XT",
                        "PartitionWise": "PartitionWise", "TabPFN": "TabPFN", "tabICL": "tabICL"}


def calibrate_against_release() -> dict:
    """Apply this repository's statistic to the authors' own output."""
    frame = pd.read_csv(RELEASE_CSV)
    frame = frame[frame["dataset"].isin(RELEASE_NAMES.values())]
    stat = table2_statistic(frame)
    column = ("pct_ERT_L1_miscoverage", "mean")
    observed = {method: float(stat.loc[method, column]) for method in stat.index}
    deviations = {}
    for method, entry in PAPER_TABLE2_L1.items():
        release_name = RELEASE_METHOD_NAMES[method]
        if release_name in observed:
            deviations[method] = round(observed[release_name] - entry["mean"], 3)
    return {
        "source": str(RELEASE_CSV),
        "note": "the authors' own committed benchmark output, used only to pin down the "
                "definition of the Table-2 percentage; not evidence for the claim",
        "recomputed_percentages": {k: round(v, 2) for k, v in observed.items()},
        "paper_percentages": {k: v["mean"] for k, v in PAPER_TABLE2_L1.items()},
        "deviation_vs_paper": deviations,
        "max_abs_deviation_vs_release_csv": max(abs(v) for v in deviations.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=ARTIFACTS / "rows/table2.jsonl")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "claim2_table2/statistic.json")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame.from_records(records)
    stat = table2_statistic(frame)

    percentages = {}
    for method in stat.index:
        percentages[method] = {
            "mean": round(float(stat.loc[method, ("pct_ERT_L1_miscoverage", "mean")]), 3),
            "std": round(float(stat.loc[method, ("pct_ERT_L1_miscoverage", "std")]), 3),
            "l2_mean": round(float(stat.loc[method, ("pct_ERT_brier_score", "mean")]), 3),
            "kl_mean": round(float(stat.loc[method, ("pct_ERT_logloss", "mean")]), 3),
            "paper_l1": PAPER_TABLE2_L1.get(method, {}).get("mean"),
            "paper_label": PAPER_TABLE2_L1.get(method, {}).get("label"),
        }

    sizes_per_dataset = int(frame.groupby("dataset")["nsamples"].nunique().max())
    result = {
        "claim": "2",
        "datasets": sorted(frame["dataset"].unique().tolist()),
        "methods": sorted(frame["method"].unique().tolist()),
        "n_experiments": int(frame["experiment"].nunique()),
        "sizes_per_dataset": sizes_per_dataset,
        "cells": int(len(frame)),
        "statistic": "see_pourcentage_improvment.ipynb: clip negatives to 0, normalise by the "
                     "max over methods and sizes within each (dataset, experiment), average "
                     "over datasets and sizes per (method, experiment), then mean and std "
                     "over the ten experiments",
        "percentages": percentages,
        "calibration": calibrate_against_release(),
        "mean_time_per_1k_samples_s": {
            method: round(float((group["time"] * 1000 / group["nsamples"]).mean()), 3)
            for method, group in frame.groupby("method")
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["percentages"], indent=2, sort_keys=True))
    print(json.dumps({k: result[k] for k in ("datasets", "methods", "n_experiments",
                                             "sizes_per_dataset", "cells")}, indent=2))
    print(json.dumps(result["calibration"]["recomputed_percentages"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
