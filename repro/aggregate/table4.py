"""Summarise the Table-4 classification cells across seeds.

    uv run --frozen python -m repro.aggregate.table4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..emit_paths import ARTIFACTS
from ..pipeline.metrics import mean_and_sem
from ..pipeline.stage_table4 import PAPER_TABLE4

FIELDS = ("ERT_L1_miscoverage", "ERT_logloss", "ERT_logloss_over", "ERT_logloss_under",
          "ERT_brier_score", "empty_set_rate", "mean_set_size", "marginal_coverage",
          "predictor_accuracy")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=ARTIFACTS / "rows/table4.jsonl")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "claim5_table4/summary.json")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(f"{record['dataset']}|{record['method']}", []).append(record)

    observed, residuals = {}, []
    for key, cells in sorted(grouped.items()):
        observed[key] = {field: mean_and_sem([c[field] for c in cells]) for field in FIELDS}
        observed[key]["seeds"] = sorted(c["experiment"] for c in cells)
        for cell in cells:
            residuals.append(abs(cell["ERT_logloss"] - cell["ERT_logloss_over"]
                                 - cell["ERT_logloss_under"]))
            residuals.append(abs(cell["ERT_L1_miscoverage"] - cell["ERT_L1_miscoverage_over"]
                                 - cell["ERT_L1_miscoverage_under"]))

    result = {
        "claim": "5",
        "kl_plus_field": "ERT_logloss_over",
        "kl_minus_field": "ERT_logloss_under",
        "field_mapping_note": "the release's ERT_underconfident_* clips predictions from below "
                              "at 1-alpha and ERT_overconfident_* from above, which are "
                              "covmetrics' *_over and *_under losses; Table 4's KL+ is therefore "
                              "ERT_logloss_over and KL- is ERT_logloss_under",
        "cells": len(records),
        "max_abs_additivity_residual": max(residuals) if residuals else 0.0,
        "observed": observed,
        "paper_table4": {f"{k[0]}|{k[1]}": v for k, v in PAPER_TABLE4.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(f"{'cell':<28}{'L1-ERT':>10}{'KL-ERT':>10}{'KL+':>10}{'KL-':>10}{'empty':>9}")
    for key, cell in observed.items():
        print(f"{key:<28}{cell['ERT_L1_miscoverage']['mean']:>+10.4f}"
              f"{cell['ERT_logloss']['mean']:>+10.4f}{cell['ERT_logloss_over']['mean']:>+10.4f}"
              f"{cell['ERT_logloss_under']['mean']:>+10.4f}{cell['empty_set_rate']['mean']:>9.4f}")
    print(f"max additivity residual {result['max_abs_additivity_residual']:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
