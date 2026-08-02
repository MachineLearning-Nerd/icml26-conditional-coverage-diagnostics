#!/usr/bin/env python3
"""Independently audit the recorded Claim 6 cross-fitting evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "raw" / "claim6_algorithm1__raw.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "claim6_independent_audit.json"
SIZES = ("2000", "10000", "50000")
FOLDS = (2, 3, 5, 10)
METRICS = ("ERT_L1_miscoverage", "ERT_brier_score", "ERT_logloss")
TOLERANCE = 1e-12


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def close(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE)


def summary(values: list[float]) -> dict[str, float | int]:
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    sem = std / math.sqrt(len(values))
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem,
        "n": len(values),
    }


def audit_recorded_summaries(evidence: dict) -> int:
    seeds = [str(seed) for seed in evidence["protocol"]["seeds"]]
    comparisons = 0
    for size in SIZES:
        for arm in ("no_cv", *(f"kfold_{fold}" for fold in FOLDS)):
            for metric in METRICS:
                values = [evidence["per_seed"][seed][size][arm][metric] for seed in seeds]
                expected = summary(values)
                recorded = evidence["summary"][size][arm][metric]
                for field, expected_value in expected.items():
                    observed_value = recorded[field]
                    require(
                        observed_value == expected_value
                        if field == "n"
                        else close(observed_value, expected_value),
                        f"summary mismatch at {size}/{arm}/{metric}/{field}",
                    )
                    comparisons += 1
    return comparisons


def contract_inputs(evidence: dict) -> dict:
    return {
        "partition_matches": evidence["partition_audit"]["matches_independent_kfold"],
        "partition_complete": evidence["partition_audit"]["test_folds_disjoint_and_complete"],
        "constant_target": evidence["partition_audit"]["constant_target_cross_validated_ert"],
        "no_cv": {
            size: evidence["summary"][size]["no_cv"]["ERT_L1_miscoverage"]["mean"]
            for size in SIZES
        },
        "cross_fitted": {
            size: {
                str(fold): evidence["summary"][size][f"kfold_{fold}"]["ERT_L1_miscoverage"]["mean"]
                for fold in FOLDS
            }
            for size in SIZES
        },
    }


def require_contract(inputs: dict) -> None:
    require(inputs["partition_matches"] is True, "recorded folds do not match independent KFold")
    require(inputs["partition_complete"] is True, "test folds are not disjoint and complete")
    require(
        all(close(value, 0.0) for value in inputs["constant_target"].values()),
        "constant-target partition control is nonzero",
    )
    for size in SIZES:
        no_cv = inputs["no_cv"][size]
        require(no_cv > 0.02, f"no-CV negative control did not inflate at n={size}")
        for fold in FOLDS:
            cross_fitted = inputs["cross_fitted"][size][str(fold)]
            require(abs(cross_fitted) < 0.01, f"cross-fitted ERT too large at n={size}, k={fold}")
            require(
                abs(cross_fitted) * 5 < abs(no_cv),
                f"cross-fitting did not reduce ERT fivefold at n={size}, k={fold}",
            )


def negative_controls(inputs: dict) -> list[dict[str, str]]:
    mutations = []

    broken_partition = copy.deepcopy(inputs)
    broken_partition["partition_matches"] = False
    mutations.append(("reject_nonmatching_partition", broken_partition))

    missing_inflation = copy.deepcopy(inputs)
    missing_inflation["no_cv"]["50000"] = 0.0
    mutations.append(("reject_missing_no_cv_inflation", missing_inflation))

    failed_cross_fit = copy.deepcopy(inputs)
    failed_cross_fit["cross_fitted"]["2000"]["5"] = 0.02
    mutations.append(("reject_failed_cross_fitting", failed_cross_fit))

    results = []
    for name, mutation in mutations:
        try:
            require_contract(mutation)
        except AuditError:
            results.append({"control": name, "result": "PASS"})
        else:
            raise AuditError(f"negative control was accepted: {name}")
    return results


def audit(input_path: Path) -> dict:
    raw = input_path.read_bytes()
    evidence = json.loads(raw)
    protocol = evidence["protocol"]
    require(evidence["claim"] == "6", "wrong claim identifier")
    require(protocol["method"] == "CheapBetterLGBMClassifier", "wrong classifier")
    require(protocol["seeds"] == [0, 1, 2, 3, 4], "wrong seeds")
    require(protocol["sizes"] == [2000, 10000, 50000], "wrong sample sizes")
    require(protocol["fold_counts"] == [2, 3, 5, 10], "wrong fold sweep")
    require(protocol["construction"] == "oracle (true ERT is exactly 0)", "wrong construction")
    require(evidence["partition_audit"]["n_splits_observed"] == 5, "wrong observed fold count")

    comparison_count = audit_recorded_summaries(evidence)
    inputs = contract_inputs(evidence)
    require_contract(inputs)
    controls = negative_controls(inputs)

    return {
        "schema_version": 1,
        "claim": 6,
        "verdict": "VERIFIED",
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "summary_values_recomputed": comparison_count,
        "protocol": {
            "construction": protocol["construction"],
            "method": protocol["method"],
            "seeds": protocol["seeds"],
            "sizes": protocol["sizes"],
            "fold_counts": protocol["fold_counts"],
        },
        "partition_checks": {
            "matches_independent_kfold": inputs["partition_matches"],
            "test_folds_disjoint_and_complete": inputs["partition_complete"],
            "constant_target_all_zero": True,
        },
        "l1_ert_means": {
            size: {
                "no_cv": inputs["no_cv"][size],
                **{f"kfold_{fold}": inputs["cross_fitted"][size][str(fold)] for fold in FOLDS},
            }
            for size in SIZES
        },
        "negative_controls": controls,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        report = audit(args.input)
    except (AuditError, KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"CLAIM 6 AUDIT FAILED: {error}")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("CLAIM 6 VERIFIED")
    print(f"input_sha256={report['input_sha256']}")
    print(f"summary_values_recomputed={report['summary_values_recomputed']}")
    print("negative_controls=3/3 rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
