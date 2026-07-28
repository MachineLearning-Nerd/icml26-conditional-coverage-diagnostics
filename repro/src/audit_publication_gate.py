#!/usr/bin/env python3
"""Fail closed on the evidence required before this paper may be published.

This is deliberately an evidence gate, not a scorer.  It counts six separate
source-anchored findings only when their underlying full outputs are present:
three deterministic ERT foundations and three observations from the released
heteroscedastic synthetic experiment.  The Appendix-H manifest is required as
protocol evidence but is not counted as a jury claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile


EPS = 1e-12
REQUIRED_DATASETS = {
    "ailerons", "diamonds", "winequality", "miami2016",
    "o11", "superconductivity", "deliverytime", "protein",
}


def load(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"missing required evidence: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError(f"invalid finite number for {label}")
    return float(value)


def foundation_claims(foundations: dict) -> list[dict]:
    claims = foundations.get("claims")
    if not isinstance(claims, dict):
        raise RuntimeError("foundation evidence has no claims object")
    constant = claims.get("1_constant_target_principle", {})
    target = constant.get("target_ert", {})
    candidate = constant.get("candidate_ert", {})
    target_values = {name: finite(target.get(name), f"target {name}") for name in ("L1", "L2", "KL")}
    candidate_values = {name: finite(candidate.get(name), f"candidate {name}") for name in ("L1", "L2", "KL")}
    if any(abs(value) > EPS for value in target_values.values()) or any(value > EPS for value in candidate_values.values()):
        raise RuntimeError("constant-target foundation does not establish the ERT principle")

    decomposition = claims.get("4_asymmetric_decomposition", {})
    conservative, aggressive = decomposition.get("conservative", {}), decomposition.get("aggressive", {})
    over_conservative = finite(conservative.get("over"), "conservative over")
    under_conservative = finite(conservative.get("under"), "conservative under")
    over_aggressive = finite(aggressive.get("over"), "aggressive over")
    under_aggressive = finite(aggressive.get("under"), "aggressive under")
    if not (over_conservative > EPS and abs(under_conservative) <= EPS and abs(over_aggressive) <= EPS and under_aggressive > EPS):
        raise RuntimeError("asymmetric decomposition foundation is not isolated")

    cross_validation = claims.get("6_kfold_cross_validation", {})
    folds = cross_validation.get("folds")
    cv_l1 = finite(cross_validation.get("cross_validated_l1_ert"), "cross-validated L1 ERT")
    if folds != 5 or abs(cv_l1) > EPS:
        raise RuntimeError("five-fold cross-validation foundation is invalid")

    return [
        {"id": "1", "claim": "constant target is not beaten under exact conditional coverage", "evidence": target_values | {"candidate": candidate_values}},
        {"id": "4", "claim": "asymmetric ERT separates over- and under-coverage", "evidence": {"conservative": conservative, "aggressive": aggressive}},
        {"id": "6", "claim": "Algorithm-1-style five-fold held-out ERT evaluation", "evidence": cross_validation},
    ]


def require_appendix_manifest(manifest: dict) -> None:
    if manifest.get("dataset_count") != 8:
        raise RuntimeError("Appendix-H manifest does not cover all eight datasets")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != REQUIRED_DATASETS:
        raise RuntimeError("Appendix-H manifest dataset identity mismatch")


def synthetic_claims(summary: dict) -> list[dict]:
    if summary.get("seed_count") != 10:
        raise RuntimeError("synthetic summary is not a ten-seed aggregate")
    methods = summary.get("methods")
    if not isinstance(methods, dict) or set(methods) != {"MSE", "HR"}:
        raise RuntimeError("synthetic summary method identity mismatch")
    try:
        mse, hr = methods["MSE"], methods["HR"]
        mse_truth = finite(mse["true_l1_miscoverage"]["mean"], "MSE truth")
        hr_truth = finite(hr["true_l1_miscoverage"]["mean"], "HR truth")
        mse_sizes, hr_sizes = mse["sizes"], hr["sizes"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("synthetic summary shape mismatch") from exc
    common = set(mse_sizes) & set(hr_sizes)
    if not common:
        raise RuntimeError("synthetic summary has no common released sizes")
    size = min(common, key=lambda item: abs(int(item) - 5_000))

    def values(method: dict, truth: float, label: str) -> dict[str, float]:
        entry = method["sizes"][size]
        ert = finite(entry["l1_ert"]["mean"], f"{label} L1-ERT")
        covgap = finite(entry["covgap"]["mean"], f"{label} CovGap")
        return {"truth": truth, "l1_ert": ert, "covgap": covgap, "l1_ert_error": abs(ert - truth), "covgap_error": abs(covgap - truth)}

    mse_values, hr_values = values(mse, mse_truth, "MSE"), values(hr, hr_truth, "HR")
    if not (mse_values["l1_ert"] > 0.0 and mse_values["l1_ert_error"] < mse_values["covgap_error"]):
        raise RuntimeError("synthetic standard-CP L1-ERT does not improve on CovGap")
    if not (hr_values["l1_ert_error"] < hr_values["covgap_error"]):
        raise RuntimeError("synthetic oracle L1-ERT does not improve on CovGap")
    ert_separation = abs(mse_values["l1_ert"] - hr_values["l1_ert"])
    covgap_separation = abs(mse_values["covgap"] - hr_values["covgap"])
    if not ert_separation > covgap_separation:
        raise RuntimeError("synthetic L1-ERT does not separate the two constructions more than CovGap")

    return [
        {"id": "3a", "claim": "at roughly 5,000 test points L1-ERT detects standard-CP conditional error more accurately than CovGap", "evidence": {"test_size": int(size), "MSE": mse_values}},
        {"id": "3b", "claim": "at roughly 5,000 test points L1-ERT recognizes the oracle conditional construction more accurately than CovGap", "evidence": {"test_size": int(size), "HR": hr_values}},
        {"id": "3c", "claim": "at roughly 5,000 test points L1-ERT separates standard and oracle constructions more strongly than CovGap", "evidence": {"test_size": int(size), "l1_ert_separation": ert_separation, "covgap_separation": covgap_separation}},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundations", type=Path, required=True)
    parser.add_argument("--appendix-manifest", type=Path, required=True)
    parser.add_argument("--synthetic-summary", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    foundations = load(args.foundations)
    appendix_manifest = load(args.appendix_manifest)
    synthetic_summary = load(args.synthetic_summary)
    require_appendix_manifest(appendix_manifest)
    claims = foundation_claims(foundations) + synthetic_claims(synthetic_summary)
    if len(claims) < 5:
        raise RuntimeError("fewer than five independently evidenced claims")
    atomic_json(args.result, {
        "publication_eligible": True,
        "claim_count": len(claims),
        "claims": claims,
        "protocol_evidence": {"appendix_h_dataset_count": appendix_manifest["dataset_count"], "synthetic_seed_count": synthetic_summary["seed_count"]},
        "scope": "strict local publication gate; GitHub/HF publication remains a separate authorized action",
    })
    print(args.result)


if __name__ == "__main__":
    main()
