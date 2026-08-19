#!/usr/bin/env python3
"""Verify the committed publication and evidence contract."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_STATUS = "PARTIAL_C1_C3_C4_LIVE_VERIFIED_C6_CPU_VERIFIED_C2_C5_BLOCKED_HISTORICAL_SCORE_6_OF_12_NO_CURRENT_SCORE"
EXPECTED_BRANCHES = {
    "audit/baseline-pinned-environment",
    "audit/c1-constant-target-population",
    "audit/c1-tabpfn-rerun",
    "audit/c2-diamonds",
    "audit/c2-food-delivery",
    "audit/c2-foundation-model-cost",
    "audit/c2-physiochemical-protein",
    "audit/c2-superconductivity",
    "audit/c2-table2-diamonds",
    "audit/c2-table2-food-delivery",
    "audit/c2-table2-protein",
    "audit/c2-table2-superconductivity",
    "audit/c3-covgap-convergence",
    "audit/c3-tabpfn-rerun",
    "audit/c4-asymmetric-decomposition",
    "audit/c5-cifar10",
    "audit/c5-fashionmnist",
    "audit/c5-mnist",
    "audit/c5-table4-classification",
    "audit/c6-algorithm1-cross-validation",
    "audit/c6-tabpfn-rerun",
    "integration/full-claim-stage-suite",
    "main",
}
EXPECTED_COMMITS = 175
CANONICAL_IDENTITY = "MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>"
CLAIM_IDS = ["C1", "C2", "C3", "C4", "C5", "C6"]
EXPECTED_CLAIM_STATUSES = {
    "C1": "VERIFIED_SCOPED_LIVE",
    "C2": "BLOCKED_COMPUTE",
    "C3": "VERIFIED_SCOPED_LIVE",
    "C4": "VERIFIED_SCOPED_LIVE",
    "C5": "BLOCKED_COMPUTE",
    "C6": "VERIFIED_SCOPED_CANDIDATE",
}


def load(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"verification failed: {message}")


def published_branches() -> set[str]:
    remote = git("for-each-ref", "refs/remotes/origin", "--format=%(refname:short)").splitlines()
    remote = {
        name.removeprefix("origin/")
        for name in remote
        if name.startswith("origin/") and name != "origin/HEAD"
    }
    if remote:
        return remote
    return set(git("for-each-ref", "refs/heads", "--format=%(refname:short)").splitlines())


def main() -> None:
    claims = load("claims.json")
    verdicts = load("reproduction_verdicts.json")
    manifest = load("EVIDENCE_MANIFEST.json")
    state = load("AUTONOMOUS_STATE.json")
    claim6 = load("outputs/claim6_independent_audit.json")
    publication = load("outputs/publication_manifest.json")

    require(claims["overall_status"] == EXPECTED_STATUS, "claims overall status")
    require(state["overall_status"] == EXPECTED_STATUS, "autonomous state overall status")
    require([claim["id"] for claim in claims["claims"]] == CLAIM_IDS, "claim ordering")
    statuses = {claim["id"]: claim["status"] for claim in claims["claims"]}
    require(statuses == EXPECTED_CLAIM_STATUSES, "claim statuses")
    require(verdicts["claim_statuses"] == EXPECTED_CLAIM_STATUSES, "verdict statuses")

    required_paths = manifest["required_paths"]
    require(all((ROOT / path).exists() for path in required_paths), "manifest paths")
    require(manifest["controls"]["source_pinned"], "source pin")
    require(manifest["controls"]["claim6_independent_checker"], "Claim 6 independent checker")
    require(manifest["controls"]["claim6_negative_controls_rejected"] == 3, "Claim 6 controls")
    require(manifest["controls"]["claims2_and5_blocker_visible"], "compute blockers")
    require(manifest["controls"]["current_score_claim"] is False, "current score boundary")

    require(claim6["claim"] == 6, "Claim 6 identifier")
    require(claim6["verdict"] == "VERIFIED", "Claim 6 verdict")
    require(claim6["summary_values_recomputed"] == 270, "Claim 6 summary count")
    require(claim6["input_sha256"] == verdicts["claim6_audit"]["input_sha256"], "Claim 6 input digest")
    require(len(claim6["negative_controls"]) == 3, "Claim 6 negative-control count")
    require(all(control["result"] == "PASS" for control in claim6["negative_controls"]), "Claim 6 negative controls")
    require(claim6["partition_checks"]["matches_independent_kfold"], "independent KFold")
    require(claim6["partition_checks"]["test_folds_disjoint_and_complete"], "fold partition")

    require(publication["publication_gate_passed"] is True, "local publication gate")
    require(publication["checks"]["identity_matches"] is True, "Space identity")
    require(publication["checks"]["claim6_independent_audit_verifies"] is True, "published Claim 6 audit")
    require(publication["checks"]["claim2_honest_blocker_visible"] is True, "published Claim 2 blocker")
    require(publication["checks"]["claim5_honest_blocker_visible"] is True, "published Claim 5 blocker")
    require(verdicts["historical_external_result"]["score"] == "6/12", "historical score")
    require(verdicts["historical_external_result"]["current_score_claim"] is False, "current score claim")
    require(verdicts["local_gate"]["publication_allowed"] is False, "publication allowed")
    require(verdicts["local_gate"]["author_endorsement_claimed"] is False, "author endorsement")

    branches = published_branches()
    require(branches == EXPECTED_BRANCHES, "published branch set")
    require(not any(branch.startswith("orx/") for branch in branches), "legacy orx branch")
    require(int(git("rev-list", "--all", "--count")) == EXPECTED_COMMITS, "reachable commit count")
    identities = git("log", "--all", "--format=%an <%ae>\n%cn <%ce>").splitlines()
    require(identities and all(identity == CANONICAL_IDENTITY for identity in identities), "canonical commit identity")
    messages = git("log", "--all", "--format=%B")
    require("co-authored-by:" not in messages.lower(), "co-author trailer")

    print(
        "FINAL_AUDIT=VERIFIED "
        f"branches={len(branches)} commits={EXPECTED_COMMITS} "
        "claims=C1:C3_live_verified,C4_live_verified,C6_cpu_candidate,C2:C5_blocked "
        "claim6=verified_summaries=270_negative_controls=3 "
        "historical_score=6/12 current_score_claim=false "
        "publication_allowed=false"
    )


if __name__ == "__main__":
    main()
