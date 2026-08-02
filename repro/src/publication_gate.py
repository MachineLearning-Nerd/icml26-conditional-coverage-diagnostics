#!/usr/bin/env python3
"""Fail closed on scientific, structural, and no-regression release gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATHS = {
    ".gitattributes",
    "README.md",
    "bucket-icon.svg",
    "index.html",
    "logbook.css",
    "logbook.js",
    "logbook.json",
    "pages/claim-1/page.md",
    "pages/claim-2/page.md",
    "pages/claim-3/page.md",
    "pages/claim-4/page.md",
    "pages/claim-5/page.md",
    "pages/claim-6/page.md",
    "pages/current-verification/page.md",
    "pages/index.md",
    "pages/overview/page.md",
    "pages/raw-evidence/page.md",
    "pages/reproduce/page.md",
    "pages/results/page.md",
    "pages/verification/page.md",
    "raw/claim1_constant_target__raw.json",
    "raw/claim3_convergence__raw.json",
    "raw/claim4_decomposition__raw.json",
    "raw/claim6_algorithm1__raw.json",
    "style.css",
    "trackio-logo-light.png",
    "trackio-logo.png",
    "trackio-wordmark-dark.png",
}
BASELINE_NODES = {
    "current-verification",
    "claim-1",
    "claim-2",
    "claim-3",
    "claim-4",
    "claim-5",
    "claim-6",
    "raw-evidence",
    "reproduce",
    "overview",
    "results",
    "verification",
}
BANKED_HASHES = {
    "pages/claim-1/page.md": "7dbe488df829cbe000543ca7133da24e66cad89b63a5ef432e29e80ee97185fd",
    "pages/claim-3/page.md": "bed4be85bf0a9be4fa00553252a599010684bb2ff197a6ac36c456cf1c2ed4fd",
    "pages/claim-4/page.md": "30592a20e122095a42e8f3a690c51d6d3b29b5bd9a6119aa5882a7c92faf5245",
}
PUBLISH_PATHS = {
    "GATE_READY.md",
    "docs/CAMPAIGN_AUDIT.md",
    "logbook.json",
    "outputs/claim6_independent_audit.json",
    "outputs/publication_manifest.json",
    "pages/claim-2/page.md",
    "pages/claim-5/page.md",
    "pages/claim-6/page.md",
    "pages/conclusion/page.md",
    "pages/executive-summary/page.md",
    "pages/index.md",
    "poster_embed.html",
    "repro/src/audit_claim6.py",
    "repro/src/publication_gate.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    audit = json.loads((ROOT / "outputs/claim6_independent_audit.json").read_text(encoding="utf-8"))
    logbook = json.loads((ROOT / "logbook.json").read_text(encoding="utf-8"))
    children = logbook["root"]["children"]
    slugs = [child["slug"] for child in children]
    expected_prefix = [
        "executive-summary",
        "claim-1",
        "claim-2",
        "claim-3",
        "claim-4",
        "claim-5",
        "claim-6",
    ]
    claim6_page = (ROOT / "pages/claim-6/page.md").read_text(encoding="utf-8")
    claim2_page = (ROOT / "pages/claim-2/page.md").read_text(encoding="utf-8")
    claim5_page = (ROOT / "pages/claim-5/page.md").read_text(encoding="utf-8")
    checks = {
        "identity_matches": logbook["space_id"] == "DineshAI/vaApZm6MKM",
        "claim6_independent_audit_verifies": audit["verdict"] == "VERIFIED",
        "claim6_raw_digest_matches": audit["input_sha256"] == "59aeb547948d67117da86f28d3708c572ba296b45594bc3c656f5307e1843852",
        "claim6_all_summaries_recomputed": audit["summary_values_recomputed"] == 270,
        "claim6_negative_controls_pass": all(control["result"] == "PASS" for control in audit["negative_controls"]),
        "claim6_job_and_source_visible": "6a6c481323ed89c748ec92cd" in claim6_page and "c6f68ec340b9e01a261a02e753666721bf210645" in claim6_page,
        "claim2_honest_blocker_visible": "BLOCKED — >2h CPU" in claim2_page,
        "claim5_honest_blocker_visible": "BLOCKED — >2h CPU" in claim5_page,
        "all_judged_paths_retained": all((ROOT / path).is_file() for path in BASELINE_PATHS),
        "all_judged_nodes_retained": BASELINE_NODES.issubset(slugs),
        "banked_claim_pages_byte_identical": all(sha256(ROOT / path) == digest for path, digest in BANKED_HASHES.items()),
        "canonical_required_prefix": slugs[: len(expected_prefix)] == expected_prefix,
        "conclusion_is_last": slugs[-1] == "conclusion",
        "poster_is_pinned_and_posterly": "poster_embed.html" in (ROOT / "pages/executive-summary/page.md").read_text(encoding="utf-8") and "Chenruishuo/posterly" in (ROOT / "poster_embed.html").read_text(encoding="utf-8"),
        "all_release_paths_are_utf8": all(
            (ROOT / path).read_text(encoding="utf-8") is not None
            for path in PUBLISH_PATHS
            if path not in {"GATE_READY.md", "outputs/publication_manifest.json"}
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"publication gates failed: {failed}")

    (ROOT / "GATE_READY.md").write_text(
        "ADDITIVE_GATE_READY: vaApZm6MKM\nClaim 6: VERIFIED\nClaims 2 and 5: BLOCKED — >2h CPU\n",
        encoding="utf-8",
    )
    manifest_paths = sorted(PUBLISH_PATHS - {"outputs/publication_manifest.json"})
    manifest = {
        "paper_id": "vaApZm6MKM",
        "judged_baseline_sha": "214cfb6aabee9c072106bb80bc2b888f356442b8",
        "checks": checks,
        "publication_gate_passed": True,
        "candidate_scope": "additive Claim 6 verification; Claims 2 and 5 remain blocked",
        "text_allowlist": sorted(PUBLISH_PATHS),
        "sha256": {path: sha256(ROOT / path) for path in manifest_paths},
    }
    (ROOT / "outputs/publication_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"publication_gate_passed": True, "paths": len(PUBLISH_PATHS)}, sort_keys=True))


if __name__ == "__main__":
    main()
