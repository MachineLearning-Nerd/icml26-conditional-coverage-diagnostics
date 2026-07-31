"""Build the evaluator-visible candidate Space tree.

    uv run --frozen python -m repro.publish.build_space --judged <judged-dir> --out space

The evaluator sees only what is reachable from the canonical entrypoint, so
every number, command, code excerpt, checker output and control output a claim
rests on is written inline on that claim's page, with the raw artifact linked
beside it as a file in the same tree.

The judged revision is copied in first and never edited except to add a
supersession banner above its verification page, so the judged file set is a
subset of the candidate file set by construction; `--check-subset` proves it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from .pages import PAGE_DESCRIPTIONS, write_pages


def build_index(pages: list[tuple[str, str]]) -> str:
    lines = ["# Repro - Conditional Coverage Diagnostics for Conformal Prediction", "",
             "Claim-by-claim CPU-only reproduction of *Conditional Coverage Diagnostics for "
             "Conformal Prediction* (arXiv `2512.11779`, OpenReview `vaApZm6MKM`).", "",
             "**Start here: [Current verification](#/current-verification)** - the claim table, "
             "the visibility matrix, the exact command and the pinned environment.", "",
             "| Page | What is on it |", "| --- | --- |"]
    for slug, title in pages:
        lines.append(f"| [{title}](#/{slug}) | {PAGE_DESCRIPTIONS.get(slug, '')} |")
    return "\n".join(lines) + "\n"


def subset_report(judged: Path, candidate: Path) -> dict:
    def relative_files(root: Path) -> set[str]:
        return {str(p.relative_to(root)) for p in root.rglob("*")
                if p.is_file() and ".cache" not in p.parts and ".git" not in p.parts}

    old, new = relative_files(judged), relative_files(candidate)
    missing = sorted(old - new)
    return {"judged_files": len(old), "candidate_files": len(new),
            "judged_is_subset": not missing, "missing": missing,
            "added": sorted(new - old)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path(".openresearch/artifacts"))
    parser.add_argument("--out", type=Path, default=Path("space"))
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(args.judged, args.out, ignore=shutil.ignore_patterns(".cache", ".git*"))
    shutil.copy2(args.judged / ".gitattributes", args.out / ".gitattributes")

    pages = write_pages(args.out, args.artifacts)

    logbook = json.loads((args.out / "logbook.json").read_text())
    existing = {child["slug"]: child for child in logbook["root"]["children"]}
    new_slugs = {slug for slug, _ in pages}
    children = [{"slug": slug, "title": title, "file": f"pages/{slug}/page.md", "children": []}
                for slug, title in pages]
    ordered_existing = [(slug, existing[slug]["title"]) for slug in existing
                        if slug not in new_slugs]
    children += [existing[slug] for slug, _ in ordered_existing]
    logbook["root"]["children"] = children
    (args.out / "logbook.json").write_text(json.dumps(logbook, indent=2) + "\n")
    (args.out / "pages/index.md").write_text(build_index(pages + ordered_existing))

    report = subset_report(args.judged, args.out)
    print(json.dumps(report, indent=2))
    return 0 if report["judged_is_subset"] else 1


if __name__ == "__main__":
    sys.exit(main())
