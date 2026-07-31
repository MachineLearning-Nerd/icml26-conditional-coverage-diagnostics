"""Build the evaluator-visible candidate Space tree.

    uv run --frozen python -m repro.publish.build_space --judged <dir> --out space

The evaluator sees only what is reachable from the canonical entrypoint, so
every number, command, code excerpt, checker output and control output that a
claim rests on is written *inline* on that claim's page, with the raw artifact
linked beside it as a downloadable file in the same tree.

Nothing from the judged revision is dropped: the judged tree is copied in
first, its verification page is relabelled as superseded, and new pages are
added around it.  `--judged` therefore doubles as the subset guarantee.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

REQUIRED_ROWS = (
    "Exact claim and source quantifiers",
    "Assumptions and their numerical audit",
    "Executable source code",
    "Exact command and pinned environment",
    "Raw numerical results inline",
    "Downloadable raw CSV/JSON",
    "Independent checker output",
    "Negative-control output",
    "Limitations and deviations",
    "Git SHA, seeds, CPU and runtime",
    "Verifier that exits non-zero on failure",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def code_excerpt(path: Path, start: int, end: int, language: str = "python") -> str:
    lines = path.read_text().splitlines()[start - 1:end]
    return f"````{language} title={path.name}:{start}-{end}\n" + "\n".join(lines) + "\n````"


def whole_file(path: Path, language: str = "python") -> str:
    return f"````{language} title={path.name}\n{path.read_text().rstrip()}\n````"


def checks_table(result: dict) -> str:
    rows = ["| Check | Result | Observed | Required |", "| --- | --- | --- | --- |"]
    for check in result["checks"]:
        observed = check["observed"]
        if isinstance(observed, list):
            observed = ", ".join(f"{v:+.5f}" if isinstance(v, float) else str(v) for v in observed)
        elif isinstance(observed, float):
            observed = f"{observed:+.6f}"
        rows.append(f"| `{check['name']}` | {'PASS' if check['passed'] else '**FAIL**'} "
                    f"| {observed} | {check['required']} |")
    return "\n".join(rows)


def cell(text: str, block_id: str, title: str, extra: str = "") -> str:
    meta = {"type": "markdown", "id": block_id, "title": title}
    if extra:
        meta["subtitle"] = extra
    return ("\n---\n<!-- trackio-cell\n" + json.dumps(meta) + "\n-->\n" + text + "\n")


def build_index(pages: list[tuple[str, str]]) -> str:
    lines = ["# Repro - Conditional Coverage Diagnostics for Conformal Prediction", "",
             "Claim-by-claim CPU-only reproduction of arXiv `2512.11779` "
             "(OpenReview `vaApZm6MKM`).", "",
             "**Start here: [Current verification](#/current-verification).** It carries the "
             "claim table, the visibility matrix and links to every per-claim page.", "",
             "| Page | What is on it |", "| --- | --- |"]
    lines.extend(f"| [{title}](#/{slug}) | {note} |" for slug, title, note in PAGE_NOTES(pages))
    return "\n".join(lines) + "\n"


def PAGE_NOTES(pages):  # noqa: N802 - small helper kept next to its only caller
    for slug, title in pages:
        yield slug, title, PAGE_DESCRIPTIONS.get(slug, "")


PAGE_DESCRIPTIONS = {
    "current-verification": "Current claim table, visibility matrix, exact command, environment",
    "claim-1": "Constant-target principle: exhaustive population sweep plus full-scale arm",
    "claim-2": "Table-2 relative-power percentages for LightGBM and PartitionWise",
    "claim-3": "CovGap versus L1-ERT convergence at 5,000 test points",
    "claim-4": "Asymmetric over/under-coverage decomposition",
    "claim-5": "Table-4 classification KL+/KL- decomposition",
    "claim-6": "Algorithm 1 cross-validation and its overfitting control",
    "raw-evidence": "Every raw JSON artifact, downloadable",
    "reproduce": "How to re-run everything from scratch",
    "overview": "Original overview cell from the judged revision",
    "results": "Original results cell from the judged revision",
    "verification": "Historical rejected baseline - the superseded five-claim gate",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path(".openresearch/artifacts"))
    parser.add_argument("--out", type=Path, default=Path("space"))
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(args.judged, args.out, ignore=shutil.ignore_patterns(".cache", ".git"))

    from .pages import write_pages
    pages = write_pages(args.out, args.artifacts)

    logbook = read(args.out / "logbook.json")
    existing = {child["slug"]: child for child in logbook["root"]["children"]}
    children = []
    for slug, title in pages:
        children.append({"slug": slug, "title": title,
                         "file": f"pages/{slug}/page.md", "children": []})
    for slug, child in existing.items():
        if slug not in {s for s, _ in pages}:
            children.append(child)
    logbook["root"]["children"] = children
    logbook["root"]["file"] = "pages/index.md"
    (args.out / "logbook.json").write_text(json.dumps(logbook, indent=2) + "\n")
    (args.out / "pages/index.md").write_text(build_index(
        pages + [(s, existing[s]["title"]) for s in existing if s not in {p for p, _ in pages}]))

    print(json.dumps({"pages": [s for s, _ in pages],
                      "files": sorted(str(p.relative_to(args.out))
                                      for p in args.out.rglob("*") if p.is_file())}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
