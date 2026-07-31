"""Rebuild artifacts from a captured run log.

In orx local mode the run log is the only channel a job's results come back
through, so every artifact is printed as well as written.  This reverses that:
given one or more log files it reconstructs `.openresearch/artifacts/...` and a
`rows.jsonl` per row tag, re-checking each artifact's SHA-256 on the way.

    uv run --frozen python -m repro.pipeline.collect logs/*.log --out .openresearch/artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .emit import BEGIN, END, ROW, canonical, sha256_of, write_atomic


def parse(text: str) -> tuple[dict[str, object], dict[str, list], list[str]]:
    artifacts: dict[str, object] = {}
    rows: dict[str, list] = {}
    problems: list[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(BEGIN):
            header = line[len(BEGIN):].strip().rstrip(">")
            fields = dict(part.split("=", 1) for part in header.split() if "=" in part)
            path, expected = fields.get("path"), fields.get("sha256")
            if index + 1 >= len(lines):
                problems.append(f"{path}: truncated before payload")
                break
            payload = lines[index + 1]
            try:
                value = json.loads(payload)
            except json.JSONDecodeError as error:
                problems.append(f"{path}: unparseable payload ({error})")
                index += 1
                continue
            actual = sha256_of(value)
            if expected and actual != expected:
                problems.append(f"{path}: sha256 mismatch (log said {expected}, payload is {actual})")
            artifacts[path] = value
            index += 2
            if index < len(lines) and lines[index].startswith(END):
                index += 1
            continue
        if line.startswith(ROW):
            tag, _, payload = line[len(ROW):].strip().partition(">>>")
            try:
                rows.setdefault(tag.strip(), []).append(json.loads(payload))
            except json.JSONDecodeError as error:
                problems.append(f"row {tag}: unparseable ({error})")
        index += 1
    return artifacts, rows, problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path(".openresearch/artifacts"))
    args = parser.parse_args()

    artifacts: dict[str, object] = {}
    rows: dict[str, list] = {}
    problems: list[str] = []
    for log in args.logs:
        a, r, p = parse(log.read_text(errors="replace"))
        for path, value in a.items():
            # Shards write dataset-specific paths, so a genuine collision means
            # two logs disagree about the same artifact and must not be merged.
            if path in artifacts and artifacts[path] != value:
                problems.append(f"{path}: conflicting copies across logs")
            artifacts[path] = value
        for tag, records in r.items():
            rows.setdefault(tag, []).extend(records)
        problems.extend(f"{log}: {item}" for item in p)

    for path, value in artifacts.items():
        write_atomic(args.out / path, json.dumps(value, sort_keys=True, indent=2) + "\n")
    for tag, records in rows.items():
        write_atomic(args.out / f"rows/{tag}.jsonl", "".join(canonical(r) + "\n" for r in records))

    print(json.dumps({
        "artifacts": sorted(artifacts),
        "row_counts": {tag: len(records) for tag, records in sorted(rows.items())},
        "problems": problems,
    }, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
