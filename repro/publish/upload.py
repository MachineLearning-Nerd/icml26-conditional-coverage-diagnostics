"""Publish the candidate tree to the existing Space, text files only.

    uv run --frozen python -m repro.publish.upload --space space --dry-run
    uv run --frozen python -m repro.publish.upload --space space --commit

Refuses to run unless:
  * every file is on the text allowlist or is one of the judged revision's own
    binary assets, which are re-uploaded byte-identically;
  * no file matches a secret-shaped pattern;
  * the judged file set is a subset of the candidate file set.

It prints the exact allowlist and a SHA-256 manifest before uploading, and
never creates a second Space.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

SPACE_ID = "DineshAI/vaApZm6MKM"

TEXT_SUFFIXES = {".md", ".json", ".jsonl", ".css", ".js", ".html", ".txt", ".csv", ".yaml", ".yml"}
ALLOWED_BINARY = {".png", ".svg", ".gitattributes"}

SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key|password|bearer)\b\s*[:=]\s*\S{8,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(space: Path) -> dict:
    files, secrets, disallowed = [], [], []
    for path in sorted(space.rglob("*")):
        if not path.is_file() or ".git" in path.parts or ".cache" in path.parts:
            continue
        relative = str(path.relative_to(space))
        suffix = path.suffix or path.name
        is_text = suffix in TEXT_SUFFIXES
        if not is_text and suffix not in ALLOWED_BINARY:
            disallowed.append(relative)
        if is_text:
            content = path.read_text(errors="replace")
            for pattern in SECRET_PATTERNS:
                if pattern.search(content):
                    secrets.append(f"{relative}: {pattern.pattern}")
        files.append({"path": relative, "bytes": path.stat().st_size,
                      "sha256": digest(path), "text": is_text})
    return {"files": files, "secret_hits": secrets, "off_allowlist": disallowed,
            "total_bytes": sum(f["bytes"] for f in files)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", type=Path, default=Path("space"))
    parser.add_argument("--judged", type=Path)
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--message", default="Claim-by-claim verification rebuild")
    args = parser.parse_args()

    report = audit(args.space)
    if args.judged:
        from .build_space import subset_report
        report["subset"] = subset_report(args.judged, args.space)

    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))
    print("\nUpload allowlist and SHA-256 manifest:")
    print(f"{'sha256':<64}  {'bytes':>9}  path")
    for entry in report["files"]:
        print(f"{entry['sha256']}  {entry['bytes']:>9}  {entry['path']}")

    if report["secret_hits"]:
        print("\nREFUSING: secret-shaped content found", file=sys.stderr)
        return 2
    if report["off_allowlist"]:
        print("\nREFUSING: files outside the allowlist", file=sys.stderr)
        return 2
    if args.judged and not report["subset"]["judged_is_subset"]:
        print("\nREFUSING: judged files missing from the candidate tree", file=sys.stderr)
        return 2

    if not args.commit:
        print("\ndry run: nothing uploaded")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    info = api.upload_folder(repo_id=SPACE_ID, repo_type="space", folder_path=str(args.space),
                             commit_message=args.message)
    print(f"\nuploaded to {SPACE_ID}")
    print(f"commit: {getattr(info, 'oid', info)}")
    revision = api.space_info(SPACE_ID).sha
    print(f"published revision: {revision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
