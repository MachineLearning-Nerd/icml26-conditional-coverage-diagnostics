#!/usr/bin/env python3
"""Independent structural audit for full CPU seed checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def valid_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != hashlib.sha256().digest_size * 2:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def audit(path: Path, require_integrity: bool) -> dict:
    record = json.loads(path.read_text())
    samples = record.get("samples")
    if not isinstance(samples, dict) or len(samples) != 10:
        raise RuntimeError(f"{path}: expected all ten test sizes")
    coverage = record.get("average_test_coverage")
    if not isinstance(coverage, (int, float)) or not math.isfinite(coverage) or not 0 <= coverage <= 1:
        raise RuntimeError(f"{path}: invalid average coverage")
    shape = record.get("shape")
    if not isinstance(shape, dict) or not all(isinstance(shape.get(k), int) and shape[k] > 0 for k in ("train", "calibration", "test")):
        raise RuntimeError(f"{path}: invalid split shape")
    integrity = record.get("coverage_integrity")
    if integrity is None and require_integrity:
        raise RuntimeError(f"{path}: missing coverage integrity")
    if integrity is not None:
        if not isinstance(integrity, dict) or integrity.get("count") != shape["test"]:
            raise RuntimeError(f"{path}: coverage count does not match test split")
        covered = integrity.get("covered")
        if not isinstance(covered, int) or not 0 <= covered <= integrity["count"] or not valid_digest(integrity.get("sha256")):
            raise RuntimeError(f"{path}: invalid persisted coverage integrity")
    return {"seed": record.get("seed"), "coverage": coverage, "integrity": integrity is not None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-integrity", action="store_true")
    args = parser.parse_args()
    paths = sorted(args.output_dir.glob(f"{args.dataset}_seed*.json"))
    if not paths:
        raise RuntimeError("no raw seed checkpoints found")
    result = [audit(path, args.require_integrity) for path in paths]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
