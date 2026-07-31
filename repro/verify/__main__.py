"""The cumulative claim verifier.

    uv run --frozen python -m repro.verify              # every claim with evidence present
    uv run --frozen python -m repro.verify --claims 1 4 # a subset

Exits non-zero if any requested claim's contract is not satisfied, if its
evidence is missing, or if a negative control fails to fire.  A claim with no
evidence is reported BLOCKED and still fails the run, so a missing artifact can
never be mistaken for a pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from .claims import CLAIM_TEXT, VERIFIERS
from .contracts import ClaimResult


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", nargs="*", default=sorted(VERIFIERS))
    parser.add_argument("--out", type=Path, default=Path(".openresearch/artifacts/verification.json"))
    parser.add_argument("--allow-missing", action="store_true",
                        help="report claims without evidence as BLOCKED but still exit non-zero")
    args = parser.parse_args()

    results = []
    for claim in args.claims:
        try:
            results.append(VERIFIERS[claim]().to_json())
        except FileNotFoundError as error:
            results.append({"claim": claim, "statement": CLAIM_TEXT[claim], "verdict": "BLOCKED",
                            "checks": [], "notes": [f"evidence not present: {error}"]})
        except Exception:
            results.append({"claim": claim, "statement": CLAIM_TEXT[claim], "verdict": "ERROR",
                            "checks": [], "notes": [traceback.format_exc()]})

    report = {"results": results,
              "verdicts": {r["claim"]: r["verdict"] for r in results},
              "all_verified": all(r["verdict"] == "VERIFIED" for r in results)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    width = max(len(c) for c in args.claims) + 2
    for result in results:
        print(f"claim {result['claim']:<{width}} {result['verdict']}")
        for check in result["checks"]:
            print(f"   [{'PASS' if check['passed'] else 'FAIL'}] {check['name']}: {check['detail']}")
            if not check["passed"]:
                print(f"          required: {check['required']}")
        for note in result["notes"]:
            print(f"   note: {note}")
    print(json.dumps(report["verdicts"], sort_keys=True))
    return 0 if report["all_verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
