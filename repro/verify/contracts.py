"""Machine-checkable claim contracts.

Each contract states, in advance, what the raw evidence must contain for the
claim to be marked VERIFIED, and what would instead falsify it.  A contract is
a list of named checks; a claim passes only if every check passes.  Nothing
here reads a verdict from the evidence — the verdict is derived.

Thresholds are stated as constants with a reason, not tuned after seeing the
numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

ARTIFACTS = Path(".openresearch/artifacts")


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    observed: object = None
    required: str = ""


@dataclass
class ClaimResult:
    claim: str
    statement: str
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "VERIFIED" if self.checks and all(c.passed for c in self.checks) else "FAILED"

    def to_json(self) -> dict:
        return {
            "claim": self.claim,
            "statement": self.statement,
            "verdict": self.verdict,
            "checks": [
                {"name": c.name, "passed": c.passed, "required": c.required,
                 "observed": c.observed, "detail": c.detail}
                for c in self.checks
            ],
            "notes": self.notes,
        }


def load_artifact(relative: str) -> dict:
    path = ARTIFACTS / relative
    if not path.exists():
        raise FileNotFoundError(f"missing evidence: {path}")
    return json.loads(path.read_text())


def load_rows(tag: str) -> list[dict]:
    path = ARTIFACTS / f"rows/{tag}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing evidence rows: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
