"""Evidence transport.

In orx local mode the run log is the only evidence channel, so every artifact
this pipeline produces is *both* written under ``.openresearch/artifacts`` and
printed to stdout inside a delimited block.  ``repro/pipeline/collect.py``
reconstructs the artifact files from a captured log with no other input.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile

BEGIN = "<<<ORX-ARTIFACT-BEGIN"
END = "<<<ORX-ARTIFACT-END"

ARTIFACT_ROOT = Path(".openresearch/artifacts")


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_of(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def artifact(relative_path: str, value: object) -> str:
    """Persist and broadcast one JSON artifact.  Returns its SHA-256."""
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    digest = sha256_of(value)
    write_atomic(ARTIFACT_ROOT / relative_path, text)
    sys.stdout.write(f"{BEGIN} path={relative_path} sha256={digest}>>>\n")
    sys.stdout.write(canonical(value) + "\n")
    sys.stdout.write(f"{END} path={relative_path}>>>\n")
    sys.stdout.flush()
    return digest


def note(message: str) -> None:
    sys.stdout.write(f"[repro] {message}\n")
    sys.stdout.flush()
