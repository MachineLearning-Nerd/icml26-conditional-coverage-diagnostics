"""Run provenance: the facts a reviewer needs to trust a number."""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import subprocess
import sys
import time

PINNED = {
    "covmetrics": "a5205aada6a0f39e3812daf087753217ef66b159",
    "conditional_coverage_estimation": "39a99dcad92205a15d93f2c5fec40c76540abf1c",
    "arxiv": "2512.11779",
    "openreview": "vaApZm6MKM",
}

TRACKED_PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "torch", "torchvision",
    "lightgbm", "catboost", "probmetrics", "pytabkit", "tabpfn", "tabicl",
    "covmetrics",
)


def _git(*args: str) -> str:
    try:
        return subprocess.run(("git", *args), capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unavailable"


def cpu_count() -> int:
    # The allocation the process may actually use, not the machine's core count.
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def snapshot() -> dict:
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": _git("status", "--porcelain") not in ("", "unavailable"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_allocation": cpu_count(),
        "cpu_count_machine": os.cpu_count(),
        "packages": {name: _version(name) for name in TRACKED_PACKAGES},
        "pinned_sources": PINNED,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _version(name: str) -> str:
    try:
        return md.version(name)
    except Exception:
        return "absent"


class Timer:
    def __enter__(self) -> "Timer":
        self.wall_start = time.perf_counter()
        self.cpu_start = time.process_time()
        return self

    def __exit__(self, *exc: object) -> None:
        self.wall_s = time.perf_counter() - self.wall_start
        self.cpu_s = time.process_time() - self.cpu_start
