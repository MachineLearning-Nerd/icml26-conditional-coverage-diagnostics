# /// script
# dependencies = [
#   "numpy", "pandas", "liac-arff", "scikit-learn", "torch", "scipy",
#   "lightgbm", "pytabkit", "probmetrics", "numba"
# ]
# ///
"""Run one durable full-scale repaired-protocol seed from mounted project data.

Expected mounts:
  /workspace  read-only synchronization of this project (including pinned source)
  /data       writable existing Hugging Face bucket for raw data and checkpoints
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("seed", type=int)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--data", type=Path, default=Path("/data"))
    args = parser.parse_args()
    command = [
        "python", str(args.workspace / "repro/src/run_full_repaired_cpu.py"),
        args.dataset, str(args.seed),
        "--raw-dir", str(args.data / "vaApZm6MKM/raw"),
        "--output-dir", str(args.data / "vaApZm6MKM/checkpoints"),
    ]
    subprocess.run(command, check=True, cwd=args.workspace)


if __name__ == "__main__":
    main()
