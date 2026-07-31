#!/usr/bin/env bash
# THE fixed reproduction command for every node of this experiment tree.
#
#   bash repro/run.sh
#
# It never takes arguments.  What a node actually computes is decided entirely
# by the committed file repro/config/stage.json on that node's branch, so every
# node runs the same command over different code and configuration.
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false

uv sync --frozen --no-progress
exec uv run --frozen --no-sync python -m repro.pipeline
