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

# The job box exposes 64 vCPUs.  Small-batch torch training (RealMLP's 256-row
# batches, the image ERT classifier's 32-row batches) spends more time in
# intra-op thread synchronisation than in arithmetic at that width, so the
# per-process thread count is capped and parallelism is taken across
# independent work items instead.  The paper's own CPU hardware was 8 and 16
# cores, so this is also closer to the reported setup.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

# The Hugging Face CPU job image ships python but not uv, so bootstrap the
# environment manager itself before it takes over dependency resolution.  Only
# uv is installed this way; every project dependency comes from uv.lock.
if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 \
      || python3 -m pip install --quiet --disable-pip-version-check uv
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi
uv --version

uv python install 3.11
uv sync --frozen --no-progress
exec uv run --frozen --no-sync python -m repro.pipeline
