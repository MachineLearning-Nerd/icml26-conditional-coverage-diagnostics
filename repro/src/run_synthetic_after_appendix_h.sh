#!/usr/bin/env bash
# Queue the synthetic claim-3 sweep strictly after the all-eight source manifest.
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root_dir"

manifest=outputs/full-cpu/appendix_h_integrity.json
while [[ ! -f "$manifest" ]]; do
  echo "waiting for all-eight Appendix-H integrity manifest"
  sleep 60
done

if [[ $(jq -r '.dataset_count' "$manifest") != "8" ]]; then
  echo "invalid Appendix-H manifest: $manifest" >&2
  exit 1
fi

for seed in {0..9}; do
  uv run --with numpy --with pandas --with scipy --with scikit-learn --with torch \
    --with lightgbm --with catboost --with pytabkit --with probmetrics --with numba --with tqdm \
    python repro/src/run_synthetic_convergence.py "$seed" --output-dir outputs/synthetic
done

uv run --with numpy --with pandas --with scipy --with scikit-learn --with torch \
  --with lightgbm --with catboost --with pytabkit --with probmetrics --with numba --with tqdm \
  python repro/src/aggregate_synthetic_convergence.py \
  --output-dir outputs/synthetic --result outputs/synthetic/summary.json
