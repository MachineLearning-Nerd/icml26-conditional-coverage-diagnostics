#!/usr/bin/env bash
# Run the expensive, direct pinned-covmetrics fifth-claim replay only after the
# existing Appendix-H and synthetic queues have fully completed. This waiter is
# deliberately zero-CPU until that evidence is durable.
set -euo pipefail

while [[ ! -f outputs/synthetic/summary.json ]]; do
  sleep 30
done

datasets=(ailerons diamonds winequality miami2016 o11 superconductivity deliverytime protein)
for dataset in "${datasets[@]}"; do
  for seed in {0..9}; do
    uv run \
      --with numpy --with pandas --with liac-arff --with scikit-learn \
      --with torch --with scipy --with lightgbm --with pytabkit \
      --with probmetrics --with numba --with catboost --with tqdm \
      python repro/src/run_full_covmetrics_l1_cpu.py "$dataset" "$seed" \
        --raw-dir /tmp/vaap-source-data --output-dir outputs/pinned-covmetrics
  done
done

uv run --with numpy python repro/src/aggregate_cpu_comparator_claim.py \
  --output-dir outputs/pinned-covmetrics \
  --result outputs/pinned-covmetrics/comparator_summary.json
