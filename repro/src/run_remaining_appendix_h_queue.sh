#!/usr/bin/env bash
# Complete the remaining Appendix-H source protocol without CPU overlap.
#
# This queue intentionally starts only after the independent Diamonds aggregate
# exists.  Each dataset is then run seed-by-seed with the same full-scale
# runner, followed by its independent audit and strict ten-seed aggregate.
set -euo pipefail

raw_dir=/tmp/vaap-source-data
output_dir=outputs/full-cpu
runner=repro/src/run_full_repaired_cpu.py
datasets=(winequality miami2016 o11 superconductivity deliverytime protein)

complete_result() {
  local result=$1
  [ -f "$result" ] && [ "$(jq '.samples | keys | length' "$result")" = 10 ]
}

# The detached Diamonds post-queue is the prerequisite gate.  It creates this
# file only after the full ten-seed audit and aggregate have both succeeded.
until [ -f "$output_dir/diamonds_summary.json" ]; do
  sleep 30
done

for dataset in "${datasets[@]}"; do
  for seed in {0..9}; do
    result="$output_dir/${dataset}_seed${seed}.json"
    if complete_result "$result"; then
      continue
    fi
    uv run --with numpy --with pandas --with liac-arff --with scikit-learn \
      --with torch --with scipy --with lightgbm --with pytabkit \
      --with probmetrics --with numba python "$runner" "$dataset" "$seed" \
      --raw-dir "$raw_dir" --output-dir "$output_dir"
  done

  uv run --with numpy python repro/src/audit_full_cpu_outputs.py "$dataset" \
    --output-dir "$output_dir" > "$output_dir/${dataset}_audit.json"
  uv run --with numpy python repro/src/aggregate_full_cpu.py "$dataset" \
    --output-dir "$output_dir" --result "$output_dir/${dataset}_summary.json"
done
