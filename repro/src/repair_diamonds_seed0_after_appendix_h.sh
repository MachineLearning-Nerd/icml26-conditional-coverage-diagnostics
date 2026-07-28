#!/usr/bin/env bash
# Repair legacy coverage-integrity checkpoints only after the serial Appendix-H
# queue is complete.  Replacements retain legacy artifacts for comparison and
# make every Appendix-H record independently auditable.
set -euo pipefail

raw_dir=/tmp/vaap-source-data
output_dir=outputs/full-cpu
runner=repro/src/run_full_repaired_cpu.py
legacy_dir="$output_dir/legacy"

# protein is last in the fail-closed remaining-Appendix-H queue.  Its summary
# is written only after all ten protein seeds pass that dataset's audit and
# strict aggregation; it therefore also proves no queue-owned CPU job remains.
until [ -f "$output_dir/protein_summary.json" ]; do
  sleep 30
done

# Do not overlap a user- or queue-owned local full-protocol runner.
while pgrep -f 'run_full_repaired_cpu.py' >/dev/null; do
  sleep 30
done

mkdir -p "$legacy_dir"

repair_seed() {
  local dataset=$1
  local seed=$2
  local checkpoint="$output_dir/${dataset}_seed${seed}.json"
  if [ -f "$checkpoint" ]; then
    mv "$checkpoint" "$legacy_dir/${dataset}_seed${seed}.pre-integrity.json"
  fi
  uv run --with numpy --with pandas --with liac-arff --with scikit-learn \
    --with torch --with scipy --with lightgbm --with pytabkit \
    --with probmetrics --with numba python "$runner" "$dataset" "$seed" \
    --raw-dir "$raw_dir" --output-dir "$output_dir"
}

# Diamonds seed 0 and all Ailerons seeds are valid legacy source results but
# predate coverage digest persistence.  Preserve them and rerun serially.
repair_seed diamonds 0
for seed in {0..9}; do
  repair_seed ailerons "$seed"
done

datasets=(ailerons diamonds winequality miami2016 o11 superconductivity deliverytime protein)
for dataset in "${datasets[@]}"; do
  uv run --with numpy python repro/src/audit_full_cpu_outputs.py "$dataset" \
    --output-dir "$output_dir" --require-integrity > "$output_dir/${dataset}_audit.json"
  uv run --with numpy python repro/src/aggregate_full_cpu.py "$dataset" \
    --output-dir "$output_dir" --result "$output_dir/${dataset}_summary.json"
done

uv run --with numpy python repro/src/aggregate_appendix_h.py \
  --output-dir "$output_dir" --result "$output_dir/appendix_h_integrity.json"
