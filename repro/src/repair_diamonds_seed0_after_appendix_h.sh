#!/usr/bin/env bash
# Repair the sole legacy Diamonds checkpoint only after the serial Appendix-H
# queue is completely finished.  The replacement retains the legacy artifact
# for comparison and makes the ten-seed Diamonds record independently auditable.
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

checkpoint="$output_dir/diamonds_seed0.json"
mkdir -p "$legacy_dir"
if [ -f "$checkpoint" ]; then
  mv "$checkpoint" "$legacy_dir/diamonds_seed0.pre-integrity.json"
fi

uv run --with numpy --with pandas --with liac-arff --with scikit-learn \
  --with torch --with scipy --with lightgbm --with pytabkit \
  --with probmetrics --with numba python "$runner" diamonds 0 \
  --raw-dir "$raw_dir" --output-dir "$output_dir"

uv run --with numpy python repro/src/audit_full_cpu_outputs.py diamonds \
  --output-dir "$output_dir" --require-integrity > "$output_dir/diamonds_audit.json"
uv run --with numpy python repro/src/aggregate_full_cpu.py diamonds \
  --output-dir "$output_dir" --result "$output_dir/diamonds_summary.json"
