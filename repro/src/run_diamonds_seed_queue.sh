#!/usr/bin/env bash
# Continue the full Diamonds seed protocol without overlapping CPU workloads.
set -euo pipefail

raw_dir=/tmp/vaap-source-data
output_dir=outputs/full-cpu
runner='repro/src/run_full_repaired_cpu.py'
active_pattern='run_full_repaired_cpu.py diamonds 2'

# Seed 2 may have been launched interactively.  Do not start another job until
# that process has exited and its atomic checkpoint contains all ten sizes.
while pgrep -f "$active_pattern" >/dev/null; do
  sleep 30
done

seed2="$output_dir/diamonds_seed2.json"
until [ -f "$seed2" ] && [ "$(jq '.samples | keys | length' "$seed2")" = 10 ]; do
  sleep 30
done

for seed in 3 4 5 6 7 8 9; do
  result="$output_dir/diamonds_seed${seed}.json"
  if [ -f "$result" ] && [ "$(jq '.samples | keys | length' "$result")" = 10 ]; then
    continue
  fi
  uv run --with numpy --with pandas --with liac-arff --with scikit-learn \
    --with torch --with scipy --with lightgbm --with pytabkit \
    --with probmetrics --with numba python "$runner" diamonds "$seed" \
    --raw-dir "$raw_dir" --output-dir "$output_dir"
done
