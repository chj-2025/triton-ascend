#!/usr/bin/env bash
set -euo pipefail

output_root=${1:?usage: $0 OUTPUT_ROOT [DEVICE]}
device=${2:-0}
probe_dir=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$output_root"
for factor in 1 2 4; do
  for q in 1 2 3 4 5 8; do
    for variant in copy gather dot gather_dot; do
      case_dir="$output_root/${variant}_f${factor}_q${q}"
      ASCEND_RT_VISIBLE_DEVICES=$device \
      TRITON_ALL_BLOCKS_PARALLEL=1 \
        python3 "$probe_dir/probe_grid_stride_resource_composition.py" \
          "$case_dir" --variant "$variant" --factor "$factor" --q "$q" \
          >"${case_dir}.log" 2>&1
    done
  done
done
