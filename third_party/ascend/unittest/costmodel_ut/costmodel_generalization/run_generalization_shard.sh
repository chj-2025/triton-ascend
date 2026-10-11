#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 DEVICE SHARD_INDEX SHARD_COUNT OUTPUT_ROOT" >&2
  exit 2
fi

device=$1
shard_index=$2
shard_count=$3
output_root=$4

source /data/kaixin/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="$device"
export TRITON_DEBUG="${TRITON_DEBUG:-0}"
export TRITON_DISABLE_LINE_INFO="${TRITON_DISABLE_LINE_INFO:-0}"

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

python "$here/run_generalization_matrix.py" \
  --output-root "$output_root" \
  --warmup 3 \
  --repeats 7 \
  --include-explicit \
  --shard-index "$shard_index" \
  --shard-count "$shard_count"
