#!/usr/bin/env bash
set -euo pipefail

card="$1"
root="$2"
source /data/kaixin/set_env.sh >/dev/null 2>&1
export ASCEND_RT_VISIBLE_DEVICES="$card"
export TRITON_DISABLE_LINE_INFO=1
unset TRITON_DEBUG || true
export TRITON_CACHE_DIR="/data/kaixin/cache_cv_dot_transition_20261005"

mkdir -p "$root"
for factor in 1 2 4; do
  for order in before after; do
    for dots in 1 2 3 4; do
      for trips in 0 14; do
        output="${root}/f${factor}_${order}_d${dots}_t${trips}"
        if [[ -f "${output}/summary.json" ]]; then
          continue
        fi
        rm -rf "$output"
        /home/pkg/ascend_py311/bin/python \
          /data/kaixin/triton-ascend/third_party/ascend/costmodel/profiles/microbench/data_provider/probe_cv_dot_transition.py \
          "$output" --scope-order "$order" --dots "$dots" --trips "$trips" \
          --logical-programs 28 --samples 16 --factor "$factor"
      done
    done
  done
done
