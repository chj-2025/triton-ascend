#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 OUTPUT_ROOT SHARD_ROOT SHARD_ROOT [SHARD_ROOT ...]" >&2
  exit 2
fi

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output_root=$1
shift

merge_args=()
for shard_root in "$@"; do
  merge_args+=(--input-root "$shard_root")
done

python "$here/merge_generalization_runs.py" \
  "${merge_args[@]}" \
  --output-root "$output_root"

profile="$output_root/costmodel_profile.json"
manifest="$output_root/generalization_matrix.json"
python "$here/validate_generalization_contract.py" \
  --manifest "$manifest" \
  --profile "$profile" \
  > "$output_root/generalization_contract.json"

set +e
python "$here/audit_generalization_matrix.py" \
  --evidence-root "$output_root" \
  --manifest "$manifest" \
  --profile "$profile" \
  --output-json "$output_root/generalization_audit.json" \
  --output-markdown "$output_root/generalization_audit.md"
audit_status=$?

python "$here/fit_general_route_residual.py" \
  --evidence-root "$output_root" \
  --manifest "$manifest" \
  --profile "$profile" \
  --mode simd \
  --output "$output_root/generic_simd_residual_diagnostic.json"
simd_status=$?

python "$here/fit_general_route_residual.py" \
  --evidence-root "$output_root" \
  --manifest "$manifest" \
  --profile "$profile" \
  --mode simt \
  --output "$output_root/generic_simt_residual_diagnostic.json"
simt_status=$?
set -e

python - "$output_root" "$audit_status" "$simd_status" "$simt_status" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
audit_status, simd_status, simt_status = map(int, sys.argv[2:])
audit = json.loads((root / "generalization_audit.json").read_text())
result = {
    "schema": "triton-costmodel-generalization-gates-v1",
    "status": "PASS"
    if audit_status == simd_status == simt_status == 0 and audit["all_pass"]
    else "FAIL",
    "audit_all_pass": audit["all_pass"],
    "audit_exit_status": audit_status,
    "simd_diagnostic_exit_status": simd_status,
    "simt_diagnostic_exit_status": simt_status,
    "expected_record_count": audit["expected_record_count"],
    "audited_record_count": audit["audited_record_count"],
}
(root / "gate_summary.json").write_text(
    json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
)
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if result["status"] == "PASS" else 1)
PY
