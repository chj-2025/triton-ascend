#!/usr/bin/env bash
# Build + run the Ascend SIMT probes ON THE BOARD (dav-c310):
#   meas  -> SIMT launch overhead (empty-launch / SIMT-scan / SIMD-scan)
#   busy  -> warp/lane parallelism (is Ascend SIMT real or fake?)
#   tput  -> saturated peak throughput SIMD vs SIMT + #independent warps
#   decomp-> WHY simt_cumsum_core is ~17x slower, component by component
# Run this in a LOGIN shell on triton_a5 from the directory holding the sources:
#     bash -lc './build_and_run.sh'          # build + run both
#     bash -lc './build_and_run.sh meas'     # just the launch-overhead probe
#     bash -lc './build_and_run.sh busy'     # just the warp/lane probe
# (a login shell is required so conda + CANN env are set up; do NOT hack LD_LIBRARY_PATH).
set -e

# --- environment (kaixin conda for toolchain, CANN env for ccec + runtime) ---
# Keep an explicitly initialized TD/PR environment; never switch TD to kaixin.
if [[ -z "${ASCEND_TOOLKIT_HOME:-}" ]]; then
  if [[ -n "${PROBE_ENV_SCRIPT:-}" ]]; then
    source "$PROBE_ENV_SCRIPT"
  elif [[ -f /data/kaixin/set_env.sh ]]; then
    source /data/kaixin/set_env.sh
  else
    echo "Initialize the target Conda/CANN environment or set PROBE_ENV_SCRIPT." >&2
    exit 1
  fi
fi

# Template headers (RegBase/VecUtils.h, RegBase/Cumulative/SIMTCumsumCore.h, ...).
# Override by exporting INC=... if your catfood checkout lives elsewhere.
INC="${INC:-/data/kaixin/AscendNPU-IR-Dev/bishengir/lib/Template/include}"
TK="${ASCEND_TOOLKIT_HOME:?ASCEND_TOOLKIT_HOME unset - did set_env.sh run?}"
HOST_ARCH="$(uname -m)"

echo "INC = $INC"
echo "TK  = $TK"
[ -f "$INC/RegBase/VecUtils.h" ] || { echo "!! RegBase/VecUtils.h not found under INC"; exit 1; }

# build one probe: <name>.cce -> <name>.o (device) and <name>_host.cpp -> <name>_host (host)
build_probe() {
  local name="$1"
  local arch="${PROBE_DEVICE_ARCH:-dav-c310}"
  case "$name" in
    tput|simd_memory|simt_memory|simt_gm_memory|simt_shuffle|setup_probe|dependent_latency|operation_rates|control_rates|scan_rates|transition_pair)
      arch="${PROBE_DEVICE_ARCH:-dav-c310-vec}" ;;
  esac
  echo "--- building $name.o (device) ---"
  ccec -c -std=c++17 -O2 --cce-aicore-only --cce-aicore-arch="$arch" \
       -I"$INC" "$name.cce" -o "$name.o"
  echo "--- building ${name}_host (host) ---"
  g++ -O2 "${name}_host.cpp" -o "${name}_host" \
      -I"$TK/$HOST_ARCH-linux/pkg_inc" -I"$TK/$HOST_ARCH-linux/include" \
      -L"$TK/$HOST_ARCH-linux/lib64" \
      -Wl,-rpath,"$TK/$HOST_ARCH-linux/lib64" -lruntime -lascendcl
}

run_probe() {
  local name="$1"
  echo "=== running $name ==="
  "./${name}_host"
  echo
}

targets="${*:-meas busy tput decomp}"
if [[ " $targets " == *" operation_rates "* ]]; then
  echo "operation_rates must use ./run_operation_rates.sh: it builds one" >&2
  echo "compile-time-specialized binary per operation and audits optimized IR." >&2
  exit 2
fi
if [[ " $targets " == *" scan_rates "* ]]; then
  echo "scan_rates must use ./run_scan_rates.sh: it specializes scan/identity" >&2
  echo "modes at compile time and audits optimized IR." >&2
  exit 2
fi
for t in $targets; do
  build_probe "$t"
done
for t in $targets; do
  run_probe "$t"
done
