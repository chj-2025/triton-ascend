"""Small NPU-profiler helper shared by generalization benchmarks."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import torch
import torch_npu


def profile_kernel(fn, output_dir: Path, kernel_name: str) -> dict:
    shutil.rmtree(output_dir, ignore_errors=True)
    config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    )
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            schedule=torch_npu.profiler.schedule(wait=0, warmup=2, active=20, repeat=1, skip_first=2),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output_dir)),
            experimental_config=config,
    ) as profiler:
        for _ in range(28):
            fn()
            profiler.step()
    torch.npu.synchronize()
    rows = []
    for path in output_dir.rglob("op_statistic.csv"):
        with path.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    matches = [row for row in rows if kernel_name in row.get("OP Type", "")]
    if len(matches) != 1:
        raise RuntimeError(f"expected one profiler row containing {kernel_name!r}, got {matches!r}")
    row = matches[0]
    return {
        "op_type": row["OP Type"],
        "count": int(row["Count"]),
        "minimum_us": float(row["Min Time(us)"]),
        "average_us": float(row["Avg Time(us)"]),
        "maximum_us": float(row["Max Time(us)"]),
        "profile_dir": str(output_dir),
    }
