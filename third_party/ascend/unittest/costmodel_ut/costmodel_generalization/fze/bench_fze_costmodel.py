import argparse
import ast
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from triton.runtime import driver

HERE = Path(__file__).resolve().parent
CASE_PATH = HERE / "test_fze.py"


def load_profile_kernel():
    helper_path = HERE.parent / "profile_utils.py"
    spec = importlib.util.spec_from_file_location("costmodel_profile_utils", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.profile_kernel


def load_case_module():
    spec = importlib.util.spec_from_file_location("fze_case", CASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def literal_tensor_data(name):
    tree = ast.parse(CASE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != name:
            continue
        value = node.value
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "tensor"
                and value.args):
            return ast.literal_eval(value.args[0])
    raise KeyError(f"could not find literal tensor {name}")


def make_inputs():
    torch.manual_seed(0)
    device = "npu"
    a = torch.randn((590, 1024), dtype=torch.float32).to(device)
    b = torch.randn((16, 1024, 1536), dtype=torch.float32).to(device)
    a_eid = torch.tensor(literal_tensor_data("A_eid"), dtype=torch.int64, device=device)
    a_idx = torch.tensor(literal_tensor_data("A_idx"), dtype=torch.int64, device=device)
    a_blk_idx = torch.tensor(literal_tensor_data("A_blk_idx"), dtype=torch.int64, device=device)
    out = torch.empty((a_idx.numel(), b.size(-1)), dtype=a.dtype, device=device)
    return a, b, a_eid, a_idx, a_blk_idx, out


def check_sampled_output(a, b, a_eid, a_idx, a_blk_idx, out):
    """Check one active row from every distinct FZE block at five N columns."""
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    out_cpu = out.cpu()
    eid_cpu = a_eid.cpu()
    idx_cpu = a_idx.cpu()
    starts = sorted(set(int(x) for x in a_blk_idx.cpu().tolist()))
    columns = torch.tensor([0, 255, 511, 1023, 1535], dtype=torch.int64)
    checked = []
    maximum = 0.0
    for start in starts:
        stop = min(start + 64, a_idx.numel())
        block_eids = eid_cpu[start:stop]
        expert = int(block_eids.min())
        active = torch.nonzero(block_eids == expert, as_tuple=False).flatten()
        if active.numel() == 0:
            continue
        row = start + int(active[0])
        input_row = a_cpu[int(idx_cpu[row])]
        weights = b_cpu[expert, :, columns]
        expected = torch.matmul(input_row, weights)
        actual = out_cpu[row, columns]
        maximum = max(maximum, float((actual - expected).abs().max()))
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        checked.append({"block_start": start, "row": row, "expert": expert})
    return maximum, checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=("auto", "simd", "simt_only"), default="auto")
    parser.add_argument("--superblock-factor", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--wall-timing", action="store_true")
    args = parser.parse_args()

    module = load_case_module()
    a, b, a_eid, a_idx, a_blk_idx, out = make_inputs()
    grid = (b.size(-1) // module.BLOCK_N, a_blk_idx.numel())
    logical_programs = grid[0] * grid[1]
    props = driver.active.utils.get_device_properties(torch.npu.current_device())
    vector_cores = int(props["num_vectorcore"])
    compile_mode = "simd_simt" if args.route == "auto" else args.route

    def launch():
        module._FZE[grid](
            a,
            a.stride(0),
            a.stride(1),
            a_eid,
            a_idx,
            a_blk_idx,
            None,
            False,
            b,
            b.stride(0),
            b.stride(1),
            b.stride(2),
            out,
            out.stride(0),
            out.stride(1),
            M=a_idx.numel(),
            K=a.size(1),
            N=b.size(-1),
            E=b.size(0),
            BLOCK_M=module.BLOCK_M,
            BLOCK_N=module.BLOCK_N,
            BLOCK_K=module.BLOCK_K,
            ACC_TYPE=module.tl.float32,
            allow_tf32=module.ALLOW_TF32,
            NO_K_MASK=True,
            NO_N_MASK=True,
            num_warps=4,
            compile_mode=compile_mode,
            superblock_factor=args.superblock_factor,
            logical_program_count_hint=logical_programs,
            physical_vector_core_count_hint=vector_cores,
            auto_simt_scope_mode="auto" if args.route == "auto" else "off",
            auto_simt_scope_dump=str(args.report) if args.report else "",
        )

    # The first launch performs compilation. Validate the exact source case before timing.
    print("phase=first_launch", file=sys.stderr, flush=True)
    launch()
    torch.npu.synchronize()
    print("phase=first_launch_done", file=sys.stderr, flush=True)
    if args.skip_check:
        max_abs, checked, correct = None, [], None
    else:
        print("phase=correctness", file=sys.stderr, flush=True)
        max_abs, checked = check_sampled_output(a, b, a_eid, a_idx, a_blk_idx, out)
        correct = True
        print("phase=correctness_done", file=sys.stderr, flush=True)

    for _ in range(args.warmup):
        launch()
    torch.npu.synchronize()
    print("phase=timing", file=sys.stderr, flush=True)

    if args.wall_timing:
        samples_us = []
        for _ in range(args.repeats):
            begin = time.perf_counter()
            launch()
            torch.npu.synchronize()
            samples_us.append((time.perf_counter() - begin) * 1.0e6)
    else:
        starts = [torch.npu.Event(enable_timing=True) for _ in range(args.repeats)]
        ends = [torch.npu.Event(enable_timing=True) for _ in range(args.repeats)]
        samples_us = []
        for start, end in zip(starts, ends):
            start.record()
            launch()
            end.record()
            torch.npu.synchronize()
            samples_us.append(float(start.elapsed_time(end)) * 1000.0)
    print("phase=timing_done", file=sys.stderr, flush=True)
    samples_us.sort()
    profiler = (load_profile_kernel()(launch, args.profile_dir, "FZE") if args.profile_dir else None)

    result = {
        "case": "fze_default",
        "route_requested": args.route,
        "compile_mode": compile_mode,
        "superblock_factor": args.superblock_factor,
        "shape": {"A": list(a.shape), "B": list(b.shape), "M": int(a_idx.numel())},
        "grid": list(grid),
        "logical_programs": logical_programs,
        "vector_cores": vector_cores,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing_kind": "host_synchronized" if args.wall_timing else "device_event",
        "correct": correct,
        "max_abs": max_abs,
        "sampled_correctness_points": checked,
        "elapsed_us": {
            "median": statistics.median(samples_us),
            "minimum": samples_us[0],
            "maximum": samples_us[-1],
            "samples": samples_us,
        },
        "profiler": profiler,
        "report": str(args.report) if args.report else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
