#!/usr/bin/env python3
"""Small correctness and SIMT route-model driver for solve_tril.py."""

from __future__ import annotations

import argparse
import functools
import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


def _install_case_utils() -> None:
    """Provide the two test helpers imported by the standalone case."""
    case_utils = types.ModuleType("utils")

    def input_guard(function):

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            guarded_args = tuple(
                value.contiguous() if isinstance(value, torch.Tensor) and not value.is_contiguous() else value
                for value in args)
            guarded_kwargs = {
                name: (value.contiguous() if isinstance(value, torch.Tensor) and not value.is_contiguous() else value)
                for name, value in kwargs.items()
            }
            return function(*guarded_args, **guarded_kwargs)

        return wrapper

    case_utils.input_guard = input_guard

    def assert_close(
        _name: str,
        expected: torch.Tensor,
        actual: torch.Tensor,
        atol: float,
    ) -> None:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=atol)

    case_utils.assert_close = assert_close
    sys.modules["utils"] = case_utils


def _load_case(case_file: str):
    _install_case_utils()
    case_path = Path(__file__).with_name(case_file)
    spec = importlib.util.spec_from_file_location("solve_tril_case", case_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {case_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bt", type=int, choices=(16, 32, 64), required=True)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--case-file", default="solve_tril.py")
    parser.add_argument(
        "--scope-mode",
        choices=("manual", "none"),
        default="manual",
        help="Use the explicit BT64 scope or the otherwise identical unscoped kernel.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.npu.set_device(0)

    batch = 1
    heads = args.heads
    tokens = args.tokens or args.bt
    width = 64
    if tokens % args.bt:
        raise ValueError("--tokens must be divisible by --bt")
    key = F.normalize(
        torch.randn(
            (batch, heads, tokens, width),
            dtype=torch.float32,
            device="npu",
        ),
        dim=-1,
    )
    chunks = key.reshape(batch, heads, -1, args.bt, width)
    matrix = (chunks @ chunks.transpose(-1, -2)).tril(-1)
    identity = torch.eye(args.bt, dtype=matrix.dtype, device=matrix.device)
    expected = torch.inverse(matrix + identity[None, None, None, ...])
    expected = expected.reshape(batch, heads, tokens, args.bt)

    case = _load_case(args.case_file)
    case_input = matrix.reshape(batch, heads, tokens, args.bt).transpose(1, 2)
    if args.scope_mode == "none":
        actual = case.solve_tril_mix(case_input, use_manual_scope=False).transpose(1, 2)
    else:
        actual = case.solve_tril_mix(case_input).transpose(1, 2)
    torch.npu.synchronize()

    max_abs_error = (actual - expected).abs().max().item()
    torch.testing.assert_close(actual, expected, atol=1.0e-4, rtol=1.0e-4)
    print(f"solve_tril BT={args.bt}: PASS; "
          f"shape={tuple(actual.shape)}; max_abs_error={max_abs_error:.6e}")


if __name__ == "__main__":
    main()
