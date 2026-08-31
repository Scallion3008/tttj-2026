"""CUDA loader for the exact-order, memory-bounded case-14 softmax."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_extension = None


def _load_extension():
    global _extension
    if _extension is None:
        root = Path(__file__).resolve().parent
        _extension = load(
            name="tttj_case14_softmax_v16",
            sources=[
                str(root / "csrc" / "case14_softmax.cpp"),
                str(root / "csrc" / "case14_softmax.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--threads=4", "-lineinfo"],
            verbose=False,
        )
    return _extension


def exact_softmax(
    input: torch.Tensor,
    logical_classes: int | None = None,
    *,
    query_start: int | None = None,
    input_scale: float = 1.0,
    inplace: bool = False,
    fast_exp: bool = False,
) -> torch.Tensor:
    """Softmax over the final axis with PyTorch's large-row reduction order."""
    if logical_classes is None:
        logical_classes = input.shape[-1]
    return _load_extension().forward(
        input,
        logical_classes,
        -1 if query_start is None else query_start,
        0 if query_start is None else input.shape[-2],
        input_scale,
        inplace,
        fast_exp,
    )


def exact_softmax_stats(
    input: torch.Tensor,
    logical_classes: int,
    *,
    query_start: int,
    input_scale: float = 1.0,
) -> torch.Tensor:
    """Return the exact-order FP32 row maximum and exponential sum."""
    return _load_extension().stats(
        input,
        logical_classes,
        query_start,
        input.shape[-2],
        input_scale,
    )
