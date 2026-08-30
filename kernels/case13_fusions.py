"""Bandwidth fusions for the D128 case-13 layerwise hybrid."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


MODEL = 128
JIT_MODEL = tl.constexpr(MODEL)
ROW_TILE = int(os.environ.get("TTTJ_STEP7_NORM_ROWS", "8"))
NUM_WARPS = int(os.environ.get("TTTJ_STEP7_NORM_WARPS", "4"))
NORM_ONLY_ROW_TILE = int(os.environ.get("TTTJ_STEP7_INPUT_NORM_ROWS", "8"))
NORM_ONLY_NUM_WARPS = int(os.environ.get("TTTJ_STEP7_INPUT_NORM_WARPS", "2"))


@triton.jit
def _welford_online(value, mean, sigma2, count):
    delta = value - mean
    new_count = count + 1.0
    new_mean = mean + delta * (1.0 / new_count)
    new_sigma2 = sigma2 + delta * (value - new_mean)
    return new_mean, new_sigma2, new_count


@triton.jit
def _welford_combine_halves(
    mean,
    sigma2,
    count,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    mean_halves = tl.permute(
        tl.reshape(mean, (ROWS, 2, WIDTH // 2)), (0, 2, 1)
    )
    sigma2_halves = tl.permute(
        tl.reshape(sigma2, (ROWS, 2, WIDTH // 2)), (0, 2, 1)
    )
    count_halves = tl.permute(
        tl.reshape(count, (ROWS, 2, WIDTH // 2)), (0, 2, 1)
    )
    mean_b, mean_a = tl.split(mean_halves)
    sigma2_b, sigma2_a = tl.split(sigma2_halves)
    count_b, count_a = tl.split(count_halves)
    delta = mean_b - mean_a
    new_count = count_a + count_b
    coefficient = 1.0 / new_count
    fraction_a = count_a * coefficient
    fraction_b = count_b * coefficient
    new_mean = fraction_a * mean_a + fraction_b * mean_b
    new_sigma2 = sigma2_a + sigma2_b
    new_sigma2 += delta * delta * count_a * fraction_b
    return new_mean, new_sigma2, new_count


@triton.jit
def _residual_layer_norm_kernel(
    residual_ptr,
    branch_ptr,
    residual_output_ptr,
    norm_output_ptr,
    weight_ptr,
    bias_ptr,
    valid_ptr,
    ROWS: tl.constexpr,
    ADD_BRANCH: tl.constexpr,
    ALL_VALID: tl.constexpr,
    MASK_BRANCH: tl.constexpr,
    MASK_COMBINED: tl.constexpr,
    STORE_RESIDUAL: tl.constexpr,
    MASK_NORM_OUTPUT: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, JIT_MODEL)
    offsets = rows[:, None] * JIT_MODEL + columns[None, :]
    residual = tl.load(residual_ptr + offsets)
    if ADD_BRANCH:
        branch = tl.load(branch_ptr + offsets)
        if not ALL_VALID:
            valid = tl.load(valid_ptr + rows)
            if MASK_BRANCH:
                branch = tl.where(valid[:, None], branch, 0.0)
        combined = (residual + branch).to(tl.float16)
    else:
        combined = residual
    if MASK_COMBINED and not ALL_VALID:
        combined = tl.where(valid[:, None], combined, 0.0)
    if STORE_RESIDUAL:
        tl.store(residual_output_ptr + offsets, combined)

    # Match PyTorch's aligned D128 FP16 LayerNorm: each lane consumes one
    # half4 and the 32 Welford states are combined with shfl_down.
    values = combined.to(tl.float32)
    lane_values = tl.reshape(values, (ROWS, 32, 4))
    lane_pairs = tl.reshape(lane_values, (ROWS, 32, 2, 2))
    even, odd = tl.split(lane_pairs)
    item0, item2 = tl.split(even)
    item1, item3 = tl.split(odd)
    mean = tl.zeros_like(item0)
    sigma2 = tl.zeros_like(item0)
    count = tl.zeros_like(item0)
    mean, sigma2, count = _welford_online(item0, mean, sigma2, count)
    mean, sigma2, count = _welford_online(item1, mean, sigma2, count)
    mean, sigma2, count = _welford_online(item2, mean, sigma2, count)
    mean, sigma2, count = _welford_online(item3, mean, sigma2, count)
    mean, sigma2, count = _welford_combine_halves(
        mean, sigma2, count, ROWS, 32
    )
    mean, sigma2, count = _welford_combine_halves(
        mean, sigma2, count, ROWS, 16
    )
    mean, sigma2, count = _welford_combine_halves(
        mean, sigma2, count, ROWS, 8
    )
    mean, sigma2, count = _welford_combine_halves(
        mean, sigma2, count, ROWS, 4
    )
    mean, sigma2, count = _welford_combine_halves(
        mean, sigma2, count, ROWS, 2
    )
    mean = tl.reshape(mean, (ROWS,))
    variance = tl.reshape(sigma2, (ROWS,)) / JIT_MODEL
    inverse_std = libdevice.rsqrt(variance + 1.0e-5)
    weight = tl.load(weight_ptr + columns)[None, :].to(tl.float32)
    bias = tl.load(bias_ptr + columns)[None, :].to(tl.float32)
    normalized = weight * (inverse_std[:, None] * (values - mean[:, None])) + bias
    if MASK_NORM_OUTPUT and not ALL_VALID:
        normalized = tl.where(valid[:, None], normalized, 0.0)
    tl.store(norm_output_ptr + offsets, normalized)


def residual_layer_norm(
    residual: torch.Tensor,
    branch: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    valid_token_mask: torch.Tensor,
    *,
    all_valid: bool,
    mask_branch: bool,
    mask_combined: bool,
    store_residual: bool = True,
    mask_norm_output: bool = False,
    _row_tile: int = ROW_TILE,
    _num_warps: int = NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    if residual.shape != branch.shape or residual.shape[-1] != MODEL:
        raise ValueError("case-13 residual fusion expects equal D128 tensors")
    if residual.dtype != torch.float16 or not residual.is_cuda:
        raise ValueError("case-13 residual fusion expects CUDA float16")
    rows = residual.numel() // MODEL
    if rows % _row_tile:
        raise ValueError("row count must be divisible by the configured row tile")
    residual_output = torch.empty_like(residual) if store_residual else residual
    norm_output = torch.empty_like(residual)
    _residual_layer_norm_kernel[(rows // _row_tile,)](
        residual,
        branch,
        residual_output,
        norm_output,
        weight,
        bias,
        valid_token_mask,
        ROWS=_row_tile,
        ADD_BRANCH=True,
        ALL_VALID=all_valid,
        MASK_BRANCH=mask_branch,
        MASK_COMBINED=mask_combined,
        STORE_RESIDUAL=store_residual,
        MASK_NORM_OUTPUT=mask_norm_output,
        num_warps=_num_warps,
        num_stages=1,
    )
    return residual_output, norm_output


def standalone_layer_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    _row_tile: int = NORM_ONLY_ROW_TILE,
    _num_warps: int = NORM_ONLY_NUM_WARPS,
) -> torch.Tensor:
    if value.shape[-1] != MODEL or value.dtype != torch.float16 or not value.is_cuda:
        raise ValueError("case-13 standalone LayerNorm expects CUDA FP16 D128")
    rows = value.numel() // MODEL
    if rows % _row_tile:
        raise ValueError("row count must be divisible by the configured row tile")
    output = torch.empty_like(value)
    _residual_layer_norm_kernel[(rows // _row_tile,)](
        value,
        value,
        value,
        output,
        weight,
        bias,
        value,
        ROWS=_row_tile,
        ADD_BRANCH=False,
        ALL_VALID=True,
        MASK_BRANCH=False,
        MASK_COMBINED=False,
        STORE_RESIDUAL=False,
        MASK_NORM_OUTPUT=False,
        num_warps=_num_warps,
        num_stages=1,
    )
    return output


def fusion_tuning() -> dict[str, int]:
    return {
        "residual_norm_rows": ROW_TILE,
        "residual_norm_warps": NUM_WARPS,
        "input_norm_rows": NORM_ONLY_ROW_TILE,
        "input_norm_warps": NORM_ONLY_NUM_WARPS,
    }
