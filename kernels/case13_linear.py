"""Tensor-core linear epilogue fusions for the case-13 hybrid."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


MODEL = 128
JIT_MODEL = tl.constexpr(MODEL)
QKV_OUTPUT = 3 * MODEL
BATCH = 64
SEQUENCE = 1024
HEADS = 4
HEAD_DIM = MODEL // HEADS
JIT_BATCH = tl.constexpr(BATCH)
JIT_SEQUENCE = tl.constexpr(SEQUENCE)
JIT_HEADS = tl.constexpr(HEADS)
JIT_HEAD_DIM = tl.constexpr(HEAD_DIM)
GELU_ROW_TILE = int(os.environ.get("TTTJ_STEP7_GELU_M", "64"))
NORM_ROW_TILE = int(os.environ.get("TTTJ_STEP7_LINEAR_NORM_M", "64"))
QKV_ROW_TILE = int(os.environ.get("TTTJ_STEP7_QKV_M", "64"))
REDUCTION_TILE = int(os.environ.get("TTTJ_STEP7_LINEAR_K", "64"))
GELU_NUM_WARPS = int(
    os.environ.get("TTTJ_STEP7_GELU_WARPS", "4")
)
GELU_NUM_STAGES = int(
    os.environ.get("TTTJ_STEP7_GELU_STAGES", "3")
)
NORM_NUM_WARPS = int(
    os.environ.get("TTTJ_STEP7_LINEAR_NORM_WARPS", "8")
)
NORM_NUM_STAGES = int(
    os.environ.get("TTTJ_STEP7_LINEAR_NORM_STAGES", "2")
)
QKV_NUM_WARPS = int(os.environ.get("TTTJ_STEP7_QKV_WARPS", "4"))
QKV_NUM_STAGES = int(os.environ.get("TTTJ_STEP7_QKV_STAGES", "2"))


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
def _linear_accumulator(
    input_ptr,
    weight_ptr,
    bias_ptr,
    row_start,
    ROWS: tl.constexpr,
    REDUCTION: tl.constexpr,
):
    rows = row_start + tl.arange(0, ROWS)
    columns = tl.arange(0, JIT_MODEL)
    reductions = tl.arange(0, REDUCTION)
    accumulator = tl.zeros((ROWS, JIT_MODEL), dtype=tl.float32)
    for reduction_start in range(0, JIT_MODEL, REDUCTION):
        input_tile = tl.load(
            input_ptr
            + rows[:, None] * JIT_MODEL
            + reduction_start
            + reductions[None, :]
        )
        weight_tile = tl.load(
            weight_ptr
            + columns[None, :] * JIT_MODEL
            + reduction_start
            + reductions[:, None]
        )
        accumulator = tl.dot(input_tile, weight_tile, accumulator)
    return accumulator + tl.load(bias_ptr + columns)[None, :]


@triton.jit
def _head_major_qkv_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    ROWS: tl.constexpr,
    REDUCTION: tl.constexpr,
):
    row_start = tl.program_id(0) * ROWS
    branch = tl.program_id(1)
    accumulator = _linear_accumulator(
        input_ptr,
        weight_ptr + branch * JIT_MODEL * JIT_MODEL,
        bias_ptr + branch * JIT_MODEL,
        row_start,
        ROWS,
        REDUCTION,
    )
    rows = row_start + tl.arange(0, ROWS)
    columns = tl.arange(0, JIT_MODEL)
    batch = rows // JIT_SEQUENCE
    sequence = rows % JIT_SEQUENCE
    head = columns // JIT_HEAD_DIM
    feature = columns % JIT_HEAD_DIM
    output_offsets = (
        branch * JIT_BATCH * JIT_HEADS * JIT_SEQUENCE * JIT_HEAD_DIM
        + batch[:, None] * JIT_HEADS * JIT_SEQUENCE * JIT_HEAD_DIM
        + head[None, :] * JIT_SEQUENCE * JIT_HEAD_DIM
        + sequence[:, None] * JIT_HEAD_DIM
        + feature[None, :]
    )
    tl.store(output_ptr + output_offsets, accumulator)


@triton.jit
def _linear_gelu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    ROWS: tl.constexpr,
    REDUCTION: tl.constexpr,
):
    row_start = tl.program_id(0) * ROWS
    accumulator = _linear_accumulator(
        input_ptr,
        weight_ptr,
        bias_ptr,
        row_start,
        ROWS,
        REDUCTION,
    )
    rounded = accumulator.to(tl.float16).to(tl.float32)
    output = 0.5 * rounded * (
        1.0 + libdevice.erf(rounded * 0.7071067811865475)
    )
    rows = row_start + tl.arange(0, ROWS)
    columns = tl.arange(0, JIT_MODEL)
    tl.store(
        output_ptr + rows[:, None] * JIT_MODEL + columns[None, :],
        output,
    )


@triton.jit
def _linear_residual_layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    residual_output_ptr,
    norm_output_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    valid_ptr,
    ROWS: tl.constexpr,
    REDUCTION: tl.constexpr,
    ALL_VALID: tl.constexpr,
    MASK_BRANCH: tl.constexpr,
    MASK_COMBINED: tl.constexpr,
    STORE_RESIDUAL: tl.constexpr,
    MASK_NORM_OUTPUT: tl.constexpr,
):
    row_start = tl.program_id(0) * ROWS
    rows = row_start + tl.arange(0, ROWS)
    columns = tl.arange(0, JIT_MODEL)
    offsets = rows[:, None] * JIT_MODEL + columns[None, :]
    accumulator = _linear_accumulator(
        input_ptr,
        weight_ptr,
        bias_ptr,
        row_start,
        ROWS,
        REDUCTION,
    )
    branch = accumulator.to(tl.float16)
    if not ALL_VALID:
        valid = tl.load(valid_ptr + rows)
        if MASK_BRANCH:
            branch = tl.where(valid[:, None], branch, 0.0)
    combined = (tl.load(residual_ptr + offsets) + branch).to(tl.float16)
    if MASK_COMBINED and not ALL_VALID:
        combined = tl.where(valid[:, None], combined, 0.0)
    if STORE_RESIDUAL:
        tl.store(residual_output_ptr + offsets, combined)

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
    norm_weight = tl.load(norm_weight_ptr + columns)[None, :].to(tl.float32)
    norm_bias = tl.load(norm_bias_ptr + columns)[None, :].to(tl.float32)
    normalized = norm_weight * (
        inverse_std[:, None] * (values - mean[:, None])
    ) + norm_bias
    if MASK_NORM_OUTPUT and not ALL_VALID:
        normalized = tl.where(valid[:, None], normalized, 0.0)
    tl.store(norm_output_ptr + offsets, normalized)


def linear_gelu(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    _row_tile: int = GELU_ROW_TILE,
    _reduction_tile: int = REDUCTION_TILE,
    _num_warps: int = GELU_NUM_WARPS,
    _num_stages: int = GELU_NUM_STAGES,
) -> torch.Tensor:
    if value.shape[-1] != MODEL or weight.shape != (MODEL, MODEL):
        raise ValueError("case-13 fused GELU expects a D128 square projection")
    rows = value.numel() // MODEL
    if rows % _row_tile or MODEL % _reduction_tile:
        raise ValueError("configured fused-GELU tiles must divide the fixed shape")
    output = torch.empty_like(value)
    _linear_gelu_kernel[(rows // _row_tile,)](
        value,
        weight,
        bias,
        output,
        ROWS=_row_tile,
        REDUCTION=_reduction_tile,
        num_warps=_num_warps,
        num_stages=_num_stages,
    )
    return output


def head_major_qkv(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    _row_tile: int = QKV_ROW_TILE,
    _reduction_tile: int = REDUCTION_TILE,
    _num_warps: int = QKV_NUM_WARPS,
    _num_stages: int = QKV_NUM_STAGES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if value.shape != (BATCH, SEQUENCE, MODEL):
        raise ValueError("case-13 head-major QKV expects the fixed B64 S1024 D128")
    if weight.shape != (QKV_OUTPUT, MODEL) or bias.shape != (QKV_OUTPUT,):
        raise ValueError("case-13 head-major QKV expects packed 3D projection")
    rows = BATCH * SEQUENCE
    if rows % _row_tile or MODEL % _reduction_tile:
        raise ValueError("configured head-major QKV tiles must divide the shape")
    output = torch.empty(
        3,
        BATCH,
        HEADS,
        SEQUENCE,
        HEAD_DIM,
        device=value.device,
        dtype=value.dtype,
    )
    _head_major_qkv_kernel[(rows // _row_tile, 3)](
        value,
        weight,
        bias,
        output,
        ROWS=_row_tile,
        REDUCTION=_reduction_tile,
        num_warps=_num_warps,
        num_stages=_num_stages,
    )
    return output[0], output[1], output[2]


def linear_residual_layer_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    valid_token_mask: torch.Tensor,
    *,
    all_valid: bool,
    mask_branch: bool,
    mask_combined: bool,
    store_residual: bool = True,
    mask_norm_output: bool = False,
    _row_tile: int = NORM_ROW_TILE,
    _reduction_tile: int = REDUCTION_TILE,
    _num_warps: int = NORM_NUM_WARPS,
    _num_stages: int = NORM_NUM_STAGES,
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.shape != residual.shape or value.shape[-1] != MODEL:
        raise ValueError("case-13 linear/norm fusion expects equal D128 tensors")
    rows = value.numel() // MODEL
    if rows % _row_tile or MODEL % _reduction_tile:
        raise ValueError("configured linear/norm tiles must divide the fixed shape")
    residual_output = torch.empty_like(residual) if store_residual else residual
    norm_output = torch.empty_like(residual)
    _linear_residual_layer_norm_kernel[(rows // _row_tile,)](
        value,
        weight,
        bias,
        residual,
        residual_output,
        norm_output,
        norm_weight,
        norm_bias,
        valid_token_mask,
        ROWS=_row_tile,
        REDUCTION=_reduction_tile,
        ALL_VALID=all_valid,
        MASK_BRANCH=mask_branch,
        MASK_COMBINED=mask_combined,
        STORE_RESIDUAL=store_residual,
        MASK_NORM_OUTPUT=mask_norm_output,
        num_warps=_num_warps,
        num_stages=_num_stages,
    )
    return residual_output, norm_output


def linear_tuning() -> dict[str, int]:
    return {
        "gelu_m": GELU_ROW_TILE,
        "gelu_warps": GELU_NUM_WARPS,
        "gelu_stages": GELU_NUM_STAGES,
        "norm_m": NORM_ROW_TILE,
        "k": REDUCTION_TILE,
        "norm_warps": NORM_NUM_WARPS,
        "norm_stages": NORM_NUM_STAGES,
        "qkv_m": QKV_ROW_TILE,
        "qkv_warps": QKV_NUM_WARPS,
        "qkv_stages": QKV_NUM_STAGES,
    }
