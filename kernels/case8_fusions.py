"""Bandwidth-bound Triton fusions for the case-8 layerwise hybrid."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


MODEL = 1024
JIT_MODEL = tl.constexpr(MODEL)
NUM_WARPS = int(os.environ.get("TTTJ_STEP6_NORM_WARPS", "1"))
NORM_MODE = int(os.environ.get("TTTJ_STEP6_NORM_MODE", "3"))


@triton.jit
def _welford_online(value, mean, sigma2, count):
    new_count = count + 1.0
    delta = value - mean
    new_mean = mean + delta * (1.0 / new_count)
    new_sigma2 = sigma2 + delta * (value - new_mean)
    return new_mean, new_sigma2, new_count


@triton.jit
def _welford_halves(mean, sigma2, count, WIDTH: tl.constexpr):
    mean_halves = tl.permute(tl.reshape(mean, (1, 2, WIDTH // 2)), (0, 2, 1))
    sigma2_halves = tl.permute(
        tl.reshape(sigma2, (1, 2, WIDTH // 2)), (0, 2, 1)
    )
    count_halves = tl.permute(
        tl.reshape(count, (1, 2, WIDTH // 2)), (0, 2, 1)
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
def _welford_warp_halves(
    mean,
    sigma2,
    count,
    WARPS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    mean_halves = tl.permute(
        tl.reshape(mean, (1, WARPS, 2, WIDTH // 2)), (0, 1, 3, 2)
    )
    sigma2_halves = tl.permute(
        tl.reshape(sigma2, (1, WARPS, 2, WIDTH // 2)), (0, 1, 3, 2)
    )
    count_halves = tl.permute(
        tl.reshape(count, (1, WARPS, 2, WIDTH // 2)), (0, 1, 3, 2)
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
    ALL_VALID: tl.constexpr,
    STORE_RESIDUAL: tl.constexpr,
    MASK_NORM_OUTPUT: tl.constexpr,
    ALGORITHM: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, JIT_MODEL)
    offsets = row * JIT_MODEL + columns
    residual = tl.load(residual_ptr + offsets)
    branch = tl.load(branch_ptr + offsets)
    combined = (residual + branch).to(tl.float16)
    if not ALL_VALID:
        valid = tl.load(valid_ptr + row)
        combined = tl.where(valid, combined, 0.0)
    if STORE_RESIDUAL:
        tl.store(residual_output_ptr + offsets, combined)

    values = combined.to(tl.float32)
    if ALGORITHM == 0:
        mean = tl.sum(values, axis=0) / JIT_MODEL
        centered = values - mean
        variance = tl.sum(centered * centered, axis=0) / JIT_MODEL
    elif ALGORITHM == 1:
        mean = tl.sum(values, axis=0) / JIT_MODEL
        mean_square = tl.sum(values * values, axis=0) / JIT_MODEL
        variance = tl.maximum(mean_square - mean * mean, 0.0)
    elif ALGORITHM == 2:
        # CUDA's aligned FP16 LayerNorm assigns one half4 to each of 256
        # threads for D1024 and reduces those per-thread Welford states in a
        # binary block tree.
        lane_values = tl.reshape(values, (1, 256, 4))
        lane_pairs = tl.reshape(lane_values, (1, 256, 2, 2))
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
        mean = tl.reshape(mean, (1, 8, 32))
        sigma2 = tl.reshape(sigma2, (1, 8, 32))
        count = tl.reshape(count, (1, 8, 32))
        mean, sigma2, count = _welford_warp_halves(
            mean, sigma2, count, 8, 32
        )
        mean, sigma2, count = _welford_warp_halves(
            mean, sigma2, count, 8, 16
        )
        mean, sigma2, count = _welford_warp_halves(
            mean, sigma2, count, 8, 8
        )
        mean, sigma2, count = _welford_warp_halves(
            mean, sigma2, count, 8, 4
        )
        mean, sigma2, count = _welford_warp_halves(
            mean, sigma2, count, 8, 2
        )
        mean = tl.reshape(mean, (1, 8))
        sigma2 = tl.reshape(sigma2, (1, 8))
        count = tl.reshape(count, (1, 8))
        # The eight warp leaders are combined through shared memory in the
        # same offset-4, offset-2, offset-1 order.
        mean, sigma2, count = _welford_halves(mean, sigma2, count, 8)
        mean, sigma2, count = _welford_halves(mean, sigma2, count, 4)
        mean, sigma2, count = _welford_halves(mean, sigma2, count, 2)
        mean = tl.reshape(mean, ())
        variance = tl.reshape(sigma2, ()) / JIT_MODEL
    else:
        # Current CUDA PyTorch uses four warps. Each logical thread consumes
        # two half4 vectors (indices lane and lane+128), then the four warp
        # leaders are reduced through shared memory.
        vector_blocks = tl.reshape(values, (1, 2, 128, 4))
        vector_blocks = tl.permute(vector_blocks, (0, 2, 3, 1))
        first_vector, second_vector = tl.split(vector_blocks)
        first_pairs = tl.reshape(first_vector, (1, 128, 2, 2))
        first_even, first_odd = tl.split(first_pairs)
        item0, item2 = tl.split(first_even)
        item1, item3 = tl.split(first_odd)
        second_pairs = tl.reshape(second_vector, (1, 128, 2, 2))
        second_even, second_odd = tl.split(second_pairs)
        item4, item6 = tl.split(second_even)
        item5, item7 = tl.split(second_odd)
        mean = tl.zeros_like(item0)
        sigma2 = tl.zeros_like(item0)
        count = tl.zeros_like(item0)
        mean, sigma2, count = _welford_online(item0, mean, sigma2, count)
        mean, sigma2, count = _welford_online(item1, mean, sigma2, count)
        mean, sigma2, count = _welford_online(item2, mean, sigma2, count)
        mean, sigma2, count = _welford_online(item3, mean, sigma2, count)
        mean, sigma2, count = _welford_online(item4, mean, sigma2, count)
        mean, sigma2, count = _welford_online(item5, mean, sigma2, count)
        mean, sigma2, count = _welford_online(item6, mean, sigma2, count)
        mean, sigma2, count = _welford_online(item7, mean, sigma2, count)
        mean = tl.reshape(mean, (1, 4, 32))
        sigma2 = tl.reshape(sigma2, (1, 4, 32))
        count = tl.reshape(count, (1, 4, 32))
        mean, sigma2, count = _welford_warp_halves(
            mean, sigma2, count, 4, 32
        )
        mean, sigma2, count = _welford_warp_halves(
            mean, sigma2, count, 4, 16
        )
        mean, sigma2, count = _welford_warp_halves(
            mean, sigma2, count, 4, 8
        )
        mean, sigma2, count = _welford_warp_halves(
            mean, sigma2, count, 4, 4
        )
        mean, sigma2, count = _welford_warp_halves(
            mean, sigma2, count, 4, 2
        )
        mean = tl.reshape(mean, (1, 4))
        sigma2 = tl.reshape(sigma2, (1, 4))
        count = tl.reshape(count, (1, 4))
        mean, sigma2, count = _welford_halves(mean, sigma2, count, 4)
        mean, sigma2, count = _welford_halves(mean, sigma2, count, 2)
        mean = tl.reshape(mean, ())
        variance = tl.reshape(sigma2, ()) / JIT_MODEL
    inverse_std = libdevice.rsqrt(variance + 1.0e-5)
    weight = tl.load(weight_ptr + columns).to(tl.float32)
    bias = tl.load(bias_ptr + columns).to(tl.float32)
    normalized = weight * (inverse_std * (values - mean)) + bias
    if MASK_NORM_OUTPUT and not ALL_VALID:
        normalized = tl.where(valid, normalized, 0.0)
    tl.store(norm_output_ptr + offsets, normalized)


def residual_layer_norm(
    residual: torch.Tensor,
    branch: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    valid_token_mask: torch.Tensor,
    *,
    all_valid: bool,
    store_residual: bool = True,
    mask_norm_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if residual.shape != branch.shape or residual.shape[-1] != MODEL:
        raise ValueError("residual fusion expects equal tensors with D=1024")
    if residual.dtype != torch.float16 or not residual.is_cuda:
        raise ValueError("residual fusion expects CUDA float16")
    rows = residual.numel() // MODEL
    residual_output = torch.empty_like(residual) if store_residual else residual
    norm_output = torch.empty_like(residual)
    _residual_layer_norm_kernel[(rows,)](
        residual,
        branch,
        residual_output,
        norm_output,
        weight,
        bias,
        valid_token_mask,
        ALL_VALID=all_valid,
        STORE_RESIDUAL=store_residual,
        MASK_NORM_OUTPUT=mask_norm_output,
        ALGORITHM=NORM_MODE,
        num_warps=NUM_WARPS,
        num_stages=1,
    )
    return residual_output, norm_output


def fusion_tuning() -> dict[str, int]:
    return {"norm_warps": NUM_WARPS, "norm_mode": NORM_MODE}
