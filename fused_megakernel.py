"""Single-launch four-layer Hopper transformer megakernel."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


SEQUENCE = 128
MODEL = 128
DEFAULT_HEADS = 4
LAYERS = 4
ELEMENTS = SEQUENCE * MODEL
WORKSPACE_SLOTS = 3
CLUSTER_WORKSPACE_SLOTS = 4
TRACE_STAGES_PER_LAYER = 9
TRACE_SLOTS = (
    LAYERS * TRACE_STAGES_PER_LAYER
    + 1
    + 3 * DEFAULT_HEADS * LAYERS
    + LAYERS
)

LAYER_STRIDE = 4 * MODEL + 6 * (MODEL * MODEL + MODEL)
NORM1_WEIGHT = 0
NORM1_BIAS = NORM1_WEIGHT + MODEL
Q_WEIGHT = NORM1_BIAS + MODEL
Q_BIAS = Q_WEIGHT + ELEMENTS
K_WEIGHT = Q_BIAS + MODEL
K_BIAS = K_WEIGHT + ELEMENTS
V_WEIGHT = K_BIAS + MODEL
V_BIAS = V_WEIGHT + ELEMENTS
OUT_WEIGHT = V_BIAS + MODEL
OUT_BIAS = OUT_WEIGHT + ELEMENTS
NORM2_WEIGHT = OUT_BIAS + MODEL
NORM2_BIAS = NORM2_WEIGHT + MODEL
FFN_IN_WEIGHT = NORM2_BIAS + MODEL
FFN_IN_BIAS = FFN_IN_WEIGHT + ELEMENTS
FFN_OUT_WEIGHT = FFN_IN_BIAS + MODEL
FFN_OUT_BIAS = FFN_OUT_WEIGHT + ELEMENTS

JIT_NORM1_WEIGHT = tl.constexpr(NORM1_WEIGHT)
JIT_NORM1_BIAS = tl.constexpr(NORM1_BIAS)
JIT_Q_WEIGHT = tl.constexpr(Q_WEIGHT)
JIT_Q_BIAS = tl.constexpr(Q_BIAS)
JIT_K_WEIGHT = tl.constexpr(K_WEIGHT)
JIT_K_BIAS = tl.constexpr(K_BIAS)
JIT_V_WEIGHT = tl.constexpr(V_WEIGHT)
JIT_V_BIAS = tl.constexpr(V_BIAS)
JIT_OUT_WEIGHT = tl.constexpr(OUT_WEIGHT)
JIT_OUT_BIAS = tl.constexpr(OUT_BIAS)
JIT_NORM2_WEIGHT = tl.constexpr(NORM2_WEIGHT)
JIT_NORM2_BIAS = tl.constexpr(NORM2_BIAS)
JIT_FFN_IN_WEIGHT = tl.constexpr(FFN_IN_WEIGHT)
JIT_FFN_IN_BIAS = tl.constexpr(FFN_IN_BIAS)
JIT_FFN_OUT_WEIGHT = tl.constexpr(FFN_OUT_WEIGHT)
JIT_FFN_OUT_BIAS = tl.constexpr(FFN_OUT_BIAS)


def _environment_integer(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _environment_is_set(name: str) -> bool:
    return name in os.environ


# Internal tuning switches make it possible to compare numerical/performance
# trade-offs in isolated Slurm processes without maintaining source variants.
NUM_CTAS = _environment_integer("TTTJ_NUM_CTAS", 1)
# Triton's multi-CTA conversion cannot lower the rank-changing splits used by
# the one-CTA reduction trees. Clustered launches select equivalent split-free
# trees below.
NORM_MODE = _environment_integer("TTTJ_NORM_MODE", 5 if NUM_CTAS > 1 else 0)
FINAL_NORM_MODE = _environment_integer("TTTJ_FINAL_NORM_MODE", NORM_MODE)
SOFTMAX_MODE = _environment_integer(
    "TTTJ_SOFTMAX_MODE", 1 if NUM_CTAS > 1 else 0
)
DIVISION_MODE = _environment_integer("TTTJ_DIVISION_MODE", 4)
EXP_MODE = _environment_integer("TTTJ_EXP_MODE", 0)
PV_MODE = _environment_integer("TTTJ_PV_MODE", 1)
JIT_PV_MODE = tl.constexpr(PV_MODE)
GELU_MODE = _environment_integer("TTTJ_GELU_MODE", 0)
CAUSAL_SKIP = bool(_environment_integer("TTTJ_CAUSAL_SKIP", 0))
NUM_WARPS = _environment_integer("TTTJ_NUM_WARPS", 4)
NUM_STAGES = _environment_integer("TTTJ_NUM_STAGES", 3)
RESIDENT_ASSIST = bool(_environment_integer("TTTJ_RESIDENT_ASSIST", 1))
RESIDENT_SPLIT_Q = bool(_environment_integer("TTTJ_RESIDENT_SPLIT_Q", 1))
LINEAR_REDUCTION_TILE = _environment_integer("TTTJ_LINEAR_K", 64)
ATTENTION_REDUCTION_TILE = _environment_integer("TTTJ_ATTENTION_K", 32)
LINEAR_ROW_TILE = _environment_integer("TTTJ_LINEAR_M", 64)
ATTENTION_ROW_TILE = _environment_integer("TTTJ_ATTENTION_M", 64)
NORM_ROW_TILE = _environment_integer("TTTJ_NORM_M", 64)
# Q overwrites its normalized input after K/V are produced, so the linear
# output must cover the full row in one tile before that alias is safe.
LINEAR_OUTPUT_TILE = 128
ALL_VALID_TOKENS = bool(_environment_integer("TTTJ_ALL_VALID", 0))
EXPLICIT_BARRIERS = bool(_environment_integer("TTTJ_BARRIERS", 0))
JIT_CLUSTERED = tl.constexpr(NUM_CTAS > 1)
JIT_CLUSTER_THREADS = tl.constexpr(NUM_WARPS * 32)
ASSIST_SEQUENCES = tl.constexpr(64)


def resolved_megakernel_tuning(
    batch_size: int,
    num_heads: int,
) -> dict[str, int]:
    """Resolve shape defaults while retaining environment tuning overrides."""
    step_3_shape = batch_size == 64 and NUM_CTAS == 1
    num_warps = NUM_WARPS
    linear_m = LINEAR_ROW_TILE
    attention_m = ATTENTION_ROW_TILE
    attention_k = ATTENTION_REDUCTION_TILE
    norm_m = NORM_ROW_TILE
    num_stages = NUM_STAGES
    if step_3_shape:
        if not _environment_is_set("TTTJ_NUM_WARPS"):
            num_warps = 8
        if not _environment_is_set("TTTJ_LINEAR_M"):
            linear_m = 128
        if not _environment_is_set("TTTJ_ATTENTION_M"):
            attention_m = 128
        if not _environment_is_set("TTTJ_ATTENTION_K"):
            attention_k = 64 if num_heads == 2 else 32
    if batch_size >= 10000 and not _environment_is_set("TTTJ_NUM_STAGES"):
        num_stages = 2
    return {
        "num_warps": num_warps,
        "num_stages": num_stages,
        "linear_m": linear_m,
        "linear_k": LINEAR_REDUCTION_TILE,
        "attention_m": attention_m,
        "attention_k": attention_k,
        "norm_m": norm_m,
    }


@triton.jit
def _cluster_sync():
    # Core Triton does not expose Hopper cluster barriers. The tensor operand
    # forces every warp thread to execute the PTX; a scalar inline-asm result
    # is otherwise assigned to only one thread. Release/acquire orders the
    # global-memory hand-offs, and the memory clobber pins compiler scheduling.
    tl.debug_barrier()
    participants = tl.arange(0, JIT_CLUSTER_THREADS)
    _ = tl.inline_asm_elementwise(
        "barrier.cluster.arrive.release; "
        "barrier.cluster.wait.acquire; "
        "mov.u32 $0, $1;",
        "=r,r,~{memory}",
        [participants],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tl.debug_barrier()


@triton.jit
def _resident_publish(THREADS: tl.constexpr):
    """Publish cooperative stores to another CTA in the same sequence DAG."""
    tl.debug_barrier()
    participants = tl.arange(0, THREADS)
    _ = tl.inline_asm_elementwise(
        "membar.gl; mov.u32 $0, $1;",
        "=r,r,~{memory}",
        [participants],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tl.debug_barrier()


@triton.jit
def _resident_acquire(pointer):
    return tl.atomic_add(pointer, 0, sem="acquire", scope="gpu")


@triton.jit
def _stage_barrier(CLUSTERED: tl.constexpr):
    if CLUSTERED:
        _cluster_sync()
    else:
        tl.debug_barrier()


@triton.jit
def _trace_tensor(
    source_ptr,
    trace_ptr,
    slot,
    E: tl.constexpr,
    CAPTURE: tl.constexpr,
):
    if CAPTURE:
        offsets = tl.arange(0, 256)
        for start in range(0, E, 256):
            tl.store(
                trace_ptr + slot * E + start + offsets,
                tl.load(source_ptr + start + offsets),
            )


@triton.jit
def _welford_online(value, mean, sigma2, count):
    delta = value - mean
    new_count = count + 1.0
    new_mean = mean + delta * (1.0 / new_count)
    new_sigma2 = sigma2 + delta * (value - new_mean)
    return new_mean, new_sigma2, new_count


@triton.jit
def _welford_reduce(
    mean_a,
    sigma2_a,
    count_a,
    mean_b,
    sigma2_b,
    count_b,
):
    delta = mean_b - mean_a
    new_count = count_a + count_b
    coefficient = 1.0 / new_count
    fraction_a = count_a * coefficient
    fraction_b = count_b * coefficient
    new_mean = fraction_a * mean_a + fraction_b * mean_b
    new_sigma2 = (
        sigma2_a
        + sigma2_b
        + delta * delta * count_a * fraction_b
    )
    return new_mean, new_sigma2, new_count


@triton.jit
def _cluster_warp_add_halves(values, WIDTH: tl.constexpr):
    halves = tl.permute(
        tl.reshape(values, (128, 2, WIDTH // 2)), (0, 2, 1)
    )
    pair = tl.arange(0, 2)
    lower = tl.sum(
        tl.where(pair[None, None, :] == 0, halves, 0.0), axis=2
    )
    upper = tl.sum(
        tl.where(pair[None, None, :] == 1, halves, 0.0), axis=2
    )
    return lower + upper


@triton.jit
def _welford_combine_halves(
    mean,
    sigma2,
    count,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    # Match cuWelfordCombine(wd, shfl_down(wd)): dataB is the lower
    # half and dataA is the upper half at each warp-reduction step.
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
def _warp_add_halves(
    values,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    halves = tl.permute(
        tl.reshape(values, (ROWS, 2, WIDTH // 2)), (0, 2, 1)
    )
    lower, upper = tl.split(halves)
    return lower + upper


@triton.jit
def _layer_norm_half(
    input_ptr,
    output_ptr,
    copy_ptr,
    weight_ptr,
    bias_ptr,
    row_start,
    D: tl.constexpr,
    ROWS: tl.constexpr,
    NORM_ALGORITHM: tl.constexpr,
    COPY_INPUT: tl.constexpr,
):
    rows = row_start + tl.arange(0, ROWS)
    columns = tl.arange(0, D)
    offsets = rows[:, None] * D + columns[None, :]
    values = tl.load(input_ptr + offsets).to(tl.float32)
    if COPY_INPUT:
        tl.store(copy_ptr + offsets, values)

    if NORM_ALGORITHM == 0:
        # PyTorch's aligned FP16 LayerNorm assigns one half4 to each lane and
        # reduces the 32 per-lane Welford accumulators with shfl_down.
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
        variance = tl.reshape(sigma2, (ROWS,)) / D
    elif NORM_ALGORITHM == 1:
        mean = tl.sum(values, axis=1) / D
        centered = values - mean[:, None]
        variance = tl.sum(centered * centered, axis=1) / D
    elif NORM_ALGORITHM == 2:
        mean = tl.sum(values, axis=1) / D
        mean_square = tl.sum(values * values, axis=1) / D
        variance = tl.maximum(mean_square - mean * mean, 0.0)
    elif NORM_ALGORITHM == 3:
        sigma2 = tl.zeros_like(values)
        count = tl.full(values.shape, 1.0, tl.float32)
        mean, sigma2, count = tl.reduce(
            (values, sigma2, count),
            axis=1,
            combine_fn=_welford_reduce,
        )
        variance = sigma2 / count
    elif NORM_ALGORITHM == 4:
        values = values.to(tl.float64)
        mean = tl.sum(values, axis=1) / D
        centered = values - mean[:, None]
        variance = tl.sum(centered * centered, axis=1) / D
    else:
        # Match PyTorch's half4-per-lane accumulation without rank-changing
        # splits, which Triton's multi-CTA layout conversion cannot lower.
        lanes = tl.arange(0, 32)
        lane_offsets = rows[:, None] * D + 4 * lanes[None, :]
        mean = tl.zeros((ROWS, 32), dtype=tl.float32)
        sigma2 = tl.zeros((ROWS, 32), dtype=tl.float32)
        count = tl.zeros((ROWS, 32), dtype=tl.float32)
        for item in range(0, 4):
            lane_value = tl.load(input_ptr + lane_offsets + item).to(
                tl.float32
            )
            mean, sigma2, count = _welford_online(
                lane_value, mean, sigma2, count
            )
        mean, sigma2, count = tl.reduce(
            (mean, sigma2, count),
            axis=1,
            combine_fn=_welford_reduce,
        )
        variance = sigma2 / count
    inverse_std = libdevice.rsqrt(variance + 1.0e-5)
    if JIT_CLUSTERED:
        # Form the broadcast in pointer space so clustered row layouts
        # replicate these vectors with ordinary global loads. Loading a 1-D
        # vector first makes Triton 3.7 emit an invalid `nvvm.mapa` conversion.
        parameter_offsets = tl.zeros((ROWS, 1), tl.int32) + columns[None, :]
        weight = tl.load(weight_ptr + parameter_offsets).to(tl.float32)
        bias = tl.load(bias_ptr + parameter_offsets).to(tl.float32)
    else:
        weight = tl.load(weight_ptr + columns)[None, :].to(tl.float32)
        bias = tl.load(bias_ptr + columns)[None, :].to(tl.float32)
    normalized = inverse_std[:, None] * (values - mean[:, None])
    tl.store(output_ptr + offsets, weight * normalized + bias)


@triton.jit(noinline=NUM_CTAS > 1)
def _linear_tile(
    input_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    valid_ptr,
    row_start,
    column_start,
    D: tl.constexpr,
    EPILOGUE: tl.constexpr,
    GELU_ALGORITHM: tl.constexpr,
    REDUCTION_TILE: tl.constexpr,
    ROW_TILE: tl.constexpr,
    OUTPUT_TILE: tl.constexpr,
    ALL_VALID: tl.constexpr,
):
    rows = row_start + tl.arange(0, ROW_TILE)
    columns = column_start + tl.arange(0, OUTPUT_TILE)
    reductions = tl.arange(0, REDUCTION_TILE)
    accumulator = tl.zeros((ROW_TILE, OUTPUT_TILE), dtype=tl.float32)
    for reduction_start in range(0, D, REDUCTION_TILE):
        input_tile = tl.load(
            input_ptr
            + rows[:, None] * D
            + reduction_start
            + reductions[None, :]
        )
        weight_tile = tl.load(
            weight_ptr
            + columns[None, :] * D
            + reduction_start
            + reductions[:, None]
        )
        accumulator = tl.dot(input_tile, weight_tile, accumulator)
    accumulator += tl.load(bias_ptr + columns)[None, :]

    if EPILOGUE == 1:
        rounded = accumulator.to(tl.float16).to(tl.float32)
        if GELU_ALGORITHM == 0:
            output = 0.5 * rounded * (
                1.0 + libdevice.erf(rounded * 0.7071067811865475)
            )
        elif GELU_ALGORITHM < 3:
            inner = 0.7978845608028654 * (
                rounded + 0.044715 * rounded * rounded * rounded
            )
            if GELU_ALGORITHM == 1:
                activation = libdevice.tanh(inner)
            else:
                activation = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
            output = 0.5 * rounded * (1.0 + activation)
        else:
            # Hastings' single-precision erf approximation is substantially
            # cheaper than libdevice erf and is accurate to about 1.5e-7.
            argument = tl.abs(rounded) * 0.7071067811865475
            reciprocal = 1.0 / (1.0 + 0.3275911 * argument)
            polynomial = 1.061405429 * reciprocal - 1.453152027
            polynomial = polynomial * reciprocal + 1.421413741
            polynomial = polynomial * reciprocal - 0.284496736
            polynomial = polynomial * reciprocal + 0.254829592
            erf_absolute = 1.0 - (
                polynomial
                * reciprocal
                * tl.exp(-(argument * argument))
            )
            erf_value = tl.where(rounded >= 0.0, erf_absolute, -erf_absolute)
            output = 0.5 * rounded * (1.0 + erf_value)
    elif EPILOGUE == 2:
        branch = accumulator.to(tl.float16)
        if not ALL_VALID:
            valid = tl.load(valid_ptr + rows)[:, None]
            branch = tl.where(valid, branch, 0.0)
        residual = tl.load(
            residual_ptr + rows[:, None] * D + columns[None, :]
        )
        output = residual + branch
    elif EPILOGUE == 3:
        branch = accumulator.to(tl.float16)
        residual = tl.load(
            residual_ptr + rows[:, None] * D + columns[None, :]
        )
        if ALL_VALID:
            output = residual + branch
        else:
            valid = tl.load(valid_ptr + rows)[:, None]
            output = tl.where(valid, residual + branch, 0.0)
    else:
        output = accumulator

    tl.store(
        output_ptr + rows[:, None] * D + columns[None, :],
        output,
    )


@triton.jit
def _attention_mxhd(
    q_ptr,
    k_ptr,
    v_ptr,
    context_ptr,
    valid_ptr,
    score_trace_ptr,
    probability_trace_ptr,
    numerator_trace_ptr,
    denominator_trace_ptr,
    row_start,
    head,
    S: tl.constexpr,
    D: tl.constexpr,
    HD: tl.constexpr,
    K: tl.constexpr,
    SCALE: tl.constexpr,
    CAPTURE: tl.constexpr,
    SOFTMAX_ALGORITHM: tl.constexpr,
    DIVISION_ALGORITHM: tl.constexpr,
    EXP_ALGORITHM: tl.constexpr,
    REDUCTION_TILE: tl.constexpr,
    ROW_TILE: tl.constexpr,
    ALL_VALID: tl.constexpr,
):
    queries = row_start + tl.arange(0, ROW_TILE)
    keys = tl.arange(0, K)
    reductions = tl.arange(0, REDUCTION_TILE)
    scores = tl.zeros((ROW_TILE, K), dtype=tl.float32)
    for reduction_start in range(0, HD, REDUCTION_TILE):
        query_tile = tl.load(
            q_ptr
            + queries[:, None] * D
            + head * HD
            + reduction_start
            + reductions[None, :]
        )
        key_tile = tl.load(
            k_ptr
            + keys[None, :] * D
            + head * HD
            + reduction_start
            + reductions[:, None]
        )
        scores = tl.dot(query_tile, key_tile, scores)

    # Preserve the reference's FP16 QK output boundary and FP32 scalar scale.
    scores = scores.to(tl.float16).to(tl.float32) * SCALE
    if ALL_VALID:
        scores = tl.where(
            keys[None, :] <= queries[:, None], scores, -float("inf")
        )
    else:
        valid_keys = tl.load(valid_ptr + keys)[None, :]
        scores = tl.where(
            (keys[None, :] <= queries[:, None]) & valid_keys,
            scores,
            -float("inf"),
        )
    # PyTorch stores the scaled score tensor in FP16 before softmax.float().
    scores = scores.to(tl.float16).to(tl.float32)
    if CAPTURE:
        tl.store(
            score_trace_ptr
            + head * S * S
            + queries[:, None] * S
            + keys[None, :],
            scores,
        )
    maximum = tl.max(scores, axis=1)
    if EXP_ALGORITHM == 0:
        numerator = libdevice.exp(scores - maximum[:, None])
    else:
        numerator = tl.exp(scores - maximum[:, None])
    if CAPTURE:
        tl.store(
            numerator_trace_ptr
            + head * S * S
            + queries[:, None] * S
            + keys[None, :],
            numerator,
        )
    if SOFTMAX_ALGORITHM == 0:
        # PersistentSoftmax assigns K / 32 columns to each warp lane, then
        # performs a shuffle-xor tree at offsets 16, 8, 4, 2, and 1.
        if K == 128:
            lane_groups = tl.reshape(numerator, (ROW_TILE, 4, 32))
            lane_groups = tl.permute(lane_groups, (0, 2, 1))
            lane_pairs = tl.reshape(lane_groups, (ROW_TILE, 32, 2, 2))
            even, odd = tl.split(lane_pairs)
            item0, item2 = tl.split(even)
            item1, item3 = tl.split(odd)
            lane_sum = item0 + item1
            lane_sum += item2
            lane_sum += item3
        elif K == 64:
            lane_groups = tl.reshape(numerator, (ROW_TILE, 2, 32))
            lane_groups = tl.permute(lane_groups, (0, 2, 1))
            item0, item1 = tl.split(lane_groups)
            lane_sum = item0 + item1
        else:
            # S=32 step-4 attention assigns one complete reduction row to a
            # warp.  Keeping the same shuffle-style tree as PyTorch's
            # persistent softmax preserves its observable FP16 boundary.
            lane_sum = numerator
        denominator = _warp_add_halves(lane_sum, ROW_TILE, 32)
        denominator = _warp_add_halves(denominator, ROW_TILE, 16)
        denominator = _warp_add_halves(denominator, ROW_TILE, 8)
        denominator = _warp_add_halves(denominator, ROW_TILE, 4)
        denominator = _warp_add_halves(denominator, ROW_TILE, 2)
        denominator = tl.reshape(denominator, (ROW_TILE,))
    else:
        denominator = tl.sum(numerator, axis=1)
    if CAPTURE:
        tl.store(denominator_trace_ptr + head * S + queries, denominator)
    if DIVISION_ALGORITHM == 0:
        # NVCC emits IEEE round-to-nearest division for PersistentSoftmax.cuh;
        # applying it elementwise is expensive but bit-exact.
        probabilities = tl.div_rn(numerator, denominator[:, None])
    elif DIVISION_ALGORITHM == 1:
        # One accurate reciprocal per row is much cheaper than K divisions.
        inverse_denominator = tl.div_rn(1.0, denominator)
        probabilities = numerator * inverse_denominator[:, None]
    elif DIVISION_ALGORITHM == 2:
        probabilities = numerator / denominator[:, None]
    elif DIVISION_ALGORITHM == 3:
        inverse_denominator = 1.0 / denominator
        inverse_denominator *= 2.0 - denominator * inverse_denominator
        probabilities = numerator * inverse_denominator[:, None]
    else:
        # Correct a reciprocal-multiply quotient with its fused residual. This
        # approaches correctly rounded division while retaining one division
        # per row instead of one per probability.
        inverse_denominator = tl.div_rn(1.0, denominator)
        probabilities = numerator * inverse_denominator[:, None]
        remainder = libdevice.fma_rn(
            -probabilities,
            denominator[:, None],
            numerator,
        )
        probabilities += remainder * inverse_denominator[:, None]
    probabilities = probabilities.to(tl.float16)
    if CAPTURE:
        tl.store(
            probability_trace_ptr
            + head * S * S
            + queries[:, None] * S
            + keys[None, :],
            probabilities,
        )

    # HD is 32, 64, or 128 for benchmark cases 1, 10, and 9.  Emitting the
    # whole head in one dot gives Hopper a natural WGMMA N tile and avoids
    # repeating the probability load/softmax for multiple 32-column slices.
    value_columns = tl.arange(0, HD)
    value_tile = tl.load(
        v_ptr
        + keys[:, None] * D
        + head * HD
        + value_columns[None, :]
    )
    if JIT_PV_MODE == 1 and K == 32:
        # cuBLAS's S32 strided-batched GEMM accumulates two K16 fragments.
        # Make that boundary explicit rather than letting WGMMA select a
        # fused K32 schedule whose final FP16 rounding differs by rare ULPs.
        probability_halves = tl.permute(
            tl.reshape(probabilities, (ROW_TILE, 2, 16)), (0, 2, 1)
        )
        probability0, probability1 = tl.split(probability_halves)
        value_halves = tl.permute(
            tl.reshape(value_tile, (2, 16, HD)), (1, 2, 0)
        )
        value0, value1 = tl.split(value_halves)
        context = tl.dot(probability0, value0)
        context += tl.dot(probability1, value1)
    else:
        context = tl.dot(probabilities, value_tile)
    tl.store(
        context_ptr
        + queries[:, None] * D
        + head * HD
        + value_columns[None, :],
        context,
    )


@triton.jit(noinline=True)
def _cluster_attention_scores_128x64(
    q_ptr,
    k_ptr,
    score_scratch_ptr,
    valid_ptr,
    score_trace_ptr,
    head,
    key_start,
    S: tl.constexpr,
    D: tl.constexpr,
    HD: tl.constexpr,
    SCALE: tl.constexpr,
    CAPTURE: tl.constexpr,
    REDUCTION_TILE: tl.constexpr,
    ALL_VALID: tl.constexpr,
):
    # A 128x64 result makes PlanCTA choose a 2x1 row split: each CTA owns 64
    # complete softmax rows instead of half of every reduction row.
    queries = tl.arange(0, 128)
    keys = key_start + tl.arange(0, 64)
    reductions = tl.arange(0, REDUCTION_TILE)
    scores = tl.zeros((128, 64), dtype=tl.float32)
    for reduction_start in range(0, HD, REDUCTION_TILE):
        query_tile = tl.load(
            q_ptr
            + queries[:, None] * D
            + head * HD
            + reduction_start
            + reductions[None, :]
        )
        key_tile = tl.load(
            k_ptr
            + keys[None, :] * D
            + head * HD
            + reduction_start
            + reductions[:, None]
        )
        scores = tl.dot(query_tile, key_tile, scores)

    scores = scores.to(tl.float16).to(tl.float32) * SCALE
    causal = keys[None, :] <= queries[:, None]
    if not ALL_VALID:
        causal &= tl.load(valid_ptr + keys)[None, :]
    scores = tl.where(causal, scores, -float("inf"))
    scores = scores.to(tl.float16)
    offsets = queries[:, None] * S + keys[None, :]
    tl.store(score_scratch_ptr + offsets, scores)
    if CAPTURE:
        tl.store(score_trace_ptr + head * S * S + offsets, scores)


@triton.jit(noinline=True)
def _cluster_attention_softmax_128(
    score_probability_ptr,
    probability_trace_ptr,
    numerator_trace_ptr,
    denominator_trace_ptr,
    head,
    S: tl.constexpr,
    CAPTURE: tl.constexpr,
    DIVISION_ALGORITHM: tl.constexpr,
    EXP_ALGORITHM: tl.constexpr,
):
    queries = tl.arange(0, 128)
    keys = tl.arange(0, 128)
    offsets = queries[:, None] * S + keys[None, :]
    scores = tl.load(score_probability_ptr + offsets).to(tl.float32)
    maximum = tl.max(scores, axis=1)
    if EXP_ALGORITHM == 0:
        numerator = libdevice.exp(scores - maximum[:, None])
    else:
        numerator = tl.exp(scores - maximum[:, None])
    lane_groups = tl.permute(
        tl.reshape(numerator, (128, 4, 32)), (0, 2, 1)
    )
    items = tl.arange(0, 4)
    item0 = tl.sum(
        tl.where(items[None, None, :] == 0, lane_groups, 0.0), axis=2
    )
    item1 = tl.sum(
        tl.where(items[None, None, :] == 1, lane_groups, 0.0), axis=2
    )
    item2 = tl.sum(
        tl.where(items[None, None, :] == 2, lane_groups, 0.0), axis=2
    )
    item3 = tl.sum(
        tl.where(items[None, None, :] == 3, lane_groups, 0.0), axis=2
    )
    lane_sum = item0 + item1
    lane_sum += item2
    lane_sum += item3
    denominator = _cluster_warp_add_halves(lane_sum, 32)
    denominator = _cluster_warp_add_halves(denominator, 16)
    denominator = _cluster_warp_add_halves(denominator, 8)
    denominator = _cluster_warp_add_halves(denominator, 4)
    denominator = _cluster_warp_add_halves(denominator, 2)
    denominator = tl.reshape(denominator, (128,))

    if DIVISION_ALGORITHM == 0:
        probabilities = tl.div_rn(numerator, denominator[:, None])
    elif DIVISION_ALGORITHM == 1:
        inverse_denominator = tl.div_rn(1.0, denominator)
        probabilities = numerator * inverse_denominator[:, None]
    elif DIVISION_ALGORITHM == 2:
        probabilities = numerator / denominator[:, None]
    elif DIVISION_ALGORITHM == 3:
        inverse_denominator = 1.0 / denominator
        inverse_denominator *= 2.0 - denominator * inverse_denominator
        probabilities = numerator * inverse_denominator[:, None]
    else:
        inverse_denominator = tl.div_rn(1.0, denominator)
        probabilities = numerator * inverse_denominator[:, None]
        remainder = libdevice.fma_rn(
            -probabilities,
            denominator[:, None],
            numerator,
        )
        probabilities += remainder * inverse_denominator[:, None]
    probabilities = probabilities.to(tl.float16)
    tl.store(score_probability_ptr + offsets, probabilities)
    if CAPTURE:
        tl.store(numerator_trace_ptr + head * S * S + offsets, numerator)
        tl.store(denominator_trace_ptr + head * S + queries, denominator)
        tl.store(
            probability_trace_ptr + head * S * S + offsets,
            probabilities,
        )


@triton.jit(noinline=True)
def _cluster_attention_context_128x32(
    probability_ptr,
    v_ptr,
    context_ptr,
    head,
    S: tl.constexpr,
    D: tl.constexpr,
    HD: tl.constexpr,
):
    queries = tl.arange(0, 128)
    keys = tl.arange(0, 128)
    probabilities = tl.load(
        probability_ptr + queries[:, None] * S + keys[None, :]
    )
    value_columns = tl.arange(0, 32)
    value_tile = tl.load(
        v_ptr
        + keys[:, None] * D
        + head * HD
        + value_columns[None, :]
    )
    context = tl.dot(probabilities, value_tile)
    tl.store(
        context_ptr
        + queries[:, None] * D
        + head * HD
        + value_columns[None, :],
        context,
    )


@triton.jit
def _cluster_attention_128x32(
    q_ptr,
    k_ptr,
    v_ptr,
    context_ptr,
    scratch_ptr,
    valid_ptr,
    score_trace_ptr,
    probability_trace_ptr,
    numerator_trace_ptr,
    denominator_trace_ptr,
    head,
    S: tl.constexpr,
    D: tl.constexpr,
    HD: tl.constexpr,
    SCALE: tl.constexpr,
    CAPTURE: tl.constexpr,
    DIVISION_ALGORITHM: tl.constexpr,
    EXP_ALGORITHM: tl.constexpr,
    REDUCTION_TILE: tl.constexpr,
    ALL_VALID: tl.constexpr,
):
    for key_start in range(0, S, 64):
        _cluster_attention_scores_128x64(
            q_ptr,
            k_ptr,
            scratch_ptr,
            valid_ptr,
            score_trace_ptr,
            head,
            key_start,
            S,
            D,
            HD,
            SCALE,
            CAPTURE,
            REDUCTION_TILE,
            ALL_VALID,
        )
    _cluster_sync()
    _cluster_attention_softmax_128(
        scratch_ptr,
        probability_trace_ptr,
        numerator_trace_ptr,
        denominator_trace_ptr,
        head,
        S,
        CAPTURE,
        DIVISION_ALGORITHM,
        EXP_ALGORITHM,
    )
    _cluster_sync()
    _cluster_attention_context_128x32(
        scratch_ptr,
        v_ptr,
        context_ptr,
        head,
        S,
        D,
        HD,
    )
    _cluster_sync()


@triton.jit
def _transformer_megakernel(
    input_ptr,
    valid_ptr,
    packed_ptr,
    workspace_ptr,
    output_ptr,
    trace_ptr,
    scheduler_ptr,
    launch_epoch,
    S: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    HD: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    E: tl.constexpr,
    LAYER_WEIGHTS: tl.constexpr,
    SCALE: tl.constexpr,
    CAPTURE: tl.constexpr,
    NORM_ALGORITHM: tl.constexpr,
    FINAL_NORM_ALGORITHM: tl.constexpr,
    SOFTMAX_ALGORITHM: tl.constexpr,
    DIVISION_ALGORITHM: tl.constexpr,
    EXP_ALGORITHM: tl.constexpr,
    GELU_ALGORITHM: tl.constexpr,
    SKIP_CAUSAL_TILES: tl.constexpr,
    LINEAR_K: tl.constexpr,
    ATTENTION_K: tl.constexpr,
    ATTENTION_M: tl.constexpr,
    LINEAR_M: tl.constexpr,
    LINEAR_N: tl.constexpr,
    NORM_M: tl.constexpr,
    WORKSPACE_STRIDE: tl.constexpr,
    ALL_VALID: tl.constexpr,
    USE_BARRIERS: tl.constexpr,
    CLUSTERED: tl.constexpr,
    ASSIST: tl.constexpr,
    ASSIST_THREADS: tl.constexpr,
    SPLIT_ASSIST_Q: tl.constexpr,
):
    worker = tl.program_id(0)
    if ASSIST:
        sequence = worker // 2
        role = worker % 2
    else:
        sequence = worker
        role: tl.constexpr = 0
    input_base = input_ptr + sequence * E
    valid_base = valid_ptr + sequence * S
    workspace_base = workspace_ptr + sequence * WORKSPACE_STRIDE * E
    x = output_ptr + sequence * E
    norm = workspace_base
    if CLUSTERED:
        q = x
        k = norm + E
        v = k + E
        saved_residual = v + E
        probability_scratch = norm
        context = x
    else:
        k = norm + E
        v = k + E
        q = v + E if ASSIST else norm
        saved_residual = x
        probability_scratch = v + E
        context = q
    if ASSIST:
        start_epochs = scheduler_ptr
        norm_epochs = start_epochs + ASSIST_SEQUENCES
        k_epochs = norm_epochs + ASSIST_SEQUENCES
        v_epochs = k_epochs + ASSIST_SEQUENCES
        qkv_epochs = v_epochs + ASSIST_SEQUENCES
        attention_epochs = qkv_epochs + ASSIST_SEQUENCES
        layer_epochs = attention_epochs + ASSIST_SEQUENCES
        if role == 0:
            tl.atomic_xchg(
                norm_epochs + sequence, 0, sem="relaxed", scope="gpu"
            )
            tl.atomic_xchg(
                k_epochs + sequence, 0, sem="relaxed", scope="gpu"
            )
            tl.atomic_xchg(
                v_epochs + sequence, 0, sem="relaxed", scope="gpu"
            )
            tl.atomic_xchg(
                qkv_epochs + sequence, 0, sem="relaxed", scope="gpu"
            )
            tl.atomic_xchg(
                attention_epochs + sequence, 0, sem="relaxed", scope="gpu"
            )
            tl.atomic_xchg(
                layer_epochs + sequence, 0, sem="relaxed", scope="gpu"
            )
            tl.atomic_xchg(
                start_epochs + sequence,
                launch_epoch,
                sem="release",
                scope="gpu",
            )
        else:
            started = _resident_acquire(start_epochs + sequence)
            while started != launch_epoch:
                started = _resident_acquire(start_epochs + sequence)
    for layer in range(NUM_LAYERS):
        weights = packed_ptr + layer * LAYER_WEIGHTS
        residual = input_base if layer == 0 else x
        if CLUSTERED:
            attention_residual = saved_residual
        else:
            attention_residual = residual
        if role == 0:
            for row_start in range(0, S, NORM_M):
                _layer_norm_half(
                    residual,
                    norm,
                    saved_residual,
                    weights + JIT_NORM1_WEIGHT,
                    weights + JIT_NORM1_BIAS,
                    row_start,
                    D,
                    NORM_M,
                    NORM_ALGORITHM,
                    CLUSTERED,
                )
            if USE_BARRIERS or CLUSTERED:
                _stage_barrier(CLUSTERED)
            _trace_tensor(norm, trace_ptr, layer * 9, E, CAPTURE)
            if ASSIST:
                _resident_publish(ASSIST_THREADS)
                tl.atomic_xchg(
                    norm_epochs + sequence,
                    layer + 1,
                    sem="release",
                    scope="gpu",
                )

            for row_start in range(0, S, LINEAR_M):
                for column_start in range(0, D, LINEAR_N):
                    if not ASSIST:
                        _linear_tile(
                            norm,
                            k,
                            weights + JIT_K_WEIGHT,
                            weights + JIT_K_BIAS,
                            residual,
                            valid_base,
                            row_start,
                            column_start,
                            D,
                            0,
                            GELU_ALGORITHM,
                            LINEAR_K,
                            LINEAR_M,
                            LINEAR_N,
                            ALL_VALID,
                        )
                    _linear_tile(
                        norm,
                        v,
                        weights + JIT_V_WEIGHT,
                        weights + JIT_V_BIAS,
                        residual,
                        valid_base,
                        row_start,
                        column_start,
                        D,
                        0,
                        GELU_ALGORITHM,
                        LINEAR_K,
                        LINEAR_M,
                        LINEAR_N,
                        ALL_VALID,
                    )
                    if ASSIST:
                        if SPLIT_ASSIST_Q:
                            _resident_publish(ASSIST_THREADS)
                            tl.atomic_xchg(
                                v_epochs + sequence,
                                layer + 1,
                                sem="release",
                                scope="gpu",
                            )
                        k_ready = _resident_acquire(k_epochs + sequence)
                        while k_ready < layer + 1:
                            k_ready = _resident_acquire(k_epochs + sequence)
                    _linear_tile(
                        norm,
                        q,
                        weights + JIT_Q_WEIGHT,
                        weights + JIT_Q_BIAS,
                        residual,
                        valid_base,
                        row_start,
                        column_start,
                        D,
                        0,
                        GELU_ALGORITHM,
                        LINEAR_K,
                        LINEAR_M,
                        64 if SPLIT_ASSIST_Q else LINEAR_N,
                        ALL_VALID,
                    )
            if USE_BARRIERS or CLUSTERED:
                _stage_barrier(CLUSTERED)
            _trace_tensor(q, trace_ptr, layer * 9 + 1, E, CAPTURE)
            _trace_tensor(k, trace_ptr, layer * 9 + 2, E, CAPTURE)
            _trace_tensor(v, trace_ptr, layer * 9 + 3, E, CAPTURE)
            if ASSIST:
                _resident_publish(ASSIST_THREADS)
                if SPLIT_ASSIST_Q:
                    tl.atomic_add(
                        qkv_epochs + sequence, 1, sem="release", scope="gpu"
                    )
                    qkv_ready = _resident_acquire(qkv_epochs + sequence)
                    while qkv_ready < 2 * (layer + 1):
                        qkv_ready = _resident_acquire(qkv_epochs + sequence)
                else:
                    tl.atomic_xchg(
                        qkv_epochs + sequence,
                        layer + 1,
                        sem="release",
                        scope="gpu",
                    )
        elif ASSIST:
            norm_ready = _resident_acquire(norm_epochs + sequence)
            while norm_ready < layer + 1:
                norm_ready = _resident_acquire(norm_epochs + sequence)
            for row_start in range(0, S, LINEAR_M):
                for column_start in range(0, D, LINEAR_N):
                    _linear_tile(
                        norm,
                        k,
                        weights + JIT_K_WEIGHT,
                        weights + JIT_K_BIAS,
                        residual,
                        valid_base,
                        row_start,
                        column_start,
                        D,
                        0,
                        GELU_ALGORITHM,
                        LINEAR_K,
                        LINEAR_M,
                        LINEAR_N,
                        ALL_VALID,
                    )
            _resident_publish(ASSIST_THREADS)
            tl.atomic_xchg(
                k_epochs + sequence,
                layer + 1,
                sem="release",
                scope="gpu",
            )
            if SPLIT_ASSIST_Q:
                v_ready = _resident_acquire(v_epochs + sequence)
                while v_ready < layer + 1:
                    v_ready = _resident_acquire(v_epochs + sequence)
                for row_start in range(0, S, LINEAR_M):
                    _linear_tile(
                        norm,
                        q,
                        weights + JIT_Q_WEIGHT,
                        weights + JIT_Q_BIAS,
                        residual,
                        valid_base,
                        row_start,
                        64,
                        D,
                        0,
                        GELU_ALGORITHM,
                        LINEAR_K,
                        LINEAR_M,
                        64,
                        ALL_VALID,
                    )
                _resident_publish(ASSIST_THREADS)
                tl.atomic_add(
                    qkv_epochs + sequence, 1, sem="release", scope="gpu"
                )
                qkv_ready = _resident_acquire(qkv_epochs + sequence)
                while qkv_ready < 2 * (layer + 1):
                    qkv_ready = _resident_acquire(qkv_epochs + sequence)
            else:
                qkv_ready = _resident_acquire(qkv_epochs + sequence)
                while qkv_ready < layer + 1:
                    qkv_ready = _resident_acquire(qkv_epochs + sequence)

        if ASSIST and H > 1:
            first_head = role * (H // 2)
            owned_heads: tl.constexpr = H // 2
        else:
            first_head = 0
            owned_heads: tl.constexpr = H
        for head_offset in range(owned_heads):
            head = first_head + head_offset
            if CLUSTERED:
                _cluster_attention_128x32(
                    q,
                    k,
                    v,
                    context,
                    probability_scratch,
                    valid_base,
                    trace_ptr + (37 + layer * 8) * E,
                    trace_ptr + (41 + layer * 8) * E,
                    trace_ptr + (69 + layer * 4) * E,
                    trace_ptr + (85 + layer) * E,
                    head,
                    S,
                    D,
                    HD,
                    SCALE,
                    CAPTURE,
                    DIVISION_ALGORITHM,
                    EXP_ALGORITHM,
                    ATTENTION_K,
                    ALL_VALID,
                )
            else:
                if ASSIST and H == 1:
                    if role == 0:
                        _attention_mxhd(
                            q,
                            k,
                            v,
                            context,
                            valid_base,
                            trace_ptr + (37 + layer * 8) * E,
                            trace_ptr + (41 + layer * 8) * E,
                            trace_ptr + (69 + layer * 4) * E,
                            trace_ptr + (85 + layer) * E,
                            0,
                            0,
                            S,
                            D,
                            HD,
                            S,
                            SCALE,
                            CAPTURE,
                            SOFTMAX_ALGORITHM,
                            DIVISION_ALGORITHM,
                            EXP_ALGORITHM,
                            ATTENTION_K,
                            ATTENTION_M,
                            ALL_VALID,
                        )
                elif SKIP_CAUSAL_TILES:
                    _attention_mxhd(
                        q,
                        k,
                        v,
                        context,
                        valid_base,
                        trace_ptr + (37 + layer * 8) * E,
                        trace_ptr + (41 + layer * 8) * E,
                        trace_ptr + (69 + layer * 4) * E,
                        trace_ptr + (85 + layer) * E,
                        0,
                        head,
                        S,
                        D,
                        HD,
                        64,
                        SCALE,
                        CAPTURE,
                        SOFTMAX_ALGORITHM,
                        DIVISION_ALGORITHM,
                        EXP_ALGORITHM,
                        ATTENTION_K,
                        64,
                        ALL_VALID,
                    )
                    _attention_mxhd(
                        q,
                        k,
                        v,
                        context,
                        valid_base,
                        trace_ptr + (37 + layer * 8) * E,
                        trace_ptr + (41 + layer * 8) * E,
                        trace_ptr + (69 + layer * 4) * E,
                        trace_ptr + (85 + layer) * E,
                        64,
                        head,
                        S,
                        D,
                        HD,
                        S,
                        SCALE,
                        CAPTURE,
                        SOFTMAX_ALGORITHM,
                        DIVISION_ALGORITHM,
                        EXP_ALGORITHM,
                        ATTENTION_K,
                        64,
                        ALL_VALID,
                    )
                else:
                    for row_start in range(0, S, ATTENTION_M):
                        _attention_mxhd(
                            q,
                            k,
                            v,
                            context,
                            valid_base,
                            trace_ptr + (37 + layer * 8) * E,
                            trace_ptr + (41 + layer * 8) * E,
                            trace_ptr + (69 + layer * 4) * E,
                            trace_ptr + (85 + layer) * E,
                            row_start,
                            head,
                            S,
                            D,
                            HD,
                            S,
                            SCALE,
                            CAPTURE,
                            SOFTMAX_ALGORITHM,
                            DIVISION_ALGORITHM,
                            EXP_ALGORITHM,
                            ATTENTION_K,
                            ATTENTION_M,
                            ALL_VALID,
                        )
        if ASSIST:
            _resident_publish(ASSIST_THREADS)
            if role == 1:
                tl.atomic_xchg(
                    attention_epochs + sequence,
                    layer + 1,
                    sem="release",
                    scope="gpu",
                )
                layer_done = _resident_acquire(layer_epochs + sequence)
                while layer_done < layer + 1:
                    layer_done = _resident_acquire(layer_epochs + sequence)
            else:
                attention_done = _resident_acquire(
                    attention_epochs + sequence
                )
                while attention_done < layer + 1:
                    attention_done = _resident_acquire(
                        attention_epochs + sequence
                    )
        elif USE_BARRIERS and not CLUSTERED:
            _stage_barrier(CLUSTERED)
        if role == 0:
            _trace_tensor(context, trace_ptr, layer * 9 + 4, E, CAPTURE)

        if role == 0:
            for row_start in range(0, S, LINEAR_M):
                for column_start in range(0, D, LINEAR_N):
                    _linear_tile(
                        context,
                        k,
                        weights + JIT_OUT_WEIGHT,
                        weights + JIT_OUT_BIAS,
                        attention_residual,
                        valid_base,
                        row_start,
                        column_start,
                        D,
                        2,
                        GELU_ALGORITHM,
                        LINEAR_K,
                        LINEAR_M,
                        LINEAR_N,
                        ALL_VALID,
                    )
            if USE_BARRIERS or CLUSTERED:
                _stage_barrier(CLUSTERED)
            _trace_tensor(k, trace_ptr, layer * 9 + 5, E, CAPTURE)

            for row_start in range(0, S, NORM_M):
                _layer_norm_half(
                    k,
                    norm,
                    norm,
                    weights + JIT_NORM2_WEIGHT,
                    weights + JIT_NORM2_BIAS,
                    row_start,
                    D,
                    NORM_M,
                    NORM_ALGORITHM,
                    False,
                )
            if USE_BARRIERS or CLUSTERED:
                _stage_barrier(CLUSTERED)
            _trace_tensor(norm, trace_ptr, layer * 9 + 6, E, CAPTURE)

            for row_start in range(0, S, LINEAR_M):
                for column_start in range(0, D, LINEAR_N):
                    _linear_tile(
                        norm,
                        v,
                        weights + JIT_FFN_IN_WEIGHT,
                        weights + JIT_FFN_IN_BIAS,
                        norm,
                        valid_base,
                        row_start,
                        column_start,
                        D,
                        1,
                        GELU_ALGORITHM,
                        LINEAR_K,
                        LINEAR_M,
                        LINEAR_N,
                        ALL_VALID,
                    )
            if USE_BARRIERS or CLUSTERED:
                _stage_barrier(CLUSTERED)
            _trace_tensor(v, trace_ptr, layer * 9 + 7, E, CAPTURE)

            if CLUSTERED:
                ffn_output = norm if layer == NUM_LAYERS - 1 else x
            else:
                ffn_output = x
            for row_start in range(0, S, LINEAR_M):
                for column_start in range(0, D, LINEAR_N):
                    _linear_tile(
                        v,
                        ffn_output,
                        weights + JIT_FFN_OUT_WEIGHT,
                        weights + JIT_FFN_OUT_BIAS,
                        k,
                        valid_base,
                        row_start,
                        column_start,
                        D,
                        3,
                        GELU_ALGORITHM,
                        LINEAR_K,
                        LINEAR_M,
                        LINEAR_N,
                        ALL_VALID,
                    )
            if USE_BARRIERS or CLUSTERED:
                _stage_barrier(CLUSTERED)
            _trace_tensor(ffn_output, trace_ptr, layer * 9 + 8, E, CAPTURE)
            if ASSIST:
                tl.atomic_xchg(
                    layer_epochs + sequence,
                    layer + 1,
                    sem="release",
                    scope="gpu",
                )

    if role == 0:
        final_norm_weight = packed_ptr + NUM_LAYERS * LAYER_WEIGHTS
        final_norm_bias = final_norm_weight + D
        sequence_output = x
        final_input = norm if CLUSTERED else x
        if ALL_VALID:
            for row_start in range(0, S, NORM_M):
                _layer_norm_half(
                    final_input,
                    sequence_output,
                    sequence_output,
                    final_norm_weight,
                    final_norm_bias,
                    row_start,
                    D,
                    NORM_M,
                    FINAL_NORM_ALGORITHM,
                    False,
                )
            if CAPTURE:
                _stage_barrier(CLUSTERED)
                _trace_tensor(
                    sequence_output, trace_ptr, NUM_LAYERS * 9, E, CAPTURE
                )
        else:
            final_norm_scratch = k if CLUSTERED else norm
            for row_start in range(0, S, NORM_M):
                _layer_norm_half(
                    final_input,
                    final_norm_scratch,
                    final_norm_scratch,
                    final_norm_weight,
                    final_norm_bias,
                    row_start,
                    D,
                    NORM_M,
                    FINAL_NORM_ALGORITHM,
                    False,
                )
            if USE_BARRIERS or CLUSTERED:
                _stage_barrier(CLUSTERED)
            _trace_tensor(
                final_norm_scratch,
                trace_ptr,
                NUM_LAYERS * 9,
                E,
                CAPTURE,
            )
            offsets = tl.arange(0, 256)
            for start in range(0, E, 256):
                indices = start + offsets
                result = tl.load(final_norm_scratch + indices)
                if ALL_VALID:
                    tl.store(sequence_output + indices, result)
                else:
                    rows = indices // D
                    valid = tl.load(valid_base + rows)
                    tl.store(
                        sequence_output + indices,
                        tl.where(valid, result, 0.0),
                    )


def fused_megakernel_forward(
    value: torch.Tensor,
    valid_mask: torch.Tensor,
    packed_weights: torch.Tensor,
    *,
    num_heads: int = DEFAULT_HEADS,
    capture_trace: bool = False,
    all_valid: bool | None = None,
    scheduler: torch.Tensor | None = None,
    launch_epoch: int = 0,
):
    batch_size = value.shape[0]
    use_assist = RESIDENT_ASSIST and batch_size == 64 and num_heads in (2, 4)
    split_assist_q = use_assist and num_heads == 2 and RESIDENT_SPLIT_Q
    if all_valid is None:
        all_valid = ALL_VALID_TOKENS
    if num_heads not in (1, 2, 4):
        raise ValueError(f"supported head counts are 1, 2, and 4; got {num_heads}")
    head_dim = MODEL // num_heads
    if capture_trace and (batch_size != 1 or num_heads != DEFAULT_HEADS):
        raise ValueError("trace capture supports one sequence with four heads")
    if NUM_CTAS > 1 and num_heads != DEFAULT_HEADS:
        raise ValueError("the experimental clustered path supports four heads only")
    if use_assist and (NUM_CTAS > 1 or capture_trace):
        raise ValueError(
            "the resident attention-assist path supports only B=64, one-CTA, "
            "non-tracing launches"
        )
    if use_assist and (
        scheduler is None
        or scheduler.dtype != torch.int32
        or scheduler.numel() < 7 * 64
    ):
        raise ValueError("resident attention assist requires an int32 scheduler")
    tuning = resolved_megakernel_tuning(batch_size, num_heads)
    if SEQUENCE % tuning["linear_m"] != 0:
        raise ValueError("TTTJ_LINEAR_M must divide 128")
    if SEQUENCE % tuning["attention_m"] != 0:
        raise ValueError("TTTJ_ATTENTION_M must divide 128")
    if SEQUENCE % tuning["norm_m"] != 0:
        raise ValueError("TTTJ_NORM_M must divide 128")
    if (
        tuning["attention_k"] > head_dim
        or head_dim % tuning["attention_k"] != 0
    ):
        raise ValueError("TTTJ_ATTENTION_K must divide and not exceed head_dim")
    workspace_slots = (
        CLUSTER_WORKSPACE_SLOTS
        if NUM_CTAS > 1 or use_assist
        else WORKSPACE_SLOTS
    )
    workspace = torch.empty(
        (batch_size, workspace_slots, SEQUENCE, MODEL),
        device=value.device,
        dtype=value.dtype,
    )
    output = torch.empty_like(value)
    trace = (
        torch.empty(
            (TRACE_SLOTS, SEQUENCE, MODEL),
            device=value.device,
            dtype=torch.float32,
        )
        if capture_trace
        else workspace
    )
    scheduler_argument = scheduler if scheduler is not None else packed_weights
    grid = batch_size * 2 if use_assist else batch_size
    _transformer_megakernel[(grid,)](
        value,
        valid_mask,
        packed_weights,
        workspace,
        output,
        trace,
        scheduler_argument,
        launch_epoch,
        S=SEQUENCE,
        D=MODEL,
        H=num_heads,
        HD=head_dim,
        NUM_LAYERS=LAYERS,
        E=ELEMENTS,
        LAYER_WEIGHTS=LAYER_STRIDE,
        SCALE=head_dim ** -0.5,
        CAPTURE=capture_trace,
        NORM_ALGORITHM=NORM_MODE,
        FINAL_NORM_ALGORITHM=FINAL_NORM_MODE,
        SOFTMAX_ALGORITHM=SOFTMAX_MODE,
        DIVISION_ALGORITHM=DIVISION_MODE,
        EXP_ALGORITHM=EXP_MODE,
        GELU_ALGORITHM=GELU_MODE,
        SKIP_CAUSAL_TILES=CAUSAL_SKIP,
        LINEAR_K=tuning["linear_k"],
        ATTENTION_K=tuning["attention_k"],
        ATTENTION_M=tuning["attention_m"],
        LINEAR_M=tuning["linear_m"],
        LINEAR_N=LINEAR_OUTPUT_TILE,
        NORM_M=tuning["norm_m"],
        WORKSPACE_STRIDE=workspace_slots,
        ALL_VALID=all_valid,
        USE_BARRIERS=EXPLICIT_BARRIERS or capture_trace,
        CLUSTERED=NUM_CTAS > 1,
        ASSIST=use_assist,
        ASSIST_THREADS=tuning["num_warps"] * 32,
        SPLIT_ASSIST_Q=split_assist_q,
        num_warps=tuning["num_warps"],
        num_ctas=NUM_CTAS,
        num_stages=tuning["num_stages"],
    )
    return (output, trace) if capture_trace else output
