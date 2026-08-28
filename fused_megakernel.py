"""Single-launch four-layer Hopper transformer megakernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


SEQUENCE = 128
MODEL = 128
HEADS = 4
HEAD_DIM = 32
LAYERS = 4
ELEMENTS = SEQUENCE * MODEL
WORKSPACE_SLOTS = 6
TRACE_STAGES_PER_LAYER = 9
TRACE_SLOTS = (
    LAYERS * TRACE_STAGES_PER_LAYER
    + 1
    + 3 * HEADS * LAYERS
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

JIT_WORKSPACE_SLOTS = tl.constexpr(WORKSPACE_SLOTS)
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
    weight_ptr,
    bias_ptr,
    row_start,
    D: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = row_start + tl.arange(0, ROWS)
    columns = tl.arange(0, D)
    offsets = rows[:, None] * D + columns[None, :]
    values = tl.load(input_ptr + offsets).to(tl.float32)

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
    inverse_std = libdevice.rsqrt(variance + 1.0e-5)
    weight = tl.load(weight_ptr + columns)[None, :].to(tl.float32)
    bias = tl.load(bias_ptr + columns)[None, :].to(tl.float32)
    normalized = inverse_std[:, None] * (values - mean[:, None])
    tl.store(output_ptr + offsets, weight * normalized + bias)


@triton.jit
def _linear_64x64(
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
):
    rows = row_start + tl.arange(0, 64)
    columns = column_start + tl.arange(0, 64)
    reductions = tl.arange(0, 32)
    accumulator = tl.zeros((64, 64), dtype=tl.float32)
    for reduction_start in range(0, D, 32):
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
        output = 0.5 * rounded * (
            1.0 + libdevice.erf(rounded * 0.7071067811865475)
        )
    elif EPILOGUE == 2:
        branch = accumulator.to(tl.float16)
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
        valid = tl.load(valid_ptr + rows)[:, None]
        output = tl.where(valid, residual + branch, 0.0)
    else:
        output = accumulator

    tl.store(
        output_ptr + rows[:, None] * D + columns[None, :],
        output,
    )


@triton.jit
def _attention_64x32(
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
    SCALE: tl.constexpr,
    CAPTURE: tl.constexpr,
):
    queries = row_start + tl.arange(0, 64)
    keys = tl.arange(0, S)
    reductions = tl.arange(0, 16)
    scores = tl.zeros((64, S), dtype=tl.float32)
    for reduction_start in range(0, HD, 16):
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
    numerator = libdevice.exp(scores - maximum[:, None])
    if CAPTURE:
        tl.store(
            numerator_trace_ptr
            + head * S * S
            + queries[:, None] * S
            + keys[None, :],
            numerator,
        )
    # PyTorch's S=128 softmax assigns four columns to each warp lane, sums
    # those four sequentially, then performs a 32-lane warp reduction.
    lane_groups = tl.reshape(numerator, (64, 4, 32))
    lane_groups = tl.permute(lane_groups, (0, 2, 1))
    lane_pairs = tl.reshape(lane_groups, (64, 32, 2, 2))
    even, odd = tl.split(lane_pairs)
    item0, item2 = tl.split(even)
    item1, item3 = tl.split(odd)
    lane_sum = item0 + item1
    lane_sum += item2
    lane_sum += item3
    # PersistentSoftmax.cuh uses shuffle-xor at offsets 16, 8, 4, 2, 1.
    # Retaining lane zero at each step produces the same FP32 expression tree.
    denominator = _warp_add_halves(lane_sum, 64, 32)
    denominator = _warp_add_halves(denominator, 64, 16)
    denominator = _warp_add_halves(denominator, 64, 8)
    denominator = _warp_add_halves(denominator, 64, 4)
    denominator = _warp_add_halves(denominator, 64, 2)
    denominator = tl.reshape(denominator, (64,))
    if CAPTURE:
        tl.store(denominator_trace_ptr + head * S + queries, denominator)
    # NVCC emits IEEE round-to-nearest division for PersistentSoftmax.cuh;
    # Triton's `/` selects its fast division path and can move FP16 rounding.
    probabilities = tl.div_rn(numerator, denominator[:, None])
    probabilities = probabilities.to(tl.float16)
    if CAPTURE:
        tl.store(
            probability_trace_ptr
            + head * S * S
            + queries[:, None] * S
            + keys[None, :],
            probabilities,
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
def _transformer_megakernel(
    input_ptr,
    valid_ptr,
    packed_ptr,
    workspace_ptr,
    output_ptr,
    trace_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    HD: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    E: tl.constexpr,
    LAYER_WEIGHTS: tl.constexpr,
    SCALE: tl.constexpr,
    CAPTURE: tl.constexpr,
):
    sequence = tl.program_id(0)
    input_base = input_ptr + sequence * E
    valid_base = valid_ptr + sequence * S
    workspace_base = workspace_ptr + sequence * JIT_WORKSPACE_SLOTS * E
    x = workspace_base
    norm = x + E
    q = norm + E
    k = q + E
    v = k + E
    scratch = v + E

    offsets = tl.arange(0, 256)
    for start in range(0, E, 256):
        tl.store(x + start + offsets, tl.load(input_base + start + offsets))
    tl.debug_barrier()

    for layer in range(NUM_LAYERS):
        weights = packed_ptr + layer * LAYER_WEIGHTS
        for row_start in range(0, S, 64):
            _layer_norm_half(
                x,
                norm,
                weights + JIT_NORM1_WEIGHT,
                weights + JIT_NORM1_BIAS,
                row_start,
                D,
                64,
            )
        tl.debug_barrier()
        _trace_tensor(norm, trace_ptr, layer * 9, E, CAPTURE)

        for row_start in range(0, S, 64):
            for column_start in range(0, D, 64):
                _linear_64x64(
                    norm,
                    q,
                    weights + JIT_Q_WEIGHT,
                    weights + JIT_Q_BIAS,
                    x,
                    valid_base,
                    row_start,
                    column_start,
                    D,
                    0,
                )
                _linear_64x64(
                    norm,
                    k,
                    weights + JIT_K_WEIGHT,
                    weights + JIT_K_BIAS,
                    x,
                    valid_base,
                    row_start,
                    column_start,
                    D,
                    0,
                )
                _linear_64x64(
                    norm,
                    v,
                    weights + JIT_V_WEIGHT,
                    weights + JIT_V_BIAS,
                    x,
                    valid_base,
                    row_start,
                    column_start,
                    D,
                    0,
                )
        tl.debug_barrier()
        _trace_tensor(q, trace_ptr, layer * 9 + 1, E, CAPTURE)
        _trace_tensor(k, trace_ptr, layer * 9 + 2, E, CAPTURE)
        _trace_tensor(v, trace_ptr, layer * 9 + 3, E, CAPTURE)

        for head in range(H):
            for row_start in range(0, S, 64):
                _attention_64x32(
                    q,
                    k,
                    v,
                    scratch,
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
                    SCALE,
                    CAPTURE,
                )
        tl.debug_barrier()
        _trace_tensor(scratch, trace_ptr, layer * 9 + 4, E, CAPTURE)

        for row_start in range(0, S, 64):
            for column_start in range(0, D, 64):
                _linear_64x64(
                    scratch,
                    norm,
                    weights + JIT_OUT_WEIGHT,
                    weights + JIT_OUT_BIAS,
                    x,
                    valid_base,
                    row_start,
                    column_start,
                    D,
                    2,
                )
        tl.debug_barrier()
        _trace_tensor(norm, trace_ptr, layer * 9 + 5, E, CAPTURE)

        for row_start in range(0, S, 64):
            _layer_norm_half(
                norm,
                x,
                weights + JIT_NORM2_WEIGHT,
                weights + JIT_NORM2_BIAS,
                row_start,
                D,
                64,
            )
        tl.debug_barrier()
        _trace_tensor(x, trace_ptr, layer * 9 + 6, E, CAPTURE)

        for row_start in range(0, S, 64):
            for column_start in range(0, D, 64):
                _linear_64x64(
                    x,
                    scratch,
                    weights + JIT_FFN_IN_WEIGHT,
                    weights + JIT_FFN_IN_BIAS,
                    norm,
                    valid_base,
                    row_start,
                    column_start,
                    D,
                    1,
                )
        tl.debug_barrier()
        _trace_tensor(scratch, trace_ptr, layer * 9 + 7, E, CAPTURE)

        for row_start in range(0, S, 64):
            for column_start in range(0, D, 64):
                _linear_64x64(
                    scratch,
                    x,
                    weights + JIT_FFN_OUT_WEIGHT,
                    weights + JIT_FFN_OUT_BIAS,
                    norm,
                    valid_base,
                    row_start,
                    column_start,
                    D,
                    3,
                )
        tl.debug_barrier()
        _trace_tensor(x, trace_ptr, layer * 9 + 8, E, CAPTURE)

    final_norm_weight = packed_ptr + NUM_LAYERS * LAYER_WEIGHTS
    final_norm_bias = final_norm_weight + D
    for row_start in range(0, S, 64):
        _layer_norm_half(
            x,
            norm,
            final_norm_weight,
            final_norm_bias,
            row_start,
            D,
            64,
        )
    tl.debug_barrier()
    _trace_tensor(norm, trace_ptr, NUM_LAYERS * 9, E, CAPTURE)

    for start in range(0, E, 256):
        indices = start + offsets
        rows = indices // D
        valid = tl.load(valid_base + rows)
        result = tl.load(norm + indices)
        tl.store(output_ptr + sequence * E + indices, tl.where(valid, result, 0.0))


def fused_megakernel_forward(
    value: torch.Tensor,
    valid_mask: torch.Tensor,
    packed_weights: torch.Tensor,
    *,
    capture_trace: bool = False,
):
    batch_size = value.shape[0]
    if capture_trace and batch_size != 1:
        raise ValueError("trace capture supports exactly one sequence")
    workspace = torch.empty(
        (batch_size, WORKSPACE_SLOTS, SEQUENCE, MODEL),
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
    _transformer_megakernel[(batch_size,)](
        value,
        valid_mask,
        packed_weights,
        workspace,
        output,
        trace,
        S=SEQUENCE,
        D=MODEL,
        H=HEADS,
        HD=HEAD_DIM,
        NUM_LAYERS=LAYERS,
        E=ELEMENTS,
        LAYER_WEIGHTS=LAYER_STRIDE,
        SCALE=HEAD_DIM ** -0.5,
        CAPTURE=capture_trace,
        num_warps=4,
        num_stages=2,
    )
    return (output, trace) if capture_trace else output
