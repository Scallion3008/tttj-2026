"""Two-pass exact-boundary streaming attention for benchmark case 14."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


MODEL = 1024
HEADS = 16
HEAD_DIM = 64
QKV_STRIDE = 3 * MODEL
JIT_MODEL = tl.constexpr(MODEL)
JIT_HEADS = tl.constexpr(HEADS)
JIT_HEAD_DIM = tl.constexpr(HEAD_DIM)
JIT_QKV_STRIDE = tl.constexpr(QKV_STRIDE)

ROW_TILE = int(os.environ.get("TTTJ_STEP8_ATTENTION_M", "64"))
KEY_TILE = int(os.environ.get("TTTJ_STEP8_ATTENTION_N", "128"))
NUM_WARPS = int(os.environ.get("TTTJ_STEP8_ATTENTION_WARPS", "8"))
NUM_STAGES = int(os.environ.get("TTTJ_STEP8_ATTENTION_STAGES", "3"))
DIVISION_MODE = int(os.environ.get("TTTJ_STEP8_DIVISION_MODE", "1"))


@triton.jit
def _two_pass_causal_attention_kernel(
    qkv_ptr,
    output_ptr,
    valid_ptr,
    SEQUENCE: tl.constexpr,
    ROWS: tl.constexpr,
    KEYS: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
    ALL_VALID: tl.constexpr,
    DIVISION_ALGORITHM: tl.constexpr,
):
    tiles_per_head = tl.cdiv(SEQUENCE, ROWS)
    task = tl.program_id(0)
    # Longest-processing-time-first order prevents the final long causal rows
    # from becoming a several-thousand-iteration tail.
    query_tile = tiles_per_head - 1 - task % tiles_per_head
    head = (task // tiles_per_head) % JIT_HEADS
    batch = task // (tiles_per_head * JIT_HEADS)

    queries = query_tile * ROWS + tl.arange(0, ROWS)
    features = tl.arange(0, JIT_HEAD_DIM)
    keys = tl.arange(0, KEYS)
    qkv_base = qkv_ptr + batch.to(tl.int64) * SEQUENCE * JIT_QKV_STRIDE
    query = tl.load(
        qkv_base
        + queries[:, None].to(tl.int64) * JIT_QKV_STRIDE
        + head * JIT_HEAD_DIM
        + features[None, :],
        mask=queries[:, None] < SEQUENCE,
        other=0.0,
    )

    row_maximum = tl.full((ROWS,), -float("inf"), dtype=tl.float32)
    key_end = tl.minimum((query_tile + 1) * ROWS, SEQUENCE)
    for key_start in tl.range(0, key_end, KEYS, num_stages=PIPE_STAGES):
        key_positions = key_start + keys
        key = tl.load(
            qkv_base
            + key_positions[None, :].to(tl.int64) * JIT_QKV_STRIDE
            + JIT_MODEL
            + head * JIT_HEAD_DIM
            + features[:, None],
            mask=key_positions[None, :] < SEQUENCE,
            other=0.0,
        )
        scores = tl.dot(query, key).to(tl.float16).to(tl.float32)
        scores = (scores * 0.125).to(tl.float16).to(tl.float32)
        keep = key_positions[None, :] <= queries[:, None]
        if not ALL_VALID:
            key_valid = tl.load(
                valid_ptr + batch * SEQUENCE + key_positions,
                mask=key_positions < SEQUENCE,
                other=False,
            )
            keep &= key_valid[None, :]
        scores = tl.where(keep, scores, -float("inf"))
        tile_maximum = tl.max(scores, axis=1)
        row_maximum = tl.maximum(row_maximum, tile_maximum)

    # A separate global-denominator pass avoids the reassociation introduced
    # by online max rescaling. It costs one additional QK traversal, but tracks
    # the materialized FP32 softmax reference much more closely.
    row_sum = tl.zeros((ROWS,), dtype=tl.float32)
    for key_start in tl.range(0, key_end, KEYS, num_stages=PIPE_STAGES):
        key_positions = key_start + keys
        key = tl.load(
            qkv_base
            + key_positions[None, :].to(tl.int64) * JIT_QKV_STRIDE
            + JIT_MODEL
            + head * JIT_HEAD_DIM
            + features[:, None],
            mask=key_positions[None, :] < SEQUENCE,
            other=0.0,
        )
        scores = tl.dot(query, key).to(tl.float16).to(tl.float32)
        scores = (scores * 0.125).to(tl.float16).to(tl.float32)
        keep = key_positions[None, :] <= queries[:, None]
        if not ALL_VALID:
            key_valid = tl.load(
                valid_ptr + batch * SEQUENCE + key_positions,
                mask=key_positions < SEQUENCE,
                other=False,
            )
            keep &= key_valid[None, :]
        numerator = libdevice.exp(scores - row_maximum[:, None])
        numerator = tl.where(keep, numerator, 0.0)
        row_sum += tl.sum(numerator, axis=1)

    context = tl.zeros((ROWS, JIT_HEAD_DIM), dtype=tl.float32)
    for key_start in tl.range(0, key_end, KEYS, num_stages=PIPE_STAGES):
        key_positions = key_start + keys
        key = tl.load(
            qkv_base
            + key_positions[None, :].to(tl.int64) * JIT_QKV_STRIDE
            + JIT_MODEL
            + head * JIT_HEAD_DIM
            + features[:, None],
            mask=key_positions[None, :] < SEQUENCE,
            other=0.0,
        )
        scores = tl.dot(query, key).to(tl.float16).to(tl.float32)
        scores = (scores * 0.125).to(tl.float16).to(tl.float32)
        keep = key_positions[None, :] <= queries[:, None]
        if not ALL_VALID:
            key_valid = tl.load(
                valid_ptr + batch * SEQUENCE + key_positions,
                mask=key_positions < SEQUENCE,
                other=False,
            )
            keep &= key_valid[None, :]
        numerator = libdevice.exp(scores - row_maximum[:, None])
        numerator = tl.where(keep, numerator, 0.0)
        if DIVISION_ALGORITHM == 0:
            probabilities = tl.div_rn(numerator, row_sum[:, None])
        else:
            inverse = tl.div_rn(1.0, row_sum)
            probabilities = numerator * inverse[:, None]
            remainder = libdevice.fma_rn(
                -probabilities, row_sum[:, None], numerator
            )
            probabilities += remainder * inverse[:, None]
        probabilities = probabilities.to(tl.float16)
        values = tl.load(
            qkv_base
            + key_positions[:, None].to(tl.int64) * JIT_QKV_STRIDE
            + 2 * JIT_MODEL
            + head * JIT_HEAD_DIM
            + features[None, :],
            mask=key_positions[:, None] < SEQUENCE,
            other=0.0,
        )
        context = tl.dot(probabilities, values, context)

    query_valid = queries < SEQUENCE
    if not ALL_VALID:
        query_valid &= tl.load(
            valid_ptr + batch * SEQUENCE + queries,
            mask=queries < SEQUENCE,
            other=False,
        )
    tl.store(
        output_ptr
        + batch.to(tl.int64) * SEQUENCE * JIT_MODEL
        + queries[:, None].to(tl.int64) * JIT_MODEL
        + head * JIT_HEAD_DIM
        + features[None, :],
        context,
        mask=query_valid[:, None],
    )


def two_pass_causal_attention(
    qkv: torch.Tensor,
    valid_token_mask: torch.Tensor,
    *,
    all_valid: bool,
) -> torch.Tensor:
    if qkv.ndim != 5 or qkv.shape[2:] != (3, HEADS, HEAD_DIM):
        raise ValueError(f"unexpected case-14 packed QKV shape {tuple(qkv.shape)}")
    if qkv.dtype != torch.float16 or not qkv.is_cuda or not qkv.is_contiguous():
        raise ValueError("case-14 packed QKV must be contiguous CUDA FP16")
    batch, sequence = qkv.shape[:2]
    output = torch.empty(
        batch,
        sequence,
        HEADS,
        HEAD_DIM,
        device=qkv.device,
        dtype=qkv.dtype,
    )
    grid = batch * HEADS * triton.cdiv(sequence, ROW_TILE)
    _two_pass_causal_attention_kernel[(grid,)](
        qkv,
        output,
        valid_token_mask,
        SEQUENCE=sequence,
        ROWS=ROW_TILE,
        KEYS=KEY_TILE,
        PIPE_STAGES=NUM_STAGES,
        ALL_VALID=all_valid,
        DIVISION_ALGORITHM=DIVISION_MODE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return output


@triton.jit
def _probability_value_kernel(
    probability_ptr,
    value_ptr,
    output_ptr,
    SEQUENCE: tl.constexpr,
    QUERIES: tl.constexpr,
    VALUE_HEAD_STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    KEYS: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    tiles_per_head = tl.cdiv(QUERIES, ROWS)
    task = tl.program_id(0)
    query_tile = task % tiles_per_head
    batch_head = task // tiles_per_head
    queries = query_tile * ROWS + tl.arange(0, ROWS)
    keys = tl.arange(0, KEYS)
    features = tl.arange(0, JIT_HEAD_DIM)
    context = tl.zeros((ROWS, JIT_HEAD_DIM), dtype=tl.float32)
    probability_base = probability_ptr + batch_head.to(tl.int64) * QUERIES * SEQUENCE
    value_base = value_ptr + batch_head.to(tl.int64) * VALUE_HEAD_STRIDE
    for key_start in tl.range(0, SEQUENCE, KEYS, num_stages=PIPE_STAGES):
        key_positions = key_start + keys
        probabilities = tl.load(
            probability_base
            + queries[:, None].to(tl.int64) * SEQUENCE
            + key_positions[None, :],
            mask=(queries[:, None] < QUERIES)
            & (key_positions[None, :] < SEQUENCE),
            other=0.0,
        )
        values = tl.load(
            value_base
            + key_positions[:, None].to(tl.int64) * JIT_HEAD_DIM
            + features[None, :],
            mask=key_positions[:, None] < SEQUENCE,
            other=0.0,
        )
        context = tl.dot(probabilities, values, context)
    tl.store(
        output_ptr
        + batch_head.to(tl.int64) * QUERIES * JIT_HEAD_DIM
        + queries[:, None].to(tl.int64) * JIT_HEAD_DIM
        + features[None, :],
        context,
        mask=queries[:, None] < QUERIES,
    )


def probability_value(
    probabilities: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Triton PV tuning candidate for exact materialized probabilities."""
    if probabilities.ndim != 4 or value.ndim != 4:
        raise ValueError("PV expects [B,H,M,N] probabilities and [B,H,N,D] values")
    batch, heads, queries, sequence = probabilities.shape
    if heads != HEADS or value.shape != (batch, heads, sequence, HEAD_DIM):
        raise ValueError("unexpected case-14 PV shapes")
    if not probabilities.is_contiguous() or value.stride(-1) != 1:
        raise ValueError("PV probabilities must be contiguous and values feature-major")
    output = torch.empty(
        batch,
        heads,
        queries,
        HEAD_DIM,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    rows = 64
    keys = 128
    grid = batch * heads * triton.cdiv(queries, rows)
    _probability_value_kernel[(grid,)](
        probabilities,
        value,
        output,
        SEQUENCE=sequence,
        QUERIES=queries,
        VALUE_HEAD_STRIDE=value.stride(1),
        ROWS=rows,
        KEYS=keys,
        PIPE_STAGES=3,
        num_warps=8,
        num_stages=3,
    )
    return output


@triton.jit
def _score_value_kernel(
    score_ptr,
    statistic_ptr,
    value_ptr,
    output_ptr,
    QUERY_START,
    SEQUENCE,
    QUERIES: tl.constexpr,
    VALUE_HEAD_STRIDE: tl.constexpr,
    OUTPUT_HEAD_STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    KEYS: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
    FAST_EXP: tl.constexpr,
):
    tiles_per_head = tl.cdiv(QUERIES, ROWS)
    task = tl.program_id(0)
    query_tile = task % tiles_per_head
    batch_head = task // tiles_per_head
    queries = query_tile * ROWS + tl.arange(0, ROWS)
    keys = tl.arange(0, KEYS)
    features = tl.arange(0, JIT_HEAD_DIM)
    row_indices = batch_head.to(tl.int64) * QUERIES + queries
    maximum = tl.load(
        statistic_ptr + row_indices * 2,
        mask=queries < QUERIES,
        other=0.0,
    )
    denominator = tl.load(
        statistic_ptr + row_indices * 2 + 1,
        mask=queries < QUERIES,
        other=1.0,
    )
    inverse = tl.div_rn(1.0, denominator)
    context = tl.zeros((ROWS, JIT_HEAD_DIM), dtype=tl.float32)
    score_base = score_ptr + batch_head.to(tl.int64) * QUERIES * SEQUENCE
    value_base = value_ptr + batch_head.to(tl.int64) * VALUE_HEAD_STRIDE
    for key_start in tl.range(0, SEQUENCE, KEYS, num_stages=PIPE_STAGES):
        key_positions = key_start + keys
        scores = tl.load(
            score_base
            + queries[:, None].to(tl.int64) * SEQUENCE
            + key_positions[None, :],
            mask=(queries[:, None] < QUERIES)
            & (key_positions[None, :] < SEQUENCE),
            other=0.0,
        )
        scores = (scores * 0.125).to(tl.float16).to(tl.float32)
        keep = key_positions[None, :] <= QUERY_START + queries[:, None]
        if FAST_EXP:
            numerator = tl.exp(scores - maximum[:, None])
        else:
            numerator = libdevice.exp(scores - maximum[:, None])
        numerator = tl.where(keep, numerator, 0.0)
        probabilities = numerator * inverse[:, None]
        remainder = libdevice.fma_rn(
            -probabilities, denominator[:, None], numerator
        )
        probabilities = (
            probabilities + remainder * inverse[:, None]
        ).to(tl.float16)
        values = tl.load(
            value_base
            + key_positions[:, None].to(tl.int64) * JIT_HEAD_DIM
            + features[None, :],
            mask=key_positions[:, None] < SEQUENCE,
            other=0.0,
        )
        context = tl.dot(probabilities, values, context)
    tl.store(
        output_ptr
        + batch_head.to(tl.int64) * OUTPUT_HEAD_STRIDE
        + queries[:, None].to(tl.int64) * JIT_HEAD_DIM
        + features[None, :],
        context,
        mask=queries[:, None] < QUERIES,
    )


def score_value(
    scores: torch.Tensor,
    statistics: torch.Tensor,
    value: torch.Tensor,
    query_start: int,
    *,
    rows: int = 128,
    keys: int = 64,
    num_warps: int = 8,
    num_stages: int = 3,
    fast_exp: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse the exact probability epilogue with the PV tensor-core mainloop."""
    batch, heads, queries, sequence = scores.shape
    if heads != HEADS or value.shape != (batch, heads, sequence, HEAD_DIM):
        raise ValueError("unexpected case-14 fused PV shapes")
    if statistics.shape != (batch * heads * queries, 2):
        raise ValueError("unexpected case-14 softmax statistics shape")
    if out is None:
        output = torch.empty(
            batch,
            heads,
            queries,
            HEAD_DIM,
            device=scores.device,
            dtype=scores.dtype,
        )
    else:
        if out.shape != (batch, heads, queries, HEAD_DIM):
            raise ValueError("unexpected case-14 fused PV output shape")
        if out.dtype != scores.dtype or out.device != scores.device:
            raise ValueError("case-14 fused PV output must match scores")
        if out.stride(-2) != HEAD_DIM or out.stride(-1) != 1:
            raise ValueError("case-14 fused PV output rows must be contiguous")
        output = out
    grid = batch * heads * triton.cdiv(queries, rows)
    _score_value_kernel[(grid,)](
        scores,
        statistics,
        value,
        output,
        QUERY_START=query_start,
        SEQUENCE=sequence,
        QUERIES=queries,
        VALUE_HEAD_STRIDE=value.stride(1),
        OUTPUT_HEAD_STRIDE=output.stride(1),
        ROWS=rows,
        KEYS=keys,
        PIPE_STAGES=num_stages,
        FAST_EXP=fast_exp,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def attention_tuning() -> dict[str, int]:
    return {
        "rows": ROW_TILE,
        "keys": KEY_TILE,
        "warps": NUM_WARPS,
        "stages": NUM_STAGES,
        "division": DIVISION_MODE,
    }
