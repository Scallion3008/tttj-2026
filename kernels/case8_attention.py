"""Fused causal attention for the packed-QKV case-8 layout."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


SEQUENCE = 128
MODEL = 1024
HEADS = 4
HEAD_DIM = 256
QKV_STRIDE = 3 * MODEL
JIT_SEQUENCE = tl.constexpr(SEQUENCE)
JIT_MODEL = tl.constexpr(MODEL)
JIT_HEADS = tl.constexpr(HEADS)
JIT_HEAD_DIM = tl.constexpr(HEAD_DIM)
JIT_QKV_STRIDE = tl.constexpr(QKV_STRIDE)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


ROW_TILE = _env_int("TTTJ_STEP6_ATTENTION_M", 64)
REDUCTION_TILE = _env_int("TTTJ_STEP6_ATTENTION_K", 128)
NUM_WARPS = _env_int("TTTJ_STEP6_ATTENTION_WARPS", 4)
NUM_STAGES = _env_int("TTTJ_STEP6_ATTENTION_STAGES", 2)
SOFTMAX_MODE = _env_int("TTTJ_STEP6_SOFTMAX_MODE", 0)
DIVISION_MODE = _env_int("TTTJ_STEP6_DIVISION_MODE", 4)
EXP_MODE = _env_int("TTTJ_STEP6_EXP_MODE", 0)
CAUSAL_SKIP = bool(_env_int("TTTJ_STEP6_CAUSAL_SKIP", 1))


@triton.jit
def _add_halves(values, ROWS: tl.constexpr, WIDTH: tl.constexpr):
    halves = tl.permute(
        tl.reshape(values, (ROWS, 2, WIDTH // 2)), (0, 2, 1)
    )
    lower, upper = tl.split(halves)
    return lower + upper


@triton.jit
def _packed_causal_attention_kernel(
    qkv_ptr,
    output_ptr,
    valid_ptr,
    ROWS: tl.constexpr,
    REDUCTION: tl.constexpr,
    KEYS: tl.constexpr,
    QUERY_START: tl.constexpr,
    TILES_PER_HEAD: tl.constexpr,
    ALL_VALID: tl.constexpr,
    SOFTMAX_ALGORITHM: tl.constexpr,
    DIVISION_ALGORITHM: tl.constexpr,
    EXP_ALGORITHM: tl.constexpr,
):
    program = tl.program_id(0)
    row_tile = program % TILES_PER_HEAD
    head = (program // TILES_PER_HEAD) % JIT_HEADS
    batch = program // (TILES_PER_HEAD * JIT_HEADS)

    queries = QUERY_START + row_tile * ROWS + tl.arange(0, ROWS)
    keys = tl.arange(0, KEYS)
    reductions = tl.arange(0, REDUCTION)
    qkv_base = qkv_ptr + batch * JIT_SEQUENCE * JIT_QKV_STRIDE

    scores = tl.zeros((ROWS, KEYS), dtype=tl.float32)
    for reduction_start in range(0, JIT_HEAD_DIM, REDUCTION):
        query = tl.load(
            qkv_base
            + queries[:, None] * JIT_QKV_STRIDE
            + head * JIT_HEAD_DIM
            + reduction_start
            + reductions[None, :]
        )
        key = tl.load(
            qkv_base
            + keys[None, :] * JIT_QKV_STRIDE
            + JIT_MODEL
            + head * JIT_HEAD_DIM
            + reduction_start
            + reductions[:, None]
        )
        scores = tl.dot(query, key, scores)

    # Match the observable half-precision QK and scale boundaries in the
    # organizer reference.  HEAD_DIM**-0.5 is exactly 1/16 here.
    scores = scores.to(tl.float16).to(tl.float32) * 0.0625
    causal = keys[None, :] <= queries[:, None]
    if not ALL_VALID:
        causal &= tl.load(valid_ptr + batch * JIT_SEQUENCE + keys)[None, :]
    scores = tl.where(causal, scores, -float("inf"))
    scores = scores.to(tl.float16).to(tl.float32)

    maximum = tl.max(scores, axis=1)
    if EXP_ALGORITHM == 0:
        numerator = libdevice.exp(scores - maximum[:, None])
    else:
        numerator = tl.exp(scores - maximum[:, None])
    if SOFTMAX_ALGORITHM == 0:
        # PyTorch's persistent S128 softmax accumulates four values per lane
        # and then performs a 32-lane shuffle tree.
        if KEYS == 128:
            lane_groups = tl.reshape(numerator, (ROWS, 4, 32))
            lane_groups = tl.permute(lane_groups, (0, 2, 1))
            lane_pairs = tl.reshape(lane_groups, (ROWS, 32, 2, 2))
            even, odd = tl.split(lane_pairs)
            item0, item2 = tl.split(even)
            item1, item3 = tl.split(odd)
            lane_sum = item0 + item1
            lane_sum += item2
            lane_sum += item3
        else:
            lane_groups = tl.reshape(numerator, (ROWS, 2, 32))
            lane_groups = tl.permute(lane_groups, (0, 2, 1))
            item0, item1 = tl.split(lane_groups)
            lane_sum = item0 + item1
        denominator = _add_halves(lane_sum, ROWS, 32)
        denominator = _add_halves(denominator, ROWS, 16)
        denominator = _add_halves(denominator, ROWS, 8)
        denominator = _add_halves(denominator, ROWS, 4)
        denominator = _add_halves(denominator, ROWS, 2)
        denominator = tl.reshape(denominator, (ROWS,))
    else:
        denominator = tl.sum(numerator, axis=1)

    if DIVISION_ALGORITHM == 0:
        probabilities = tl.div_rn(numerator, denominator[:, None])
    elif DIVISION_ALGORITHM == 1:
        inverse = tl.div_rn(1.0, denominator)
        probabilities = numerator * inverse[:, None]
    elif DIVISION_ALGORITHM == 2:
        probabilities = numerator / denominator[:, None]
    else:
        inverse = tl.div_rn(1.0, denominator)
        probabilities = numerator * inverse[:, None]
        remainder = libdevice.fma_rn(
            -probabilities, denominator[:, None], numerator
        )
        probabilities += remainder * inverse[:, None]
    probabilities = probabilities.to(tl.float16)

    columns = tl.arange(0, JIT_HEAD_DIM)
    values = tl.load(
        qkv_base
        + keys[:, None] * JIT_QKV_STRIDE
        + 2 * JIT_MODEL
        + head * JIT_HEAD_DIM
        + columns[None, :]
    )
    context = tl.dot(probabilities, values)
    tl.store(
        output_ptr
        + batch * JIT_SEQUENCE * JIT_MODEL
        + queries[:, None] * JIT_MODEL
        + head * JIT_HEAD_DIM
        + columns[None, :],
        context,
    )


def packed_causal_attention(
    qkv: torch.Tensor,
    valid_token_mask: torch.Tensor,
    *,
    all_valid: bool,
) -> torch.Tensor:
    if qkv.shape != (64, SEQUENCE, QKV_STRIDE):
        raise ValueError(f"unexpected packed QKV shape {tuple(qkv.shape)}")
    if qkv.dtype != torch.float16 or not qkv.is_cuda or not qkv.is_contiguous():
        raise ValueError("packed QKV must be contiguous CUDA float16")
    if SEQUENCE % ROW_TILE or HEAD_DIM % REDUCTION_TILE:
        raise ValueError("attention tile sizes must divide the fixed dimensions")
    output = torch.empty(
        (qkv.shape[0], SEQUENCE, MODEL), device=qkv.device, dtype=qkv.dtype
    )
    launches = (
        ((64, 0, 1), (128, 64, 1))
        if CAUSAL_SKIP and ROW_TILE == 64
        else ((128, 0, SEQUENCE // ROW_TILE),)
    )
    for keys, query_start, tiles_per_head in launches:
        grid = qkv.shape[0] * HEADS * tiles_per_head
        _packed_causal_attention_kernel[(grid,)](
            qkv,
            output,
            valid_token_mask,
            ROWS=ROW_TILE,
            REDUCTION=REDUCTION_TILE,
            KEYS=keys,
            QUERY_START=query_start,
            TILES_PER_HEAD=tiles_per_head,
            ALL_VALID=all_valid,
            SOFTMAX_ALGORITHM=SOFTMAX_MODE,
            DIVISION_ALGORITHM=DIVISION_MODE,
            EXP_ALGORITHM=EXP_MODE,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    return output


def attention_tuning() -> dict[str, int]:
    return {
        "rows": ROW_TILE,
        "reduction": REDUCTION_TILE,
        "warps": NUM_WARPS,
        "stages": NUM_STAGES,
        "softmax": SOFTMAX_MODE,
        "division": DIVISION_MODE,
        "exp": EXP_MODE,
        "causal_skip": int(CAUSAL_SKIP),
    }
