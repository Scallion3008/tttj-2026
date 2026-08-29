"""Completion-driven transformer megakernel for benchmark cases 2--4/12.

The small-batch sequence-resident kernel intentionally assigns one CTA to a
whole sequence.  That is efficient once the batch fills the H200, but leaves
most SMs idle in the step-4 cases.  This kernel instead flattens batch and
sequence for the linear phases and distributes independent projection,
attention, and row tiles over a persistent grid.  Completed tasks atomically
release only their direct successors, allowing independent phases and layers
to overlap without a device-wide computation barrier.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from fused_megakernel import (
    ATTENTION_REDUCTION_TILE,
    DIVISION_MODE,
    EXP_MODE,
    FFN_IN_BIAS,
    FFN_IN_WEIGHT,
    FFN_OUT_BIAS,
    FFN_OUT_WEIGHT,
    GELU_MODE,
    K_BIAS,
    K_WEIGHT,
    LAYER_STRIDE,
    NORM1_BIAS,
    NORM1_WEIGHT,
    NORM2_BIAS,
    NORM2_WEIGHT,
    OUT_BIAS,
    OUT_WEIGHT,
    Q_BIAS,
    Q_WEIGHT,
    SOFTMAX_MODE,
    V_BIAS,
    V_WEIGHT,
    _attention_mxhd,
    _layer_norm_half,
    _linear_tile,
)


MODEL = 128
HEADS = 4
HEAD_DIM = 32
LAYERS = 4
LINEAR_M = 64
LINEAR_N = 128
LINEAR_K = 64
NORM_M = 64
WORKSPACE_SLOTS_PER_LAYER = 4
WORKSPACE_SLOTS = WORKSPACE_SLOTS_PER_LAYER
JIT_WORKSPACE_SLOTS_PER_LAYER = tl.constexpr(WORKSPACE_SLOTS_PER_LAYER)


def _environment_integer(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _environment_is_set(name: str) -> bool:
    return name in os.environ


DAG_NUM_WARPS = _environment_integer("TTTJ_DAG_NUM_WARPS", 4)
DAG_NUM_STAGES = _environment_integer("TTTJ_DAG_NUM_STAGES", 3)
DAG_LINEAR_M = _environment_integer("TTTJ_DAG_LINEAR_M", LINEAR_M)
DAG_LINEAR_N = _environment_integer("TTTJ_DAG_LINEAR_N", LINEAR_N)
DAG_LINEAR_K = _environment_integer("TTTJ_DAG_LINEAR_K", LINEAR_K)
DAG_NORM_M = _environment_integer("TTTJ_DAG_NORM_M", NORM_M)
DAG_MAX_PROGRAMS = _environment_integer("TTTJ_DAG_PROGRAMS", 132)
JIT_DAG_THREADS = tl.constexpr(DAG_NUM_WARPS * 32)
DAG_PARALLEL_TAIL = bool(_environment_integer("TTTJ_DAG_PARALLEL_TAIL", 1))
DAG_PARALLEL_S32_ATTENTION = bool(
    _environment_integer("TTTJ_DAG_PARALLEL_S32_ATTENTION", 1)
)

# Scheduler layout.  Every task occupies exactly one monotonically allocated
# queue slot; case 12 is the largest graph at 2,848 tasks.
SCHEDULER_ELEMENTS = 8192
QUEUE_BASE = tl.constexpr(8)
QUEUE_CAPACITY = tl.constexpr(4096)
START = tl.constexpr(0)
HEAD = tl.constexpr(1)
TAIL = tl.constexpr(2)
DONE = tl.constexpr(3)

TASK_LN1 = tl.constexpr(0)
TASK_QKV = tl.constexpr(1)
TASK_ATTENTION = tl.constexpr(2)
TASK_OUT = tl.constexpr(3)
TASK_LN2 = tl.constexpr(4)
TASK_FFN_IN = tl.constexpr(5)
TASK_FFN_OUT = tl.constexpr(6)
TASK_FINAL_NORM = tl.constexpr(7)
TASK_TYPE_BITS = tl.constexpr(3)
TASK_LAYER_BITS = tl.constexpr(2)
TASK_INDEX_SHIFT = tl.constexpr(5)
TASK_TYPE_MASK = tl.constexpr(7)
TASK_LAYER_MASK = tl.constexpr(3)

JIT_NORM1_WEIGHT = tl.constexpr(NORM1_WEIGHT)
JIT_NORM1_BIAS = tl.constexpr(NORM1_BIAS)
JIT_Q_WEIGHT = tl.constexpr(Q_WEIGHT)
JIT_Q_BIAS = tl.constexpr(Q_BIAS)
JIT_OUT_WEIGHT = tl.constexpr(OUT_WEIGHT)
JIT_OUT_BIAS = tl.constexpr(OUT_BIAS)
JIT_NORM2_WEIGHT = tl.constexpr(NORM2_WEIGHT)
JIT_NORM2_BIAS = tl.constexpr(NORM2_BIAS)
JIT_FFN_IN_WEIGHT = tl.constexpr(FFN_IN_WEIGHT)
JIT_FFN_IN_BIAS = tl.constexpr(FFN_IN_BIAS)
JIT_FFN_OUT_WEIGHT = tl.constexpr(FFN_OUT_WEIGHT)
JIT_FFN_OUT_BIAS = tl.constexpr(FFN_OUT_BIAS)


def is_step_4_shape(value: torch.Tensor, num_heads: int) -> bool:
    if value.ndim != 3 or value.shape[2] != MODEL or num_heads != HEADS:
        return False
    batch, sequence, _ = value.shape
    return (sequence == 128 and batch in (1, 4, 16)) or (
        sequence == 32 and batch == 64
    )


def resolved_dag_tuning(batch_size: int, sequence: int) -> dict[str, int]:
    num_warps = DAG_NUM_WARPS
    linear_m = DAG_LINEAR_M
    attention_m = 32 if sequence == 32 else 64
    if sequence == 128:
        if not _environment_is_set("TTTJ_DAG_NUM_WARPS"):
            num_warps = 8
        if not _environment_is_set("TTTJ_DAG_LINEAR_M"):
            linear_m = 128
        if not _environment_is_set("TTTJ_DAG_ATTENTION_M"):
            attention_m = 128
    if _environment_is_set("TTTJ_DAG_ATTENTION_M"):
        attention_m = _environment_integer("TTTJ_DAG_ATTENTION_M", attention_m)
    tokens = batch_size * sequence
    row_tiles = triton.cdiv(tokens, linear_m)
    qkv_tasks = 3 * row_tiles * triton.cdiv(MODEL, DAG_LINEAR_N)
    # The production static DAG owns one full linear row per sequence/group.
    # S128 uses M128; S32 pairs two sequences into one M64 row.
    attention_tasks = row_tiles
    linear_tasks = row_tiles * triton.cdiv(MODEL, DAG_LINEAR_N)
    programs = min(DAG_MAX_PROGRAMS, max(qkv_tasks, attention_tasks, linear_tasks))
    return {
        "programs": programs,
        "num_warps": num_warps,
        "num_stages": DAG_NUM_STAGES,
        "linear_m": linear_m,
        "linear_n": DAG_LINEAR_N,
        "linear_k": DAG_LINEAR_K,
        "attention_m": attention_m,
        "attention_k": ATTENTION_REDUCTION_TILE,
        "norm_m": DAG_NORM_M,
    }


@triton.jit
def _task_descriptor(task_type, layer, index):
    return (index << TASK_INDEX_SHIFT) | (layer << TASK_TYPE_BITS) | task_type


@triton.jit
def _enqueue(scheduler_ptr, descriptor):
    slot = tl.atomic_add(
        scheduler_ptr + TAIL, 1, sem="relaxed", scope="gpu"
    )
    # Tail reserves before the descriptor is stored.  Consumers that claim
    # this slot spin on its zero sentinel until the release exchange publishes
    # the complete descriptor.
    tl.atomic_xchg(
        scheduler_ptr + QUEUE_BASE + slot,
        descriptor + 1,
        sem="release",
        scope="gpu",
    )


@triton.jit
def _increment_dependency(counter_ptr, target):
    previous = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    return previous == target - 1


@triton.jit
def _publish_cta_writes():
    """Make cooperative-CTA stores visible before a scalar release atomic."""
    tl.debug_barrier()
    participants = tl.arange(0, JIT_DAG_THREADS)
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
def _dequeue(scheduler_ptr, TOTAL_TASKS: tl.constexpr):
    descriptor = -1
    searching = True
    while searching:
        head = tl.atomic_add(
            scheduler_ptr + HEAD, 0, sem="acquire", scope="gpu"
        )
        tail = tl.atomic_add(
            scheduler_ptr + TAIL, 0, sem="acquire", scope="gpu"
        )
        if head < tail:
            previous = tl.atomic_cas(
                scheduler_ptr + HEAD,
                head,
                head + 1,
                sem="acq_rel",
                scope="gpu",
            )
            if previous == head:
                entry = tl.atomic_add(
                    scheduler_ptr + QUEUE_BASE + head,
                    0,
                    sem="acquire",
                    scope="gpu",
                )
                while entry == 0:
                    entry = tl.atomic_add(
                        scheduler_ptr + QUEUE_BASE + head,
                        0,
                        sem="acquire",
                        scope="gpu",
                    )
                descriptor = entry - 1
                searching = False
        else:
            completed = tl.atomic_add(
                scheduler_ptr + DONE, 0, sem="acquire", scope="gpu"
            )
            if completed == TOTAL_TASKS:
                searching = False
    return descriptor


@triton.jit
def _dag_transformer_megakernel(
    input_ptr,
    valid_ptr,
    packed_ptr,
    workspace_ptr,
    output_ptr,
    scheduler_ptr,
    B: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    HD: tl.constexpr,
    T: tl.constexpr,
    E: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    LAYER_WEIGHTS: tl.constexpr,
    SCALE: tl.constexpr,
    LINEAR_ROW_TILE: tl.constexpr,
    LINEAR_COLUMN_TILE: tl.constexpr,
    LINEAR_REDUCTION_TILE: tl.constexpr,
    ATTENTION_ROW_TILE: tl.constexpr,
    ATTENTION_REDUCTION_TILE: tl.constexpr,
    NORM_ROW_TILE: tl.constexpr,
    NORM_ALGORITHM: tl.constexpr,
    SOFTMAX_ALGORITHM: tl.constexpr,
    DIVISION_ALGORITHM: tl.constexpr,
    EXP_ALGORITHM: tl.constexpr,
    GELU_ALGORITHM: tl.constexpr,
    ALL_VALID: tl.constexpr,
):
    worker = tl.program_id(0)
    linear_column_tiles: tl.constexpr = D // LINEAR_COLUMN_TILE
    linear_row_tiles: tl.constexpr = T // LINEAR_ROW_TILE
    norm_tasks: tl.constexpr = T // NORM_ROW_TILE
    linear_tasks: tl.constexpr = linear_row_tiles * linear_column_tiles
    qkv_tasks: tl.constexpr = 3 * linear_tasks
    attention_query_tiles: tl.constexpr = S // ATTENTION_ROW_TILE
    attention_tasks: tl.constexpr = linear_row_tiles
    tasks_per_layer: tl.constexpr = 14 * linear_row_tiles + attention_tasks
    total_tasks: tl.constexpr = NUM_LAYERS * tasks_per_layer + norm_tasks

    attention_dependencies = scheduler_ptr + QUEUE_BASE + QUEUE_CAPACITY
    context_dependencies = attention_dependencies + NUM_LAYERS * attention_tasks
    out_dependencies = context_dependencies + NUM_LAYERS * linear_row_tiles
    ffn_in_dependencies = out_dependencies + NUM_LAYERS * linear_row_tiles
    ffn_out_dependencies = ffn_in_dependencies + NUM_LAYERS * linear_row_tiles

    # The adapter asynchronously clears scheduler storage on the same stream.
    # One worker publishes the initial LayerNorm tasks; this is queue startup,
    # not a barrier between computation phases.
    if worker == 0:
        initial = tl.arange(0, 32)
        initial_descriptor = _task_descriptor(TASK_LN1, 0, initial)
        tl.store(
            scheduler_ptr + QUEUE_BASE + initial,
            initial_descriptor + 1,
            mask=initial < norm_tasks,
        )
        tl.atomic_xchg(
            scheduler_ptr + TAIL,
            norm_tasks,
            sem="release",
            scope="gpu",
        )
        tl.atomic_xchg(
            scheduler_ptr + START, 1, sem="release", scope="gpu"
        )
    started = tl.atomic_add(
        scheduler_ptr + START, 0, sem="acquire", scope="gpu"
    )
    while started == 0:
        started = tl.atomic_add(
            scheduler_ptr + START, 0, sem="acquire", scope="gpu"
        )

    descriptor = _dequeue(scheduler_ptr, total_tasks)
    while descriptor >= 0:
        task_type = descriptor & TASK_TYPE_MASK
        layer = (descriptor >> TASK_TYPE_BITS) & TASK_LAYER_MASK
        task = descriptor >> TASK_INDEX_SHIFT
        weights = packed_ptr + layer * LAYER_WEIGHTS
        residual = input_ptr if layer == 0 else output_ptr
        norm = workspace_ptr + layer * JIT_WORKSPACE_SLOTS_PER_LAYER * E
        q = norm + E
        k = q + E
        v = k + E

        if task_type == TASK_LN1:
            _layer_norm_half(
                residual,
                norm,
                norm,
                weights + JIT_NORM1_WEIGHT,
                weights + JIT_NORM1_BIAS,
                task * NORM_ROW_TILE,
                D,
                NORM_ROW_TILE,
                NORM_ALGORITHM,
                False,
            )
            _publish_cta_writes()
            for branch in range(0, 3):
                for column_tile in range(0, linear_column_tiles):
                    qkv_index = (
                        (task * 3 + branch) * linear_column_tiles
                        + column_tile
                    )
                    _enqueue(
                        scheduler_ptr,
                        _task_descriptor(TASK_QKV, layer, qkv_index),
                    )

        elif task_type == TASK_QKV:
            column_tile = task % linear_column_tiles
            branch_tile = task // linear_column_tiles
            branch = branch_tile % 3
            row_tile = branch_tile // 3
            branch_stride = D * D + D
            _linear_tile(
                norm,
                q + branch * E,
                weights + JIT_Q_WEIGHT + branch * branch_stride,
                weights + JIT_Q_BIAS + branch * branch_stride,
                residual,
                valid_ptr,
                row_tile * LINEAR_ROW_TILE,
                column_tile * LINEAR_COLUMN_TILE,
                D,
                0,
                GELU_ALGORITHM,
                LINEAR_REDUCTION_TILE,
                LINEAR_ROW_TILE,
                LINEAR_COLUMN_TILE,
                ALL_VALID,
            )

            # Each N64 projection tile owns two complete 32-wide heads.  Q
            # releases its query tile; causal K/V tiles release every query
            # tile that consumes their prefix.
            if S == 128:
                sequence = row_tile // 2
                projection_row_tile = row_tile % 2
                for head_offset in range(0, 2):
                    head = column_tile * 2 + head_offset
                    if branch == 0:
                        attention_index = (
                            (sequence * H + head) * attention_query_tiles
                            + projection_row_tile
                        )
                        target = 1 + 2 * (projection_row_tile + 1)
                        if _increment_dependency(
                            attention_dependencies
                            + layer * attention_tasks
                            + attention_index,
                            target,
                        ):
                            _enqueue(
                                scheduler_ptr,
                                _task_descriptor(
                                    TASK_ATTENTION, layer, attention_index
                                ),
                            )
                    else:
                        for query_tile in range(0, attention_query_tiles):
                            if projection_row_tile <= query_tile:
                                attention_index = (
                                    (sequence * H + head)
                                    * attention_query_tiles
                                    + query_tile
                                )
                                target = 1 + 2 * (query_tile + 1)
                                if _increment_dependency(
                                    attention_dependencies
                                    + layer * attention_tasks
                                    + attention_index,
                                    target,
                                ):
                                    _enqueue(
                                        scheduler_ptr,
                                        _task_descriptor(
                                            TASK_ATTENTION,
                                            layer,
                                            attention_index,
                                        ),
                                    )
            else:
                # S32 linears pair two sequences in one M64 tile.
                for sequence_offset_in_tile in range(0, 2):
                    sequence = row_tile * 2 + sequence_offset_in_tile
                    for head_offset in range(0, 2):
                        head = column_tile * 2 + head_offset
                        attention_index = sequence * H + head
                        if _increment_dependency(
                            attention_dependencies
                            + layer * attention_tasks
                            + attention_index,
                            3,
                        ):
                            _enqueue(
                                scheduler_ptr,
                                _task_descriptor(
                                    TASK_ATTENTION, layer, attention_index
                                ),
                            )

        elif task_type == TASK_ATTENTION:
            query_tile = task % attention_query_tiles
            head_task = task // attention_query_tiles
            head = head_task % H
            sequence = head_task // H
            sequence_offset = sequence * S * D
            # The first S128 query half consumes only its 64-key causal
            # prefix, so it can execute while second-half K/V projection is
            # still in flight.  Later queries consume the full sequence.
            if S == 128 and query_tile == 0:
                _attention_mxhd(
                    q + sequence_offset,
                    k + sequence_offset,
                    v + sequence_offset,
                    norm + sequence_offset,
                    valid_ptr + sequence * S,
                    workspace_ptr,
                    workspace_ptr,
                    workspace_ptr,
                    workspace_ptr,
                    0,
                    head,
                    S,
                    D,
                    HD,
                    64,
                    SCALE,
                    False,
                    SOFTMAX_ALGORITHM,
                    DIVISION_ALGORITHM,
                    EXP_ALGORITHM,
                    ATTENTION_REDUCTION_TILE,
                    ATTENTION_ROW_TILE,
                    ALL_VALID,
                )
            else:
                _attention_mxhd(
                    q + sequence_offset,
                    k + sequence_offset,
                    v + sequence_offset,
                    norm + sequence_offset,
                    valid_ptr + sequence * S,
                    workspace_ptr,
                    workspace_ptr,
                    workspace_ptr,
                    workspace_ptr,
                    query_tile * ATTENTION_ROW_TILE,
                    head,
                    S,
                    D,
                    HD,
                    S,
                    SCALE,
                    False,
                    SOFTMAX_ALGORITHM,
                    DIVISION_ALGORITHM,
                    EXP_ALGORITHM,
                    ATTENTION_REDUCTION_TILE,
                    ATTENTION_ROW_TILE,
                    ALL_VALID,
                )
            if S == 128:
                row_tile = sequence * attention_query_tiles + query_tile
                context_target = H
            else:
                row_tile = sequence // 2
                context_target = 2 * H
            if _increment_dependency(
                context_dependencies + layer * linear_row_tiles + row_tile,
                context_target,
            ):
                for column_tile in range(0, linear_column_tiles):
                    linear_index = row_tile * linear_column_tiles + column_tile
                    _enqueue(
                        scheduler_ptr,
                        _task_descriptor(TASK_OUT, layer, linear_index),
                    )

        elif task_type == TASK_OUT:
            row_tile = task // linear_column_tiles
            column_tile = task % linear_column_tiles
            _linear_tile(
                norm,
                q,
                weights + JIT_OUT_WEIGHT,
                weights + JIT_OUT_BIAS,
                residual,
                valid_ptr,
                row_tile * LINEAR_ROW_TILE,
                column_tile * LINEAR_COLUMN_TILE,
                D,
                2,
                GELU_ALGORITHM,
                LINEAR_REDUCTION_TILE,
                LINEAR_ROW_TILE,
                LINEAR_COLUMN_TILE,
                ALL_VALID,
            )
            if _increment_dependency(
                out_dependencies + layer * linear_row_tiles + row_tile,
                linear_column_tiles,
            ):
                _enqueue(
                    scheduler_ptr,
                    _task_descriptor(TASK_LN2, layer, row_tile),
                )

        elif task_type == TASK_LN2:
            _layer_norm_half(
                q,
                norm,
                norm,
                weights + JIT_NORM2_WEIGHT,
                weights + JIT_NORM2_BIAS,
                task * NORM_ROW_TILE,
                D,
                NORM_ROW_TILE,
                NORM_ALGORITHM,
                False,
            )
            for column_tile in range(0, linear_column_tiles):
                linear_index = task * linear_column_tiles + column_tile
                _enqueue(
                    scheduler_ptr,
                    _task_descriptor(TASK_FFN_IN, layer, linear_index),
                )

        elif task_type == TASK_FFN_IN:
            row_tile = task // linear_column_tiles
            column_tile = task % linear_column_tiles
            _linear_tile(
                norm,
                v,
                weights + JIT_FFN_IN_WEIGHT,
                weights + JIT_FFN_IN_BIAS,
                norm,
                valid_ptr,
                row_tile * LINEAR_ROW_TILE,
                column_tile * LINEAR_COLUMN_TILE,
                D,
                1,
                GELU_ALGORITHM,
                LINEAR_REDUCTION_TILE,
                LINEAR_ROW_TILE,
                LINEAR_COLUMN_TILE,
                ALL_VALID,
            )
            if _increment_dependency(
                ffn_in_dependencies + layer * linear_row_tiles + row_tile,
                linear_column_tiles,
            ):
                for next_column in range(0, linear_column_tiles):
                    linear_index = row_tile * linear_column_tiles + next_column
                    _enqueue(
                        scheduler_ptr,
                        _task_descriptor(
                            TASK_FFN_OUT, layer, linear_index
                        ),
                    )

        elif task_type == TASK_FFN_OUT:
            row_tile = task // linear_column_tiles
            column_tile = task % linear_column_tiles
            _linear_tile(
                v,
                output_ptr,
                weights + JIT_FFN_OUT_WEIGHT,
                weights + JIT_FFN_OUT_BIAS,
                q,
                valid_ptr,
                row_tile * LINEAR_ROW_TILE,
                column_tile * LINEAR_COLUMN_TILE,
                D,
                3,
                GELU_ALGORITHM,
                LINEAR_REDUCTION_TILE,
                LINEAR_ROW_TILE,
                LINEAR_COLUMN_TILE,
                ALL_VALID,
            )
            if _increment_dependency(
                ffn_out_dependencies + layer * linear_row_tiles + row_tile,
                linear_column_tiles,
            ):
                if layer < NUM_LAYERS - 1:
                    _enqueue(
                        scheduler_ptr,
                        _task_descriptor(TASK_LN1, layer + 1, row_tile),
                    )
                else:
                    _enqueue(
                        scheduler_ptr,
                        _task_descriptor(TASK_FINAL_NORM, 0, row_tile),
                    )

        else:
            final_norm_weight = packed_ptr + NUM_LAYERS * LAYER_WEIGHTS
            final_norm_bias = final_norm_weight + D
            row_start = task * NORM_ROW_TILE
            _layer_norm_half(
                output_ptr,
                output_ptr,
                output_ptr,
                final_norm_weight,
                final_norm_bias,
                row_start,
                D,
                NORM_ROW_TILE,
                NORM_ALGORITHM,
                False,
            )
            if not ALL_VALID:
                rows = row_start + tl.arange(0, NORM_ROW_TILE)
                columns = tl.arange(0, D)
                values = tl.load(
                    output_ptr + rows[:, None] * D + columns[None, :]
                )
                valid = tl.load(valid_ptr + rows)[:, None]
                tl.store(
                    output_ptr + rows[:, None] * D + columns[None, :],
                    tl.where(valid, values, 0.0),
                )

        tl.atomic_add(
            scheduler_ptr + DONE, 1, sem="release", scope="gpu"
        )
        descriptor = _dequeue(scheduler_ptr, total_tasks)


@triton.jit
def _completion_dag_megakernel(
    input_ptr,
    valid_ptr,
    packed_ptr,
    workspace_ptr,
    output_ptr,
    scheduler_ptr,
    B: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    HD: tl.constexpr,
    T: tl.constexpr,
    E: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    LAYER_WEIGHTS: tl.constexpr,
    SCALE: tl.constexpr,
    LINEAR_ROW_TILE: tl.constexpr,
    LINEAR_COLUMN_TILE: tl.constexpr,
    LINEAR_REDUCTION_TILE: tl.constexpr,
    ATTENTION_ROW_TILE: tl.constexpr,
    ATTENTION_REDUCTION_TILE: tl.constexpr,
    NORM_ROW_TILE: tl.constexpr,
    NORM_ALGORITHM: tl.constexpr,
    SOFTMAX_ALGORITHM: tl.constexpr,
    DIVISION_ALGORITHM: tl.constexpr,
    EXP_ALGORITHM: tl.constexpr,
    GELU_ALGORITHM: tl.constexpr,
    ALL_VALID: tl.constexpr,
):
    worker = tl.program_id(0)
    linear_column_tiles: tl.constexpr = D // LINEAR_COLUMN_TILE
    linear_row_tiles: tl.constexpr = T // LINEAR_ROW_TILE
    attention_query_tiles: tl.constexpr = S // ATTENTION_ROW_TILE
    attention_tasks: tl.constexpr = linear_row_tiles
    qkv_tasks: tl.constexpr = 3 * linear_row_tiles * linear_column_tiles
    total_tasks: tl.constexpr = (
        linear_row_tiles + NUM_LAYERS * (qkv_tasks + attention_tasks)
    )
    projection_groups: tl.constexpr = B if S == 128 else B // 2
    projection_dependencies = scheduler_ptr + QUEUE_BASE + QUEUE_CAPACITY

    # Publish the initial layer's independent row LayerNorm tasks.
    if worker == 0:
        initial = tl.arange(0, 32)
        descriptor = _task_descriptor(TASK_LN1, 0, initial)
        tl.store(
            scheduler_ptr + QUEUE_BASE + initial,
            descriptor + 1,
            mask=initial < linear_row_tiles,
        )
        tl.atomic_xchg(
            scheduler_ptr + TAIL,
            linear_row_tiles,
            sem="release",
            scope="gpu",
        )
        tl.atomic_xchg(
            scheduler_ptr + START, 1, sem="release", scope="gpu"
        )
    started = tl.atomic_add(
        scheduler_ptr + START, 0, sem="acquire", scope="gpu"
    )
    while started == 0:
        started = tl.atomic_add(
            scheduler_ptr + START, 0, sem="acquire", scope="gpu"
        )

    descriptor = _dequeue(scheduler_ptr, total_tasks)
    while descriptor >= 0:
        task_type = descriptor & TASK_TYPE_MASK
        layer = (descriptor >> TASK_TYPE_BITS) & TASK_LAYER_MASK
        task = descriptor >> TASK_INDEX_SHIFT
        weights = packed_ptr + layer * LAYER_WEIGHTS
        norm = workspace_ptr + layer * JIT_WORKSPACE_SLOTS_PER_LAYER * E
        q = norm + E
        k = q + E
        v = k + E
        residual = input_ptr if layer == 0 else output_ptr

        if task_type == TASK_LN1:
            _layer_norm_half(
                residual,
                norm,
                norm,
                weights + JIT_NORM1_WEIGHT,
                weights + JIT_NORM1_BIAS,
                task * NORM_ROW_TILE,
                D,
                NORM_ROW_TILE,
                NORM_ALGORITHM,
                False,
            )
            _publish_cta_writes()
            for branch in range(0, 3):
                for column_tile in range(0, linear_column_tiles):
                    qkv_index = (
                        (task * 3 + branch) * linear_column_tiles
                        + column_tile
                    )
                    _enqueue(
                        scheduler_ptr,
                        _task_descriptor(TASK_QKV, layer, qkv_index),
                    )

        elif task_type == TASK_QKV:
            column_tile = task % linear_column_tiles
            branch_tile = task // linear_column_tiles
            branch = branch_tile % 3
            row_tile = branch_tile // 3
            branch_stride = D * D + D
            _linear_tile(
                norm,
                q + branch * E,
                weights + JIT_Q_WEIGHT + branch * branch_stride,
                weights + JIT_Q_BIAS + branch * branch_stride,
                residual,
                valid_ptr,
                row_tile * LINEAR_ROW_TILE,
                column_tile * LINEAR_COLUMN_TILE,
                D,
                0,
                GELU_ALGORITHM,
                LINEAR_REDUCTION_TILE,
                LINEAR_ROW_TILE,
                LINEAR_COLUMN_TILE,
                ALL_VALID,
            )
            # Completion is published by one scalar lane, while the WGMMA
            # result is stored cooperatively.  Join the CTA before the
            # release atomic so consumers cannot observe a partially stored
            # Q/K/V tile.
            _publish_cta_writes()

            if S == 128:
                projection_group = row_tile // 2
                projection_target = 6 * linear_column_tiles
                first_sequence = projection_group
                sequences_in_group: tl.constexpr = 1
            else:
                projection_group = row_tile
                projection_target = 3 * linear_column_tiles
                first_sequence = projection_group * 2
                sequences_in_group: tl.constexpr = 2
            if _increment_dependency(
                projection_dependencies
                + layer * projection_groups
                + projection_group,
                projection_target,
            ):
                # The final projection acquire observes every Q/K/V release
                # for this sequence group.  Publish all of its attention jobs
                # together; other groups and layers continue independently.
                if S == 128:
                    first_row_tile = projection_group * 2
                    attention_rows_in_group: tl.constexpr = 2
                else:
                    first_row_tile = projection_group
                    attention_rows_in_group: tl.constexpr = 1
                for row_in_group in range(0, attention_rows_in_group):
                    _enqueue(
                        scheduler_ptr,
                        _task_descriptor(
                            TASK_ATTENTION,
                            layer,
                            first_row_tile + row_in_group,
                        ),
                    )

        else:
            row_tile = task
            if S == 128:
                first_sequence = row_tile // attention_query_tiles
                query_tile = row_tile % attention_query_tiles
                sequences_in_attention_task: tl.constexpr = 1
            else:
                first_sequence = row_tile * 2
                query_tile = 0
                sequences_in_attention_task: tl.constexpr = 2
            for sequence_in_task in range(0, sequences_in_attention_task):
                sequence = first_sequence + sequence_in_task
                sequence_offset = sequence * S * D
                for head in range(0, H):
                    if S == 128 and query_tile == 0:
                        _attention_mxhd(
                            q + sequence_offset,
                            k + sequence_offset,
                            v + sequence_offset,
                            norm + sequence_offset,
                            valid_ptr + sequence * S,
                            workspace_ptr,
                            workspace_ptr,
                            workspace_ptr,
                            workspace_ptr,
                            0,
                            head,
                            S,
                            D,
                            HD,
                            64,
                            SCALE,
                            False,
                            SOFTMAX_ALGORITHM,
                            DIVISION_ALGORITHM,
                            EXP_ALGORITHM,
                            ATTENTION_REDUCTION_TILE,
                            ATTENTION_ROW_TILE,
                            ALL_VALID,
                        )
                    else:
                        _attention_mxhd(
                            q + sequence_offset,
                            k + sequence_offset,
                            v + sequence_offset,
                            norm + sequence_offset,
                            valid_ptr + sequence * S,
                            workspace_ptr,
                            workspace_ptr,
                            workspace_ptr,
                            workspace_ptr,
                            query_tile * ATTENTION_ROW_TILE,
                            head,
                            S,
                            D,
                            HD,
                            S,
                            SCALE,
                            False,
                            SOFTMAX_ALGORITHM,
                            DIVISION_ALGORITHM,
                            EXP_ALGORITHM,
                            ATTENTION_REDUCTION_TILE,
                            ATTENTION_ROW_TILE,
                            ALL_VALID,
                        )

            # Context columns are likewise produced cooperatively and then
            # consumed by the same CTA's full-row output projection.
            _publish_cta_writes()
            if True:
                # The winning acquire observes every head's release sequence;
                # distribute that visibility across the consumer CTA before
                # loading the complete context row.
                row_start = row_tile * LINEAR_ROW_TILE
                # Once a complete M64 context row is ready, its output, norm,
                # and FFN chain is strictly row-local.  Executing that chain
                # in the releasing CTA avoids nine queue operations while
                # other CTAs continue QKV/attention work independently.
                _linear_tile(
                    norm,
                    q,
                    weights + JIT_OUT_WEIGHT,
                    weights + JIT_OUT_BIAS,
                    residual,
                    valid_ptr,
                    row_start,
                    0,
                    D,
                    2,
                    GELU_ALGORITHM,
                    LINEAR_REDUCTION_TILE,
                    LINEAR_ROW_TILE,
                    LINEAR_COLUMN_TILE,
                    ALL_VALID,
                )
                _publish_cta_writes()
                _layer_norm_half(
                    q,
                    norm,
                    norm,
                    weights + JIT_NORM2_WEIGHT,
                    weights + JIT_NORM2_BIAS,
                    row_start,
                    D,
                    NORM_ROW_TILE,
                    NORM_ALGORITHM,
                    False,
                )
                _publish_cta_writes()
                _linear_tile(
                    norm,
                    v,
                    weights + JIT_FFN_IN_WEIGHT,
                    weights + JIT_FFN_IN_BIAS,
                    norm,
                    valid_ptr,
                    row_start,
                    0,
                    D,
                    1,
                    GELU_ALGORITHM,
                    LINEAR_REDUCTION_TILE,
                    LINEAR_ROW_TILE,
                    LINEAR_COLUMN_TILE,
                    ALL_VALID,
                )
                _publish_cta_writes()
                _linear_tile(
                    v,
                    output_ptr,
                    weights + JIT_FFN_OUT_WEIGHT,
                    weights + JIT_FFN_OUT_BIAS,
                    q,
                    valid_ptr,
                    row_start,
                    0,
                    D,
                    3,
                    GELU_ALGORITHM,
                    LINEAR_REDUCTION_TILE,
                    LINEAR_ROW_TILE,
                    LINEAR_COLUMN_TILE,
                    ALL_VALID,
                )
                _publish_cta_writes()

                if layer < NUM_LAYERS - 1:
                    next_weights = weights + LAYER_WEIGHTS
                    next_norm = (
                        workspace_ptr
                        + (layer + 1) * JIT_WORKSPACE_SLOTS_PER_LAYER * E
                    )
                    _layer_norm_half(
                        output_ptr,
                        next_norm,
                        next_norm,
                        next_weights + JIT_NORM1_WEIGHT,
                        next_weights + JIT_NORM1_BIAS,
                        row_start,
                        D,
                        NORM_ROW_TILE,
                        NORM_ALGORITHM,
                        False,
                    )
                    _publish_cta_writes()
                    for branch in range(0, 3):
                        for column_tile in range(0, linear_column_tiles):
                            qkv_index = (
                                (row_tile * 3 + branch)
                                * linear_column_tiles
                                + column_tile
                            )
                            _enqueue(
                                scheduler_ptr,
                                _task_descriptor(
                                    TASK_QKV, layer + 1, qkv_index
                                ),
                            )
                else:
                    final_norm_weight = (
                        packed_ptr + NUM_LAYERS * LAYER_WEIGHTS
                    )
                    final_norm_bias = final_norm_weight + D
                    _layer_norm_half(
                        output_ptr,
                        output_ptr,
                        output_ptr,
                        final_norm_weight,
                        final_norm_bias,
                        row_start,
                        D,
                        NORM_ROW_TILE,
                        NORM_ALGORITHM,
                        False,
                    )
                    if not ALL_VALID:
                        rows = row_start + tl.arange(0, NORM_ROW_TILE)
                        columns = tl.arange(0, D)
                        values = tl.load(
                            output_ptr
                            + rows[:, None] * D
                            + columns[None, :]
                        )
                        valid = tl.load(valid_ptr + rows)[:, None]
                        tl.store(
                            output_ptr
                            + rows[:, None] * D
                            + columns[None, :],
                            tl.where(valid, values, 0.0),
                        )

        tl.atomic_add(
            scheduler_ptr + DONE, 1, sem="release", scope="gpu"
        )
        descriptor = _dequeue(scheduler_ptr, total_tasks)


@triton.jit
def _parallel_s128_tail_half(
    norm,
    q,
    v,
    output_ptr,
    weights,
    residual,
    valid_ptr,
    packed_ptr,
    row_start,
    D: tl.constexpr,
    LINEAR_REDUCTION_TILE: tl.constexpr,
    NORM_ALGORITHM: tl.constexpr,
    GELU_ALGORITHM: tl.constexpr,
    ALL_VALID: tl.constexpr,
    FINAL_LAYER: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    LAYER_WEIGHTS: tl.constexpr,
):
    _linear_tile(
        norm,
        q,
        weights + JIT_OUT_WEIGHT,
        weights + JIT_OUT_BIAS,
        residual,
        valid_ptr,
        row_start,
        0,
        D,
        2,
        GELU_ALGORITHM,
        LINEAR_REDUCTION_TILE,
        64,
        128,
        ALL_VALID,
    )
    tl.debug_barrier()
    _layer_norm_half(
        q,
        norm,
        norm,
        weights + JIT_NORM2_WEIGHT,
        weights + JIT_NORM2_BIAS,
        row_start,
        D,
        64,
        NORM_ALGORITHM,
        False,
    )
    tl.debug_barrier()
    _linear_tile(
        norm,
        v,
        weights + JIT_FFN_IN_WEIGHT,
        weights + JIT_FFN_IN_BIAS,
        norm,
        valid_ptr,
        row_start,
        0,
        D,
        1,
        GELU_ALGORITHM,
        LINEAR_REDUCTION_TILE,
        64,
        128,
        ALL_VALID,
    )
    tl.debug_barrier()
    _linear_tile(
        v,
        output_ptr,
        weights + JIT_FFN_OUT_WEIGHT,
        weights + JIT_FFN_OUT_BIAS,
        q,
        valid_ptr,
        row_start,
        0,
        D,
        3,
        GELU_ALGORITHM,
        LINEAR_REDUCTION_TILE,
        64,
        128,
        ALL_VALID,
    )
    tl.debug_barrier()
    if FINAL_LAYER:
        final_norm_weight = packed_ptr + NUM_LAYERS * LAYER_WEIGHTS
        final_norm_bias = final_norm_weight + D
        _layer_norm_half(
            output_ptr,
            output_ptr,
            output_ptr,
            final_norm_weight,
            final_norm_bias,
            row_start,
            D,
            64,
            NORM_ALGORITHM,
            False,
        )
        if not ALL_VALID:
            rows = row_start + tl.arange(0, 64)
            columns = tl.arange(0, D)
            values = tl.load(
                output_ptr + rows[:, None] * D + columns[None, :]
            )
            valid = tl.load(valid_ptr + rows)[:, None]
            tl.store(
                output_ptr + rows[:, None] * D + columns[None, :],
                tl.where(valid, values, 0.0),
            )
    _publish_cta_writes()


@triton.jit
def _static_sequence_dag_megakernel(
    input_ptr,
    valid_ptr,
    packed_ptr,
    workspace_ptr,
    output_ptr,
    scheduler_ptr,
    launch_epoch,
    B: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    HD: tl.constexpr,
    T: tl.constexpr,
    E: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    LAYER_WEIGHTS: tl.constexpr,
    SCALE: tl.constexpr,
    LINEAR_ROW_TILE: tl.constexpr,
    LINEAR_COLUMN_TILE: tl.constexpr,
    LINEAR_REDUCTION_TILE: tl.constexpr,
    ATTENTION_ROW_TILE: tl.constexpr,
    ATTENTION_REDUCTION_TILE: tl.constexpr,
    NORM_ROW_TILE: tl.constexpr,
    NORM_ALGORITHM: tl.constexpr,
    SOFTMAX_ALGORITHM: tl.constexpr,
    DIVISION_ALGORITHM: tl.constexpr,
    EXP_ALGORITHM: tl.constexpr,
    GELU_ALGORITHM: tl.constexpr,
    ALL_VALID: tl.constexpr,
    PARALLEL_TAIL: tl.constexpr,
    PARALLEL_S32_ATTENTION: tl.constexpr,
):
    worker = tl.program_id(0)
    if S == 128:
        groups: tl.constexpr = B
        rows_per_group: tl.constexpr = S // LINEAR_ROW_TILE
        roles: tl.constexpr = 3 * rows_per_group
        group = worker // roles
        role = worker % roles
        branch = role // rows_per_group
        local_row = role % rows_per_group
    else:
        groups: tl.constexpr = B // 2
        roles: tl.constexpr = 3
        rows_per_group: tl.constexpr = 1
        group = worker // roles
        role = worker % roles
        branch = role
        local_row = 0

    qkv_counts = scheduler_ptr
    layer_epochs = qkv_counts + groups
    norm_ready = layer_epochs + groups
    attention_counts = norm_ready + groups * rows_per_group
    tail_counts = attention_counts + groups
    start_epochs = tail_counts + groups

    # Initialize only this independent sequence group's scheduler state.  A
    # host-provided monotonically increasing epoch distinguishes the new
    # launch from stale state, eliminating the separate device memset launch.
    if role == 0:
        tl.atomic_xchg(
            qkv_counts + group, 0, sem="relaxed", scope="gpu"
        )
        tl.atomic_xchg(
            layer_epochs + group, 0, sem="relaxed", scope="gpu"
        )
        tl.atomic_xchg(
            attention_counts + group, 0, sem="relaxed", scope="gpu"
        )
        tl.atomic_xchg(
            tail_counts + group, 0, sem="relaxed", scope="gpu"
        )
        for group_row in range(0, rows_per_group):
            tl.atomic_xchg(
                norm_ready + group * rows_per_group + group_row,
                0,
                sem="relaxed",
                scope="gpu",
            )
        tl.atomic_xchg(
            start_epochs + group,
            launch_epoch,
            sem="release",
            scope="gpu",
        )
    else:
        started = tl.atomic_add(
            start_epochs + group, 0, sem="acquire", scope="gpu"
        )
        while started != launch_epoch:
            started = tl.atomic_add(
                start_epochs + group, 0, sem="acquire", scope="gpu"
            )
    row_tile = group * rows_per_group + local_row
    row_start = row_tile * LINEAR_ROW_TILE
    norm = workspace_ptr
    q = norm + E
    k = q + E
    v = k + E
    branch_stride: tl.constexpr = D * D + D

    for layer in range(NUM_LAYERS):
        weights = packed_ptr + layer * LAYER_WEIGHTS
        residual = input_ptr if layer == 0 else output_ptr

        # Q owns LayerNorm production for its row.  K/V roles wait only for
        # that row, so separate sequence groups naturally drift across phases.
        if branch == 0:
            for norm_offset in range(0, LINEAR_ROW_TILE, NORM_ROW_TILE):
                _layer_norm_half(
                    residual,
                    norm,
                    norm,
                    weights + JIT_NORM1_WEIGHT,
                    weights + JIT_NORM1_BIAS,
                    row_start + norm_offset,
                    D,
                    NORM_ROW_TILE,
                    NORM_ALGORITHM,
                    False,
                )
            _publish_cta_writes()
            tl.atomic_xchg(
                norm_ready + group * rows_per_group + local_row,
                layer + 1,
                sem="release",
                scope="gpu",
            )
        else:
            ready = tl.atomic_add(
                norm_ready + group * rows_per_group + local_row,
                0,
                sem="acquire",
                scope="gpu",
            )
            while ready < layer + 1:
                ready = tl.atomic_add(
                    norm_ready + group * rows_per_group + local_row,
                    0,
                    sem="acquire",
                    scope="gpu",
                )

        _linear_tile(
            norm,
            q + branch * E,
            weights + JIT_Q_WEIGHT + branch * branch_stride,
            weights + JIT_Q_BIAS + branch * branch_stride,
            residual,
            valid_ptr,
            row_start,
            0,
            D,
            0,
            GELU_ALGORITHM,
            LINEAR_REDUCTION_TILE,
            LINEAR_ROW_TILE,
            LINEAR_COLUMN_TILE,
            ALL_VALID,
        )
        _publish_cta_writes()
        tl.atomic_add(
            qkv_counts + group, 1, sem="release", scope="gpu"
        )

        if role == 0:
            complete = tl.atomic_add(
                qkv_counts + group, 0, sem="acquire", scope="gpu"
            )
            while complete < roles:
                complete = tl.atomic_add(
                    qkv_counts + group, 0, sem="acquire", scope="gpu"
                )
            _publish_cta_writes()

            # Attention and the residual/FFN tail own complete M64 rows.
            # Other sequence groups continue their QKV or prior/later layers
            # while this CTA advances its group.
            if S == 128 and LINEAR_ROW_TILE == 128:
                attention_complete = tl.atomic_add(
                    attention_counts + group,
                    0,
                    sem="acquire",
                    scope="gpu",
                )
                while attention_complete < 2:
                    attention_complete = tl.atomic_add(
                        attention_counts + group,
                        0,
                        sem="acquire",
                        scope="gpu",
                    )
                _publish_cta_writes()
            elif S == 128:
                # Both query halves consume the same V tensor.  Produce both
                # contexts before either FFN tail aliases V storage.
                for query_tile in range(0, S // ATTENTION_ROW_TILE):
                    sequence_offset = group * S * D
                    for head in range(0, H):
                        if ATTENTION_ROW_TILE == 64 and query_tile == 0:
                            _attention_mxhd(
                                q + sequence_offset,
                                k + sequence_offset,
                                v + sequence_offset,
                                norm + sequence_offset,
                                valid_ptr + group * S,
                                workspace_ptr,
                                workspace_ptr,
                                workspace_ptr,
                                workspace_ptr,
                                0,
                                head,
                                S,
                                D,
                                HD,
                                ATTENTION_ROW_TILE,
                                SCALE,
                                False,
                                SOFTMAX_ALGORITHM,
                                DIVISION_ALGORITHM,
                                EXP_ALGORITHM,
                                ATTENTION_REDUCTION_TILE,
                                ATTENTION_ROW_TILE,
                                ALL_VALID,
                            )
                        else:
                            _attention_mxhd(
                                q + sequence_offset,
                                k + sequence_offset,
                                v + sequence_offset,
                                norm + sequence_offset,
                                valid_ptr + group * S,
                                workspace_ptr,
                                workspace_ptr,
                                workspace_ptr,
                                workspace_ptr,
                                query_tile * ATTENTION_ROW_TILE,
                                head,
                                S,
                                D,
                                HD,
                                S,
                                SCALE,
                                False,
                                SOFTMAX_ALGORITHM,
                                DIVISION_ALGORITHM,
                                EXP_ALGORITHM,
                                ATTENTION_REDUCTION_TILE,
                                ATTENTION_ROW_TILE,
                                ALL_VALID,
                            )
                tl.debug_barrier()
            if S == 128 and LINEAR_ROW_TILE == 128 and PARALLEL_TAIL:
                role0_rows: tl.constexpr = 0
            else:
                role0_rows: tl.constexpr = rows_per_group
            for group_row in range(0, role0_rows):
                owned_row_tile = group * rows_per_group + group_row
                owned_row_start = owned_row_tile * LINEAR_ROW_TILE
                if S == 128:
                    first_sequence = group
                    query_tile = group_row
                    sequences_in_row: tl.constexpr = 0
                else:
                    first_sequence = group * 2
                    query_tile = 0
                    sequences_in_row: tl.constexpr = (
                        1 if PARALLEL_S32_ATTENTION else 2
                    )
                for sequence_in_row in range(0, sequences_in_row):
                    sequence = first_sequence + sequence_in_row
                    sequence_offset = sequence * S * D
                    for head in range(0, H):
                        if S == 128 and query_tile == 0:
                            _attention_mxhd(
                                q + sequence_offset,
                                k + sequence_offset,
                                v + sequence_offset,
                                norm + sequence_offset,
                                valid_ptr + sequence * S,
                                workspace_ptr,
                                workspace_ptr,
                                workspace_ptr,
                                workspace_ptr,
                                0,
                                head,
                                S,
                                D,
                                HD,
                                64,
                                SCALE,
                                False,
                                SOFTMAX_ALGORITHM,
                                DIVISION_ALGORITHM,
                                EXP_ALGORITHM,
                                ATTENTION_REDUCTION_TILE,
                                ATTENTION_ROW_TILE,
                                ALL_VALID,
                            )
                        else:
                            _attention_mxhd(
                                q + sequence_offset,
                                k + sequence_offset,
                                v + sequence_offset,
                                norm + sequence_offset,
                                valid_ptr + sequence * S,
                                workspace_ptr,
                                workspace_ptr,
                                workspace_ptr,
                                workspace_ptr,
                                query_tile * ATTENTION_ROW_TILE,
                                head,
                                S,
                                D,
                                HD,
                                S,
                                SCALE,
                                False,
                                SOFTMAX_ALGORITHM,
                                DIVISION_ALGORITHM,
                                EXP_ALGORITHM,
                                ATTENTION_REDUCTION_TILE,
                                ATTENTION_ROW_TILE,
                                ALL_VALID,
                            )
                if S == 32 and PARALLEL_S32_ATTENTION:
                    _publish_cta_writes()
                    tl.atomic_add(
                        attention_counts + group,
                        1,
                        sem="release",
                        scope="gpu",
                    )
                    attention_complete = tl.atomic_add(
                        attention_counts + group,
                        0,
                        sem="acquire",
                        scope="gpu",
                    )
                    while attention_complete < 2:
                        attention_complete = tl.atomic_add(
                            attention_counts + group,
                            0,
                            sem="acquire",
                            scope="gpu",
                        )
                    _publish_cta_writes()
                else:
                    tl.debug_barrier()
                _linear_tile(
                    norm,
                    q,
                    weights + JIT_OUT_WEIGHT,
                    weights + JIT_OUT_BIAS,
                    residual,
                    valid_ptr,
                    owned_row_start,
                    0,
                    D,
                    2,
                    GELU_ALGORITHM,
                    LINEAR_REDUCTION_TILE,
                    LINEAR_ROW_TILE,
                    LINEAR_COLUMN_TILE,
                    ALL_VALID,
                )
                tl.debug_barrier()
                for norm_offset in range(0, LINEAR_ROW_TILE, NORM_ROW_TILE):
                    _layer_norm_half(
                        q,
                        norm,
                        norm,
                        weights + JIT_NORM2_WEIGHT,
                        weights + JIT_NORM2_BIAS,
                        owned_row_start + norm_offset,
                        D,
                        NORM_ROW_TILE,
                        NORM_ALGORITHM,
                        False,
                    )
                tl.debug_barrier()
                _linear_tile(
                    norm,
                    v,
                    weights + JIT_FFN_IN_WEIGHT,
                    weights + JIT_FFN_IN_BIAS,
                    norm,
                    valid_ptr,
                    owned_row_start,
                    0,
                    D,
                    1,
                    GELU_ALGORITHM,
                    LINEAR_REDUCTION_TILE,
                    LINEAR_ROW_TILE,
                    LINEAR_COLUMN_TILE,
                    ALL_VALID,
                )
                tl.debug_barrier()
                _linear_tile(
                    v,
                    output_ptr,
                    weights + JIT_FFN_OUT_WEIGHT,
                    weights + JIT_FFN_OUT_BIAS,
                    q,
                    valid_ptr,
                    owned_row_start,
                    0,
                    D,
                    3,
                    GELU_ALGORITHM,
                    LINEAR_REDUCTION_TILE,
                    LINEAR_ROW_TILE,
                    LINEAR_COLUMN_TILE,
                    ALL_VALID,
                )
                _publish_cta_writes()

            if layer == NUM_LAYERS - 1 and not PARALLEL_TAIL:
                final_norm_weight = packed_ptr + NUM_LAYERS * LAYER_WEIGHTS
                final_norm_bias = final_norm_weight + D
                for group_row in range(0, rows_per_group):
                    owned_row_start = (
                        group * rows_per_group + group_row
                    ) * LINEAR_ROW_TILE
                    for norm_offset in range(
                        0, LINEAR_ROW_TILE, NORM_ROW_TILE
                    ):
                        _layer_norm_half(
                            output_ptr,
                            output_ptr,
                            output_ptr,
                            final_norm_weight,
                            final_norm_bias,
                            owned_row_start + norm_offset,
                            D,
                            NORM_ROW_TILE,
                            NORM_ALGORITHM,
                            False,
                        )
                    if not ALL_VALID:
                        for norm_offset in range(
                            0, LINEAR_ROW_TILE, NORM_ROW_TILE
                        ):
                            rows = (
                                owned_row_start
                                + norm_offset
                                + tl.arange(0, NORM_ROW_TILE)
                            )
                            columns = tl.arange(0, D)
                            values = tl.load(
                                output_ptr
                                + rows[:, None] * D
                                + columns[None, :]
                            )
                            valid = tl.load(valid_ptr + rows)[:, None]
                            tl.store(
                                output_ptr
                                + rows[:, None] * D
                                + columns[None, :],
                                tl.where(valid, values, 0.0),
                            )
                _publish_cta_writes()

            if S == 128 and LINEAR_ROW_TILE == 128 and PARALLEL_TAIL:
                tails_complete = tl.atomic_add(
                    tail_counts + group, 0, sem="acquire", scope="gpu"
                )
                while tails_complete < 2:
                    tails_complete = tl.atomic_add(
                        tail_counts + group,
                        0,
                        sem="acquire",
                        scope="gpu",
                    )
                _publish_cta_writes()

            tl.atomic_xchg(
                qkv_counts + group, 0, sem="relaxed", scope="gpu"
            )
            if (
                (S == 32 and PARALLEL_S32_ATTENTION)
                or (S == 128 and LINEAR_ROW_TILE == 128)
            ):
                tl.atomic_xchg(
                    attention_counts + group,
                    0,
                    sem="relaxed",
                    scope="gpu",
                )
            if S == 128 and LINEAR_ROW_TILE == 128 and PARALLEL_TAIL:
                tl.atomic_xchg(
                    tail_counts + group,
                    0,
                    sem="relaxed",
                    scope="gpu",
                )
            tl.atomic_xchg(
                layer_epochs + group,
                layer + 1,
                sem="release",
                scope="gpu",
            )
        else:
            if S == 128 and LINEAR_ROW_TILE == 128:
                complete = tl.atomic_add(
                    qkv_counts + group, 0, sem="acquire", scope="gpu"
                )
                while complete < roles:
                    complete = tl.atomic_add(
                        qkv_counts + group, 0, sem="acquire", scope="gpu"
                    )
                _publish_cta_writes()
                sequence_offset = group * S * D
                first_head = (role - 1) * 2
                for head_offset in range(0, 2):
                    head = first_head + head_offset
                    _attention_mxhd(
                        q + sequence_offset,
                        k + sequence_offset,
                        v + sequence_offset,
                        norm + sequence_offset,
                        valid_ptr + group * S,
                        workspace_ptr,
                        workspace_ptr,
                        workspace_ptr,
                        workspace_ptr,
                        0,
                        head,
                        S,
                        D,
                        HD,
                        S,
                        SCALE,
                        False,
                        SOFTMAX_ALGORITHM,
                        DIVISION_ALGORITHM,
                        EXP_ALGORITHM,
                        ATTENTION_REDUCTION_TILE,
                        ATTENTION_ROW_TILE,
                        ALL_VALID,
                    )
                _publish_cta_writes()
                tl.atomic_add(
                    attention_counts + group,
                    1,
                    sem="release",
                    scope="gpu",
                )
                if PARALLEL_TAIL:
                    attention_complete = tl.atomic_add(
                        attention_counts + group,
                        0,
                        sem="acquire",
                        scope="gpu",
                    )
                    while attention_complete < 2:
                        attention_complete = tl.atomic_add(
                            attention_counts + group,
                            0,
                            sem="acquire",
                            scope="gpu",
                        )
                    _publish_cta_writes()
                    half_row_start = group * S + (role - 1) * 64
                    _parallel_s128_tail_half(
                        norm,
                        q,
                        v,
                        output_ptr,
                        weights,
                        residual,
                        valid_ptr,
                        packed_ptr,
                        half_row_start,
                        D,
                        LINEAR_REDUCTION_TILE,
                        NORM_ALGORITHM,
                        GELU_ALGORITHM,
                        ALL_VALID,
                        layer == NUM_LAYERS - 1,
                        NUM_LAYERS,
                        LAYER_WEIGHTS,
                    )
                    tl.atomic_add(
                        tail_counts + group,
                        1,
                        sem="release",
                        scope="gpu",
                    )
            if S == 32 and role == 1 and PARALLEL_S32_ATTENTION:
                complete = tl.atomic_add(
                    qkv_counts + group, 0, sem="acquire", scope="gpu"
                )
                while complete < roles:
                    complete = tl.atomic_add(
                        qkv_counts + group, 0, sem="acquire", scope="gpu"
                    )
                _publish_cta_writes()
                sequence = group * 2 + 1
                sequence_offset = sequence * S * D
                for head in range(0, H):
                    _attention_mxhd(
                        q + sequence_offset,
                        k + sequence_offset,
                        v + sequence_offset,
                        norm + sequence_offset,
                        valid_ptr + sequence * S,
                        workspace_ptr,
                        workspace_ptr,
                        workspace_ptr,
                        workspace_ptr,
                        0,
                        head,
                        S,
                        D,
                        HD,
                        S,
                        SCALE,
                        False,
                        SOFTMAX_ALGORITHM,
                        DIVISION_ALGORITHM,
                        EXP_ALGORITHM,
                        ATTENTION_REDUCTION_TILE,
                        ATTENTION_ROW_TILE,
                        ALL_VALID,
                    )
                _publish_cta_writes()
                tl.atomic_add(
                    attention_counts + group,
                    1,
                    sem="release",
                    scope="gpu",
                )
            epoch = tl.atomic_add(
                layer_epochs + group, 0, sem="acquire", scope="gpu"
            )
            while epoch < layer + 1:
                epoch = tl.atomic_add(
                    layer_epochs + group, 0, sem="acquire", scope="gpu"
                )


def dag_megakernel_forward(
    value: torch.Tensor,
    valid_mask: torch.Tensor,
    packed_weights: torch.Tensor,
    scheduler: torch.Tensor,
    launch_epoch: int,
    *,
    all_valid: bool,
) -> torch.Tensor:
    batch, sequence, model = value.shape
    if not is_step_4_shape(value, HEADS):
        raise ValueError(
            "completion-DAG kernel supports cases 2, 3, 4, and 12 only; "
            f"got B={batch}, S={sequence}, D={model}"
        )
    tuning = resolved_dag_tuning(batch, sequence)
    if batch * sequence % tuning["linear_m"]:
        raise ValueError("TTTJ_DAG_LINEAR_M must divide the flattened token count")
    if MODEL % tuning["linear_n"]:
        raise ValueError("TTTJ_DAG_LINEAR_N must divide 128")
    if batch * sequence % tuning["norm_m"]:
        raise ValueError("TTTJ_DAG_NORM_M must divide the flattened token count")
    if HEAD_DIM % tuning["attention_k"]:
        raise ValueError("attention reduction tile must divide head dimension")
    if sequence % tuning["attention_m"]:
        raise ValueError("attention row tile must divide sequence length")
    if scheduler.dtype != torch.int32 or scheduler.numel() != SCHEDULER_ELEMENTS:
        raise ValueError(
            f"scheduler must contain {SCHEDULER_ELEMENTS} int32 CUDA values"
        )

    elements = batch * sequence * model
    workspace = torch.empty(
        (WORKSPACE_SLOTS, batch, sequence, model),
        device=value.device,
        dtype=value.dtype,
    )
    output = torch.empty_like(value)
    _static_sequence_dag_megakernel[(tuning["programs"],)](
        value,
        valid_mask,
        packed_weights,
        workspace,
        output,
        scheduler,
        launch_epoch,
        B=batch,
        S=sequence,
        D=MODEL,
        H=HEADS,
        HD=HEAD_DIM,
        T=batch * sequence,
        E=elements,
        NUM_LAYERS=LAYERS,
        LAYER_WEIGHTS=LAYER_STRIDE,
        SCALE=HEAD_DIM**-0.5,
        LINEAR_ROW_TILE=tuning["linear_m"],
        LINEAR_COLUMN_TILE=tuning["linear_n"],
        LINEAR_REDUCTION_TILE=tuning["linear_k"],
        ATTENTION_ROW_TILE=tuning["attention_m"],
        ATTENTION_REDUCTION_TILE=tuning["attention_k"],
        NORM_ROW_TILE=tuning["norm_m"],
        NORM_ALGORITHM=0,
        SOFTMAX_ALGORITHM=SOFTMAX_MODE,
        DIVISION_ALGORITHM=DIVISION_MODE,
        EXP_ALGORITHM=EXP_MODE,
        GELU_ALGORITHM=GELU_MODE,
        ALL_VALID=all_valid,
        PARALLEL_TAIL=DAG_PARALLEL_TAIL and sequence == 128,
        PARALLEL_S32_ATTENTION=DAG_PARALLEL_S32_ATTENTION,
        num_warps=tuning["num_warps"],
        num_stages=tuning["num_stages"],
    )
    return output
