#!/usr/bin/env python3
"""Isolate S32 attention rounding against the PyTorch reference."""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl

from benchmark_step_4 import models_for_case
from benchmark_steps_1_2 import compare_stage
from fused_megakernel import _attention_mxhd
from torch_transformer_benchmark import generate_random_case


@triton.jit
def _s32_attention_checkpoint(
    q_ptr,
    k_ptr,
    v_ptr,
    valid_ptr,
    context_ptr,
    scores_ptr,
    probabilities_ptr,
    numerators_ptr,
    denominators_ptr,
):
    task = tl.program_id(0)
    sequence = task // 4
    head = task % 4
    sequence_elements: tl.constexpr = 32 * 128
    trace_elements: tl.constexpr = 4 * 32 * 32
    _attention_mxhd(
        q_ptr + sequence * sequence_elements,
        k_ptr + sequence * sequence_elements,
        v_ptr + sequence * sequence_elements,
        context_ptr + sequence * sequence_elements,
        valid_ptr + sequence * 32,
        scores_ptr + sequence * trace_elements,
        probabilities_ptr + sequence * trace_elements,
        numerators_ptr + sequence * trace_elements,
        denominators_ptr + sequence * 4 * 32,
        0,
        head,
        32,
        128,
        32,
        32,
        32**-0.5,
        True,
        0,
        4,
        0,
        32,
        32,
        False,
    )


def exact_difference(reference: torch.Tensor, candidate: torch.Tensor) -> str:
    difference = candidate.float() - reference.float()
    return (
        f"different={int((difference != 0).sum().item())}/{difference.numel()} "
        f"max_abs={float(difference.abs().max().item()):.7g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--padding-ratio", type=float, default=0.25)
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    baseline, _, _, config, _, _ = models_for_case(12)
    value, valid = generate_random_case(
        config,
        torch.device("cuda"),
        torch.float16,
        args.seed,
        args.padding_ratio,
        args.scale,
    )
    layer = baseline.layers[0]
    with torch.inference_mode():
        normalized = layer.norm1(value)
        q_linear = layer.attention.q_proj(normalized)
        k_linear = layer.attention.k_proj(normalized)
        v_linear = layer.attention.v_proj(normalized)
        q = layer.attention._split_heads(q_linear)
        k = layer.attention._split_heads(k_linear)
        v = layer.attention._split_heads(v_linear)
        reference_scores = torch.matmul(q, k.transpose(-2, -1))
        reference_scores = reference_scores * layer.attention.scale
        causal = torch.ones(32, 32, device="cuda", dtype=torch.bool).triu(1)
        reference_scores = reference_scores.masked_fill(causal, -float("inf"))
        reference_scores = reference_scores.masked_fill(
            ~valid[:, None, None, :], -float("inf")
        )
        reference_probabilities = torch.softmax(
            reference_scores.float(), dim=-1
        ).half()
        reference_context = torch.matmul(reference_probabilities, v)
        reference_context = (
            reference_context.transpose(1, 2).contiguous().view(64, 32, 128)
        )

    q_flat = q.transpose(1, 2).contiguous().view(64, 32, 128)
    k_flat = k.transpose(1, 2).contiguous().view(64, 32, 128)
    v_flat = v.transpose(1, 2).contiguous().view(64, 32, 128)
    context = torch.empty_like(q_flat)
    scores = torch.empty((64, 4, 32, 32), device="cuda", dtype=torch.float32)
    probabilities = torch.empty_like(scores, dtype=torch.float16)
    numerators = torch.empty_like(scores)
    denominators = torch.empty((64, 4, 32), device="cuda", dtype=torch.float32)
    _s32_attention_checkpoint[(64 * 4,)](
        q_flat,
        k_flat,
        v_flat,
        valid,
        context,
        scores,
        probabilities,
        numerators,
        denominators,
        num_warps=4,
        num_stages=3,
    )
    torch.cuda.synchronize()
    for name, reference, candidate in (
        ("scores", reference_scores, scores),
        ("probabilities", reference_probabilities, probabilities),
        ("context", reference_context, context),
    ):
        print(name, exact_difference(reference, candidate), compare_stage(reference, candidate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
