#!/usr/bin/env python3
"""Localize case-14 fused-attention error at reference dtype boundaries."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from benchmarks.benchmark_step_8 import FULL_CONFIG
from benchmarks.torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    compare_outputs,
    generate_random_case,
)
from flash_attn_3.flash_attn_interface import flash_attn_func


HEADS = 16
HEAD_DIM = 64
SCALE = HEAD_DIM**-0.5


def _summary(name: str, reference: torch.Tensor, candidate: torch.Tensor) -> None:
    difference = (candidate.float() - reference.float()).abs()
    strict = compare_outputs(reference, candidate, rtol=0.01, atol=0.001)
    doubled = compare_outputs(reference, candidate, rtol=0.02, atol=0.002)
    unequal = int((reference != candidate).sum().item())
    print(
        f"{name}: unequal={unequal}/{reference.numel()} "
        f"strict_failed={strict.failed_elements} double_failed={doubled.failed_elements} "
        f"max_abs={float(difference.max()):.8g} "
        f"mean_abs={float(difference.mean()):.8g}"
    )


def _project(layer, normalized: torch.Tensor):
    batch, sequence = normalized.shape[:2]
    return tuple(
        projection(normalized).view(batch, sequence, HEADS, HEAD_DIM)
        for projection in (
            layer.attention.q_proj,
            layer.attention.k_proj,
            layer.attention.v_proj,
        )
    )


def _explicit_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    fp32_scores: bool = False,
    fp32_pv: bool = False,
):
    sequence = q.shape[1]
    qh, kh, vh = (value.transpose(1, 2) for value in (q, k, v))
    if fp32_scores:
        scores = torch.matmul(qh.float(), kh.float().transpose(-2, -1)) * SCALE
    else:
        # Preserve both FP16 boundaries in the organizer reference: the QK
        # GEMM output is rounded before the separate FP16 scale multiply.
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * SCALE
    causal = torch.ones(
        sequence, sequence, device=q.device, dtype=torch.bool
    ).tril_()
    scores.masked_fill_(~causal, float("-inf"))
    probabilities_float = torch.softmax(scores.float(), dim=-1)
    probabilities = probabilities_float.half()
    if fp32_pv:
        context = torch.matmul(probabilities.float(), vh.float()).half()
    else:
        context = torch.matmul(probabilities, vh)
    return context.transpose(1, 2).contiguous(), torch.logsumexp(
        scores.float(), dim=-1
    )


def _block(layer, value: torch.Tensor, provider: str, stages: dict[str, torch.Tensor]):
    normalized = layer.norm1(value)
    q, k, v = _project(layer, normalized)
    if provider == "exact":
        context, _ = _explicit_attention(q, k, v)
    else:
        context = flash_attn_func(q, k, v, causal=True)
    stages["context"] = context
    branch = layer.attention.out_proj(context.reshape_as(value))
    stages["attention_branch"] = branch
    attention_residual = value + branch
    stages["attention_residual"] = attention_residual
    normalized2 = layer.norm2(attention_residual)
    stages["norm2"] = normalized2
    ffn_branch = layer.ffn_out(
        F.gelu(layer.ffn_in(normalized2), approximate="none")
    )
    stages["ffn_branch"] = ffn_branch
    output = attention_residual + ffn_branch
    stages["block_output"] = output
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    properties = torch.cuda.get_device_properties(0)
    print(
        f"gpu={properties.name} sm={properties.multi_processor_count} "
        f"memory={properties.total_memory / 2**30:.2f} GiB"
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1234)
    model = BaselineTransformer(FULL_CONFIG).cuda().half().eval()

    config = TransformerConfig(
        batch_size=1,
        seq_len=args.sequence,
        d_model=1024,
        num_heads=HEADS,
        ffn_dim=1024,
        num_layers=2,
        causal=True,
    )
    for trial in range(args.trials):
        value, mask = generate_random_case(
            config,
            torch.device("cuda"),
            torch.float16,
            4321 + trial,
            0.0,
            1.0,
        )
        print(f"trial={trial}")
        with torch.inference_mode():
            normalized = model.layers[0].norm1(value)
            q, k, v = _project(model.layers[0], normalized)
            exact_context, exact_lse = _explicit_attention(q, k, v)
            fp32_score_context, fp32_score_lse = _explicit_attention(
                q, k, v, fp32_scores=True
            )
            fp32_pv_context, _ = _explicit_attention(q, k, v, fp32_pv=True)
            fa3_context, fa3_lse = flash_attn_func(
                q, k, v, causal=True, return_attn_probs=True
            )
            _summary("layer0.context.fa3", exact_context, fa3_context)
            _summary("layer0.context.fp32_score", exact_context, fp32_score_context)
            _summary("layer0.context.fp32_pv", exact_context, fp32_pv_context)
            lse_difference = (fa3_lse.float() - exact_lse.float()).abs()
            lse_fp32_difference = (fa3_lse.float() - fp32_score_lse.float()).abs()
            print(
                "layer0.lse.fa3_vs_half_scores: "
                f"max_abs={float(lse_difference.max()):.8g} "
                f"mean_abs={float(lse_difference.mean()):.8g}"
            )
            print(
                "layer0.lse.fa3_vs_fp32_scores: "
                f"max_abs={float(lse_fp32_difference.max()):.8g} "
                f"mean_abs={float(lse_fp32_difference.mean()):.8g}"
            )

            exact_value = value
            fa3_value = value
            for layer_index, layer in enumerate(model.layers):
                exact_layer_stages: dict[str, torch.Tensor] = {}
                fa3_layer_stages: dict[str, torch.Tensor] = {}
                exact_value = _block(layer, exact_value, "exact", exact_layer_stages)
                fa3_value = _block(layer, fa3_value, "fa3", fa3_layer_stages)
                for stage in (
                    "context",
                    "attention_branch",
                    "attention_residual",
                    "norm2",
                    "ffn_branch",
                    "block_output",
                ):
                    _summary(
                        f"layer{layer_index}.{stage}",
                        exact_layer_stages[stage],
                        fa3_layer_stages[stage],
                    )
            exact_output = model.final_norm(exact_value)
            fa3_output = model.final_norm(fa3_value)
            _summary("final_output", exact_output, fa3_output)
        del value, mask


if __name__ == "__main__":
    main()
