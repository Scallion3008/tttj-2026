#!/usr/bin/env python3
"""Provider and numerical probes for step 8 / benchmark case 14 attention."""

from __future__ import annotations

import argparse
import gc
import math
import statistics

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


HEADS = 16
HEAD_DIM = 64


def _time(callable_, warmup: int, repetitions: int) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            callable_()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
        for start, end in zip(starts, ends):
            start.record()
            callable_()
            end.record()
        torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def explicit_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    sequence = q.shape[1]
    qh, kh, vh = (x.transpose(1, 2) for x in (q, k, v))
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * (HEAD_DIM**-0.5)
    causal = torch.ones(sequence, sequence, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    probabilities = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(probabilities, vh).transpose(1, 2)


def provider_call(
    provider: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    if provider == "fa3":
        from flash_attn_3.flash_attn_interface import flash_attn_func

        return flash_attn_func(q, k, v, causal=True)
    backend = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }[provider]
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
        ).transpose(1, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("fa3", "flash", "cudnn"), required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--sequence", type=int, default=100_000)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--accuracy-sequence", type=int, default=1024)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    properties = torch.cuda.get_device_properties(0)
    print(
        f"gpu={properties.name} sm={properties.multi_processor_count} "
        f"memory={properties.total_memory / 2**30:.2f} GiB "
        f"torch={torch.__version__} cudnn={torch.backends.cudnn.version()}"
    )
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # The exact materialized reference is feasible at this reduced sequence.
    aq, ak, av = [
        torch.randn(
            1,
            args.accuracy_sequence,
            HEADS,
            HEAD_DIM,
            device="cuda",
            dtype=torch.float16,
        )
        for _ in range(3)
    ]
    with torch.inference_mode():
        reference = explicit_attention(aq, ak, av)
        candidate = provider_call(args.provider, aq, ak, av)
        difference = (candidate.float() - reference.float()).abs()
        allowed = (difference <= 0.001) | (
            difference <= 0.01 * reference.float().abs()
        )
        print(
            f"accuracy S={args.accuracy_sequence}: "
            f"failed={int((~allowed).sum())}/{reference.numel()} "
            f"max_abs={float(difference.max()):.8g} "
            f"mean_abs={float(difference.mean()):.8g}"
        )
    del aq, ak, av, reference, candidate, difference, allowed
    gc.collect()
    torch.cuda.empty_cache()

    shape = (args.batch, args.sequence, HEADS, HEAD_DIM)
    q, k, v = [torch.empty(shape, device="cuda", dtype=torch.float16) for _ in range(3)]
    # Finite deterministic contents without paying for three large RNG state walks.
    q.zero_()
    k.zero_()
    v.fill_(0.125)
    call = lambda: provider_call(args.provider, q, k, v)
    try:
        samples = _time(call, args.warmup, args.repetitions)
    except Exception as error:
        print(f"provider_error={type(error).__name__}: {error}")
        raise
    useful_flops = (
        2.0
        * args.batch
        * args.sequence
        * (args.sequence + 1)
        * HEADS
        * HEAD_DIM
    )
    median_ms = statistics.median(samples)
    print(
        f"provider={args.provider} shape={shape} samples_ms={samples} "
        f"median_ms={median_ms:.6f} "
        f"useful_tflops={useful_flops / (median_ms * 1e9):.3f}"
    )
    print(f"peak_allocated_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}")


if __name__ == "__main__":
    main()
