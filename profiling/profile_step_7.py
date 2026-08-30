"""Single exact-attention invocation for Nsight Compute range profiling."""

from __future__ import annotations

import torch

from kernels.case13_hybrid import (
    CASE13_BATCH,
    CASE13_HEAD_DIM,
    CASE13_HEADS,
    CASE13_SEQUENCE,
    _compiled_causal_scores_additive,
    _compiled_probabilities_value,
)


def exact_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    additive_causal: torch.Tensor,
) -> torch.Tensor:
    scores = _compiled_causal_scores_additive(q, k, additive_causal)
    probabilities = torch.softmax(scores, dim=-1)
    return _compiled_probabilities_value(probabilities, v)


def main() -> int:
    torch.manual_seed(1234)
    q, k, v = [
        torch.randn(
            CASE13_BATCH,
            CASE13_HEADS,
            CASE13_SEQUENCE,
            CASE13_HEAD_DIM,
            device="cuda",
            dtype=torch.float16,
        )
        for _ in range(3)
    ]
    causal = torch.ones(
        CASE13_SEQUENCE,
        CASE13_SEQUENCE,
        device="cuda",
        dtype=torch.bool,
    ).tril()
    additive_causal = torch.zeros(
        CASE13_SEQUENCE,
        CASE13_SEQUENCE,
        device="cuda",
        dtype=torch.float16,
    ).masked_fill(~causal, float("-inf"))
    for _ in range(3):
        exact_attention(q, k, v, additive_causal)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("step7_exact_attention")
    output = exact_attention(q, k, v, additive_causal)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print(f"checksum={output.float().sum().item():.8g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
