"""Benchmark bit-exact compiler variants of case-13 score construction."""

from __future__ import annotations

import torch
import triton


BATCH = 64
HEADS = 4
SEQUENCE = 1024
HEAD_DIM = 32
SCALE = HEAD_DIM**-0.5


def scores_mask_input(
    q: torch.Tensor,
    k: torch.Tensor,
    causal: torch.Tensor,
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) * SCALE
    return scores.masked_fill(~causal, float("-inf"))


def scores_index_mask(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) * SCALE
    indices = torch.arange(SEQUENCE, device=q.device)
    return scores.masked_fill(
        indices[None, :] > indices[:, None],
        float("-inf"),
    )


def scores_additive_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    additive: torch.Tensor,
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) * SCALE
    return scores + additive


def pv_default(
    probabilities: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(probabilities, value)


def pv_autotune(
    probabilities: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(probabilities, value)


def main() -> int:
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0):
        raise RuntimeError(f"Hopper sm_90 is required, got {properties.name}")
    torch.manual_seed(1234)
    q, k, v = [
        torch.randn(
            BATCH,
            HEADS,
            SEQUENCE,
            HEAD_DIM,
            device="cuda",
            dtype=torch.float16,
        )
        for _ in range(3)
    ]
    causal = torch.ones(
        SEQUENCE, SEQUENCE, device="cuda", dtype=torch.bool
    ).tril()
    additive = torch.zeros(
        SEQUENCE, SEQUENCE, device="cuda", dtype=torch.float16
    ).masked_fill(~causal, float("-inf"))
    variants = {
        "input-default": torch.compile(
            scores_mask_input, fullgraph=True, dynamic=False, mode="default"
        ),
        "input-autotune": torch.compile(
            scores_mask_input,
            fullgraph=True,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        ),
        "index-default": torch.compile(
            scores_index_mask, fullgraph=True, dynamic=False, mode="default"
        ),
        "index-autotune": torch.compile(
            scores_index_mask,
            fullgraph=True,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        ),
        "add-default": torch.compile(
            scores_additive_mask, fullgraph=True, dynamic=False, mode="default"
        ),
        "add-autotune": torch.compile(
            scores_additive_mask,
            fullgraph=True,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        ),
    }
    pv_variants = {
        "pv-eager": pv_default,
        "pv-default": torch.compile(
            pv_default, fullgraph=True, dynamic=False, mode="default"
        ),
        "pv-autotune": torch.compile(
            pv_autotune,
            fullgraph=True,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        ),
    }
    arguments = {
        "input-default": (q, k, causal),
        "input-autotune": (q, k, causal),
        "index-default": (q, k),
        "index-autotune": (q, k),
        "add-default": (q, k, additive),
        "add-autotune": (q, k, additive),
    }

    with torch.inference_mode():
        expected_scores = scores_mask_input(q, k, causal)
        probabilities = torch.softmax(expected_scores, dim=-1)
        expected = torch.matmul(probabilities, v)
        print(f"gpu={properties.name} sms={properties.multi_processor_count}")
        for name, operation in variants.items():
            args = arguments[name]
            scores = operation(*args)
            output = torch.matmul(torch.softmax(scores, dim=-1), v)
            scores_exact = bool(torch.equal(expected_scores, scores))
            output_exact = bool(torch.equal(expected, output))
            median, low, high = triton.testing.do_bench(
                lambda operation=operation, args=args: operation(*args),
                warmup=25,
                rep=100,
                quantiles=[0.5, 0.2, 0.8],
            )
            full_median, full_low, full_high = triton.testing.do_bench(
                lambda operation=operation, args=args: torch.matmul(
                    torch.softmax(operation(*args), dim=-1), v
                ),
                warmup=25,
                rep=100,
                quantiles=[0.5, 0.2, 0.8],
            )
            print(
                f"{name:16s} scores_exact={scores_exact} "
                f"output_exact={output_exact} scores={float(median):.6f} ms "
                f"scores_p20={float(low):.6f} scores_p80={float(high):.6f} "
                f"attention={float(full_median):.6f} ms "
                f"attention_p20={float(full_low):.6f} "
                f"attention_p80={float(full_high):.6f}"
            )
        for name, operation in pv_variants.items():
            output = operation(probabilities, v)
            median, low, high = triton.testing.do_bench(
                lambda operation=operation: operation(probabilities, v),
                warmup=25,
                rep=100,
                quantiles=[0.5, 0.2, 0.8],
            )
            print(
                f"{name:16s} output_exact={bool(torch.equal(expected, output))} "
                f"median={float(median):.6f} ms p20={float(low):.6f} "
                f"p80={float(high):.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
