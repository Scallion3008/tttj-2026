"""Measure compiler-generated exact-attention alternatives for step 7."""

from __future__ import annotations

import argparse

import torch


B, H, S, D = 64, 4, 1024, 32


def explicit(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) * (D**-0.5)
    causal = torch.ones(S, S, device=q.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    probabilities = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(probabilities, v)


def scores_only(q: torch.Tensor, k: torch.Tensor, causal: torch.Tensor):
    scores = torch.matmul(q, k.transpose(-2, -1)) * (D**-0.5)
    return scores.masked_fill(~causal, float("-inf"))


def softmax_only(scores: torch.Tensor) -> torch.Tensor:
    return torch.softmax(scores, dim=-1)


def report(name: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    error = (expected.float() - actual.float()).abs()
    passed = (error <= 0.001) | (error <= 0.01 * expected.float().abs())
    print(
        f"{name:24s} failed={actual.numel() - int(passed.sum())} "
        f"unequal={int((actual != expected).sum())} "
        f"max_abs={float(error.max()):.7g} mean_abs={float(error.mean()):.7g}"
    )


def timing(name: str, function) -> None:
    for _ in range(5):
        function()
    events = []
    for _ in range(30):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        function()
        end.record()
        end.synchronize()
        events.append(begin.elapsed_time(end))
    events.sort()
    print(
        f"{name:24s} median={events[len(events)//2]:.6f} ms "
        f"p20={events[len(events)//5]:.6f} p80={events[4*len(events)//5]:.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(17)
    q, k, v = [
        torch.randn(B, S, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
        for _ in range(3)
    ]
    causal = torch.ones(S, S, device="cuda", dtype=torch.bool).tril()
    expected = explicit(q, k, v)
    compiled_full = torch.compile(
        explicit,
        fullgraph=True,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )
    compiled_scores = torch.compile(
        scores_only,
        fullgraph=True,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )
    compiled_softmax = torch.compile(
        softmax_only,
        fullgraph=True,
        dynamic=False,
        mode="default",
    )

    def split_compile():
        scores = compiled_scores(q, k, causal)
        probabilities = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        return torch.matmul(probabilities, v)

    def dtype_softmax():
        scores = compiled_scores(q, k, causal)
        probabilities = torch.softmax(
            scores, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        return torch.matmul(probabilities, v)

    def half_to_float_softmax():
        scores = compiled_scores(q, k, causal)
        probabilities = torch._softmax(scores, -1, True).to(q.dtype)
        return torch.matmul(probabilities, v)

    def half_softmax():
        scores = compiled_scores(q, k, causal)
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(probabilities, v)

    def compiled_half_softmax():
        scores = compiled_scores(q, k, causal)
        probabilities = compiled_softmax(scores)
        return torch.matmul(probabilities, v)

    def half_softmax_context():
        return half_softmax().transpose(1, 2).contiguous()

    def direct_context():
        scores = compiled_scores(q, k, causal)
        probabilities = torch.softmax(scores, dim=-1)
        context = torch.empty(
            B, S, H, D, device=q.device, dtype=q.dtype
        )
        torch.matmul(probabilities, v, out=context.transpose(1, 2))
        return context

    full = compiled_full(q, k, v)
    split = split_compile()
    dtype_result = dtype_softmax()
    report("compiled-full", expected, full)
    report("compiled-scores", expected, split)
    report("dtype-softmax", expected, dtype_result)
    try:
        half_to_float_result = half_to_float_softmax()
        report("half-to-float-softmax", expected, half_to_float_result)
    except Exception as error:
        print(f"half-to-float-softmax RUN_FAIL {error!r}")
    report("half-softmax", expected, half_softmax())
    report("compiled-half-softmax", expected, compiled_half_softmax())
    expected_context = expected.transpose(1, 2).contiguous()
    try:
        report("direct-context", expected_context, direct_context())
    except Exception as error:
        print(f"direct-context RUN_FAIL {error!r}")
    if args.skip_timing:
        return 0
    timing("explicit", lambda: explicit(q, k, v))
    timing("compiled-full", lambda: compiled_full(q, k, v))
    timing("compiled-scores", split_compile)
    timing("dtype-softmax", dtype_softmax)
    try:
        timing("half-to-float-softmax", half_to_float_softmax)
    except Exception:
        pass
    timing("half-softmax", half_softmax)
    timing("compiled-half-softmax", compiled_half_softmax)
    timing("half-softmax-context", half_softmax_context)
    try:
        timing("direct-context", direct_context)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
