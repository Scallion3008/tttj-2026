#!/usr/bin/env python3
"""Compare the fused case-14 softmax directly with pinned PyTorch."""

import torch

from kernels.case14_softmax import exact_softmax


def main() -> None:
    torch.manual_seed(1234)
    for sequence in (2048, 4096, 8192, 100_000):
        rows = 32 if sequence == 100_000 else 1024
        scores = torch.randn(rows, sequence, device="cuda", dtype=torch.float16)
        positions = torch.arange(sequence, device="cuda")
        query_positions = torch.linspace(
            0, sequence - 1, rows, device="cuda", dtype=torch.int64
        )
        scores.masked_fill_(
            positions[None, :] > query_positions[:, None], -float("inf")
        )
        reference = torch.softmax(scores.float(), dim=-1).half()
        output = exact_softmax(scores)
        difference = (reference.float() - output.float()).abs()
        print(
            f"sequence={sequence} unequal={int((reference != output).sum())}/"
            f"{reference.numel()} max_abs={float(difference.max()):.8g} "
            f"mean_abs={float(difference.mean()):.8g}"
        )
    sequence = 100_000
    positions = torch.arange(sequence, device="cuda")
    for stride in (1, 8):
        rows = 4096
        query_positions = torch.arange(rows, device="cuda") * stride
        scores = torch.randn(rows, sequence, device="cuda", dtype=torch.float16)
        scores.masked_fill_(
            positions[None, :] > query_positions[:, None], -float("inf")
        )
        reference = torch.softmax(scores.float(), dim=-1).half()
        output = exact_softmax(scores)
        unequal = (reference != output).any(dim=-1)
        mismatched_rows = torch.nonzero(unequal).flatten()
        print(
            f"sequence={sequence} prefix_stride={stride} "
            f"unequal_rows={int(unequal.sum())}/{rows} "
            f"first_rows={mismatched_rows[:32].tolist()}"
        )
    for prefix in (1, 17, 127, 128, 129, 4095, 4096, 32768, 99999):
        rows = 32
        scores = torch.randn(rows, prefix, device="cuda", dtype=torch.float16)
        reference_input = torch.full(
            (rows, sequence), -float("inf"), device="cuda", dtype=torch.float16
        )
        reference_input[:, :prefix] = scores
        reference = torch.softmax(reference_input.float(), dim=-1).half()[:, :prefix]
        output = exact_softmax(scores, sequence)
        print(
            f"compressed_prefix={prefix} unequal={int((reference != output).sum())}/"
            f"{reference.numel()}"
        )
    rows = 320
    query_start = sequence - rows
    raw_scores = torch.randn(rows, sequence, device="cuda", dtype=torch.float16)
    reference_input = raw_scores.mul(0.125)
    query_positions = torch.arange(query_start, sequence, device="cuda")
    reference_input.masked_fill_(
        positions[None, :] > query_positions[:, None],
        -float("inf"),
    )
    reference = torch.softmax(reference_input.float(), dim=-1).half()
    output = exact_softmax(
        raw_scores.clone(),
        sequence,
        query_start=query_start,
        input_scale=0.125,
        inplace=True,
    )
    print(
        f"fused_scale_mask_inplace unequal={int((reference != output).sum())}/"
        f"{reference.numel()}"
    )
    fast_output = exact_softmax(
        raw_scores.clone(),
        sequence,
        query_start=query_start,
        input_scale=0.125,
        inplace=True,
        fast_exp=True,
    )
    fast_difference = (reference.float() - fast_output.float()).abs()
    print(
        f"fast_exp unequal={int((reference != fast_output).sum())}/"
        f"{reference.numel()} max_abs={float(fast_difference.max()):.8g}"
    )


if __name__ == "__main__":
    main()
