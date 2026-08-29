#!/usr/bin/env python3
"""Correctness, latency, and roofline benchmark for cases 2, 3, 4, and 12."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton

from dag_megakernel import resolved_dag_tuning
from sequence_resident import SequenceResidentTransformer
from torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    compare_outputs,
    generate_random_case,
)


MODEL = 128
HEADS = 4
FFN = 128
LAYERS = 4
DTYPE_BYTES = 2
H200_NVL_DENSE_FP16_TFLOPS = 835.5
H200_SXM_DENSE_FP16_TFLOPS = 989.5
H200_HBM_TBPS = 4.8
SHAPES = {
    2: (1, 128),
    3: (4, 128),
    4: (16, 128),
    12: (64, 32),
}


def useful_flops(batch: int, sequence: int) -> int:
    return LAYERS * (
        8 * batch * sequence * MODEL * MODEL
        + 4 * batch * sequence * MODEL * FFN
        + 2 * batch * sequence * (sequence + 1) * MODEL
    )


def packed_weight_bytes() -> int:
    layer_parameters = (
        4 * MODEL * MODEL
        + 2 * MODEL * FFN
        + 9 * MODEL
        + FFN
    )
    return DTYPE_BYTES * (LAYERS * layer_parameters + 2 * MODEL)


def roofline_bytes(batch: int, sequence: int) -> int:
    tensor_bytes = batch * sequence * MODEL * DTYPE_BYTES
    return (22 * LAYERS + 2) * tensor_bytes + packed_weight_bytes()


@dataclass(frozen=True)
class Roofline:
    batch: int
    sequence: int
    peak_compute_tflops: float
    peak_memory_tbps: float

    @property
    def arithmetic_intensity(self) -> float:
        return useful_flops(self.batch, self.sequence) / roofline_bytes(
            self.batch, self.sequence
        )

    @property
    def memory_line_tflops(self) -> float:
        return self.arithmetic_intensity * self.peak_memory_tbps

    @property
    def bound_tflops(self) -> float:
        return min(self.peak_compute_tflops, self.memory_line_tflops)

    @property
    def binding_resource(self) -> str:
        return (
            "compute"
            if self.peak_compute_tflops <= self.memory_line_tflops
            else "memory"
        )

    def percent(self, milliseconds: float) -> float:
        achieved = useful_flops(self.batch, self.sequence) / (milliseconds * 1e9)
        return 100.0 * achieved / self.bound_tflops

    def effective_tbps(self, milliseconds: float) -> float:
        return roofline_bytes(self.batch, self.sequence) / (milliseconds * 1e9)


class SDPATransformer(torch.nn.Module):
    def __init__(self, model: BaselineTransformer) -> None:
        super().__init__()
        self.model = model
        self._last_mask: torch.Tensor | None = None
        self._last_all_valid = False

    def forward(
        self, x: torch.Tensor, valid_token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, sequence, model = x.shape
        if valid_token_mask is None:
            all_valid = True
        elif valid_token_mask is self._last_mask:
            all_valid = self._last_all_valid
        else:
            self._last_mask = valid_token_mask
            self._last_all_valid = bool(valid_token_mask.all().item())
            all_valid = self._last_all_valid
        causal_mask = None
        if not all_valid:
            keys = torch.arange(sequence, device=x.device)[None, :]
            lengths = valid_token_mask.sum(dim=1)[:, None]
            valid_keys = keys < lengths
            causal = torch.ones(
                sequence, sequence, device=x.device, dtype=torch.bool
            ).tril()
            causal_mask = causal[None, None, :, :] & valid_keys[:, None, None, :]
        for layer in self.model.layers:
            normalized = layer.norm1(x)
            attention = layer.attention
            q = attention._split_heads(attention.q_proj(normalized))
            k = attention._split_heads(attention.k_proj(normalized))
            v = attention._split_heads(attention.v_proj(normalized))
            if all_valid:
                context = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True
                )
            else:
                context = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=causal_mask
                )
            context = context.transpose(1, 2).contiguous().view(batch, sequence, model)
            branch = attention.out_proj(context)
            if valid_token_mask is not None:
                branch = branch.masked_fill(~valid_token_mask[..., None], 0)
            x = x + branch
            x = x + layer.ffn_out(
                F.gelu(layer.ffn_in(layer.norm2(x)), approximate="none")
            )
            if valid_token_mask is not None:
                x = x.masked_fill(~valid_token_mask[..., None], 0)
        x = self.model.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


MODELS: dict[int, tuple[torch.nn.Module, ...]] = {}
TIMES: dict[tuple[int, str], tuple[float, float, float]] = {}
WARMUP = 25
REPETITIONS = 75
SEED = 1234
PEAK_COMPUTE = H200_NVL_DENSE_FP16_TFLOPS
PEAK_MEMORY = H200_HBM_TBPS


def models_for_case(case_number: int):
    if case_number not in MODELS:
        batch, sequence = SHAPES[case_number]
        config = TransformerConfig(
            batch_size=batch,
            seq_len=sequence,
            d_model=MODEL,
            num_heads=HEADS,
            ffn_dim=FFN,
            num_layers=LAYERS,
            causal=True,
        )
        torch.manual_seed(SEED)
        baseline = BaselineTransformer(config).cuda().half().eval()
        optimized = SequenceResidentTransformer(copy.deepcopy(baseline)).cuda().eval()
        optimized.prepare()
        sdpa = SDPATransformer(copy.deepcopy(baseline)).cuda().eval()
        value, valid = generate_random_case(
            config,
            torch.device("cuda"),
            torch.float16,
            SEED + 100_000 + case_number,
            0.0,
            1.0,
        )
        MODELS[case_number] = baseline, sdpa, optimized, config, value, valid
    return MODELS[case_number]


def measure(case_number: int, provider: str) -> tuple[float, float, float]:
    key = case_number, provider
    if key not in TIMES:
        baseline, sdpa, optimized, _, value, valid = models_for_case(case_number)
        model = {"dag": optimized, "sdpa": sdpa, "torch": baseline}[provider]

        def invoke() -> None:
            with torch.inference_mode():
                model(value, valid)

        median, low, high = triton.testing.do_bench(
            invoke,
            warmup=WARMUP,
            rep=REPETITIONS,
            quantiles=[0.5, 0.2, 0.8],
        )
        TIMES[key] = float(median), float(low), float(high)
    return TIMES[key]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["case_number"],
        x_vals=list(SHAPES),
        x_log=False,
        line_arg="provider",
        line_vals=["dag"],
        line_names=["completion-driven DAG"],
        styles=[("blue", "-")],
        ylabel="effective % of binding roofline",
        plot_name="step-4-h200-roofline",
        args={},
    )
)
def roofline_benchmark(case_number: int, provider: str):
    median, low, high = measure(case_number, provider)
    batch, sequence = SHAPES[case_number]
    roofline = Roofline(batch, sequence, PEAK_COMPUTE, PEAK_MEMORY)
    return roofline.percent(median), roofline.percent(high), roofline.percent(low)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["case_number"],
        x_vals=list(SHAPES),
        x_log=False,
        line_arg="provider",
        line_vals=["dag", "sdpa", "torch"],
        line_names=["completion-driven DAG", "Torch SDPA", "explicit Torch"],
        styles=[("blue", "-"), ("orange", "-"), ("green", "-")],
        ylabel="latency (ms)",
        plot_name="step-4-h200-latency",
        args={},
    )
)
def latency_benchmark(case_number: int, provider: str):
    return measure(case_number, provider)


def run_text_report(mark, cases: list[int]) -> None:
    benchmark = mark.benchmarks
    print(f"\n{benchmark.plot_name} ({benchmark.ylabel}):")
    print("case  " + "  ".join(f"{name:>22s}" for name in benchmark.line_names))
    for case_number in benchmark.x_vals:
        if case_number not in cases:
            continue
        values = []
        for provider in benchmark.line_vals:
            median, _, _ = mark.fn(
                case_number=case_number, provider=provider, **benchmark.args
            )
            values.append(median)
        print(
            f"{case_number:4d}  "
            + "  ".join(f"{value:22.4f}" for value in values)
        )


def run_accuracy(
    cases: list[int], trials: int, padding_ratios: list[float], scales: list[float]
) -> bool:
    print("\n=== Strict numerical gate ===")
    passed = True
    with torch.inference_mode():
        for case_number in cases:
            baseline, _, optimized, config, _, _ = models_for_case(case_number)
            for padding_ratio in padding_ratios:
                for scale in scales:
                    failed = 0
                    max_abs = 0.0
                    worst = None
                    for trial in range(trials):
                        value, valid = generate_random_case(
                            config,
                            torch.device("cuda"),
                            torch.float16,
                            SEED + trial + round(100 * padding_ratio) + round(scale * 1000),
                            padding_ratio,
                            scale,
                        )
                        result = compare_outputs(
                            baseline(value, valid),
                            optimized(value, valid),
                            rtol=0.01,
                            atol=0.001,
                        )
                        failed += result.failed_elements
                        if result.max_abs_error >= max_abs:
                            max_abs = result.max_abs_error
                            worst = (
                                result.worst_index,
                                result.reference_at_worst,
                                result.optimized_at_worst,
                            )
                    status = "PASS" if failed == 0 else "FAIL"
                    passed &= failed == 0
                    print(
                        f"case {case_number:2d} padding={padding_ratio:.2f} "
                        f"scale={scale:g} {status} failed={failed}/"
                        f"{trials * config.batch_size * config.seq_len * MODEL} "
                        f"max_abs={max_abs:.7g}"
                    )
                    if failed:
                        print(f"  worst={worst}")
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", type=int, choices=SHAPES, default=SHAPES)
    parser.add_argument("--accuracy-trials", type=int, default=3)
    parser.add_argument("--padding-ratios", nargs="+", type=float, default=(0.0,))
    parser.add_argument("--scales", nargs="+", type=float, default=(1.0,))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--peak-tflops", type=float)
    parser.add_argument("--peak-memory-tbps", type=float, default=H200_HBM_TBPS)
    return parser.parse_args()


def main() -> int:
    global WARMUP, REPETITIONS, SEED, PEAK_COMPUTE, PEAK_MEMORY
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("an NVIDIA GPU is required")
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0) or "H200" not in properties.name:
        raise RuntimeError(f"an H200 sm_90 GPU is required, got {properties.name}")
    SEED = args.seed
    WARMUP = 10 if args.quick else 100
    REPETITIONS = 30 if args.quick else 500
    PEAK_COMPUTE = args.peak_tflops or (
        H200_NVL_DENSE_FP16_TFLOPS
        if "NVL" in properties.name
        else H200_SXM_DENSE_FP16_TFLOPS
    )
    PEAK_MEMORY = args.peak_memory_tbps
    cases = list(dict.fromkeys(args.cases))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(
        f"torch={torch.__version__} triton={triton.__version__} "
        f"gpu={properties.name} sms={properties.multi_processor_count}"
    )
    for case_number in cases:
        batch, sequence = SHAPES[case_number]
        roofline = Roofline(batch, sequence, PEAK_COMPUTE, PEAK_MEMORY)
        print(
            f"case {case_number:2d} B={batch} S={sequence} tuning="
            f"{resolved_dag_tuning(batch, sequence)} flops="
            f"{useful_flops(batch, sequence) / 1e9:.6f}G bytes="
            f"{roofline_bytes(batch, sequence) / 1e6:.6f}M intensity="
            f"{roofline.arithmetic_intensity:.2f}F/B binding={roofline.binding_resource}"
        )
    scales = list(args.scales)
    if args.quick and scales == [1.0]:
        scales = [1.0]
    passed = run_accuracy(
        cases, args.accuracy_trials, list(args.padding_ratios), scales
    )
    run_text_report(latency_benchmark, cases)
    run_text_report(roofline_benchmark, cases)
    print("\nDetailed DAG metrics:")
    for case_number in cases:
        batch, sequence = SHAPES[case_number]
        milliseconds = measure(case_number, "dag")[0]
        roofline = Roofline(batch, sequence, PEAK_COMPUTE, PEAK_MEMORY)
        print(
            f"case {case_number:2d}: {milliseconds:.6f} ms, "
            f"{useful_flops(batch, sequence) / (milliseconds * 1e9):.3f} TFLOP/s, "
            f"{roofline.effective_tbps(milliseconds):.3f} TB/s, "
            f"{roofline.percent(milliseconds):.2f}% {roofline.binding_resource} roofline"
        )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
