#!/usr/bin/env python3
"""Correctness, latency, and roofline benchmark for cases 1, 9, and 10."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton

from benchmarks.benchmark_steps_1_2 import make_models
from benchmarks.torch_transformer_benchmark import compare_outputs, generate_random_case
from kernels.fused_megakernel import resolved_megakernel_tuning


BATCH = 64
SEQUENCE = 128
MODEL = 128
FFN = 128
LAYERS = 4
DTYPE_BYTES = 2

# NVIDIA quotes sparse FP16 Tensor Core rates on the H200 product page.  The
# dense rate is half of that figure.  The cluster's 141 GiB part identifies as
# H200 NVL, whose dense FP16 rate is 835.5 TFLOP/s.  Both H200 variants are
# specified at 4.8 TB/s HBM bandwidth.
H200_NVL_DENSE_FP16_TFLOPS = 835.5
H200_SXM_DENSE_FP16_TFLOPS = 989.5
H200_HBM_TBPS = 4.8
H100_NVL_DENSE_FP16_TFLOPS = 835.5
H100_NVL_HBM_TBPS = 3.9

CASE_FOR_HEADS = {4: 1, 1: 9, 2: 10}


def useful_flops() -> int:
    """Reference-model FLOPs, counting one FMA as two operations."""
    projections_and_ffn = LAYERS * (
        8 * BATCH * SEQUENCE * MODEL * MODEL
        + 4 * BATCH * SEQUENCE * MODEL * FFN
    )
    # QK and PV over the causal triangle, including its diagonal.
    causal_attention = (
        LAYERS * 2 * BATCH * SEQUENCE * (SEQUENCE + 1) * MODEL
    )
    return projections_and_ffn + causal_attention


def packed_weight_bytes() -> int:
    """One cold read of the packed FP16 parameters."""
    layer_parameters = (
        4 * MODEL * MODEL
        + 2 * MODEL * FFN
        + 9 * MODEL
        + FFN
    )
    parameters = LAYERS * layer_parameters + 2 * MODEL
    return DTYPE_BYTES * parameters


def activation_traffic_bytes() -> int:
    """Logical global activation traffic for the all-valid fused kernel.

    Each layer transfers 22 full tensors: LayerNorm1 (2), QKV (6), attention
    (4), output projection/residual (3), LayerNorm2 (2), FFN input (2), and
    FFN output/residual (3).  Final LayerNorm adds one read and one write.
    """
    tensor_bytes = BATCH * SEQUENCE * MODEL * DTYPE_BYTES
    return (22 * LAYERS + 2) * tensor_bytes


def roofline_bytes() -> int:
    """Logical activation traffic plus one packed-weight footprint."""
    return activation_traffic_bytes() + packed_weight_bytes()


@dataclass(frozen=True)
class Roofline:
    peak_compute_tflops: float
    peak_memory_tbps: float

    @property
    def arithmetic_intensity(self) -> float:
        return useful_flops() / roofline_bytes()

    @property
    def bound_tflops(self) -> float:
        memory_line = self.arithmetic_intensity * self.peak_memory_tbps
        return min(self.peak_compute_tflops, memory_line)

    @property
    def binding_resource(self) -> str:
        memory_line = self.arithmetic_intensity * self.peak_memory_tbps
        return "compute" if self.peak_compute_tflops <= memory_line else "memory"

    def percent(self, milliseconds: float) -> float:
        achieved_tflops = useful_flops() / (milliseconds * 1.0e9)
        return 100.0 * achieved_tflops / self.bound_tflops

    def effective_tbps(self, milliseconds: float) -> float:
        return roofline_bytes() / (milliseconds * 1.0e9)


MODELS: dict[
    int,
    tuple[
        torch.nn.Module,
        torch.nn.Module,
        torch.nn.Module,
        object,
        torch.Tensor,
        torch.Tensor,
    ],
] = {}
TIMES: dict[tuple[int, str], tuple[float, float, float]] = {}
ROOFLINE: Roofline
WARMUP: int
REPETITIONS: int
SEED: int


class SDPATransformer(torch.nn.Module):
    """Layerwise PyTorch model with only the attention core changed to SDPA."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self._last_valid_token_mask: torch.Tensor | None = None
        self._last_valid_token_mask_version: int | None = None
        self._last_mask_was_all_valid = False

    def _mask_is_all_valid(self, mask: torch.Tensor) -> bool:
        try:
            version: int | None = mask._version
        except RuntimeError:
            version = None
        if (
            self._last_valid_token_mask is not mask
            or self._last_valid_token_mask_version != version
        ):
            self._last_valid_token_mask = mask
            self._last_valid_token_mask_version = version
            self._last_mask_was_all_valid = bool(mask.all().item())
        return self._last_mask_was_all_valid

    def forward(
        self,
        value: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = value
        for layer in self.model.layers:
            normalized = layer.norm1(x)
            attention = layer.attention
            q = attention._split_heads(attention.q_proj(normalized))
            k = attention._split_heads(attention.k_proj(normalized))
            v = attention._split_heads(attention.v_proj(normalized))
            if (
                valid_token_mask is None
                or self._mask_is_all_valid(valid_token_mask)
            ):
                context = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=True,
                )
            else:
                causal = torch.ones(
                    (SEQUENCE, SEQUENCE),
                    device=x.device,
                    dtype=torch.bool,
                ).tril()
                attention_mask = (
                    causal[None, None, :, :]
                    & valid_token_mask[:, None, None, :]
                )
                context = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attention_mask,
                )
            context = (
                context.transpose(1, 2)
                .contiguous()
                .view(BATCH, SEQUENCE, MODEL)
            )
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


def models_for_heads(num_heads: int):
    if num_heads not in MODELS:
        baseline, optimized, config = make_models(
            BATCH, SEED, False, num_heads=num_heads
        )
        sdpa = SDPATransformer(copy.deepcopy(baseline)).eval()
        value, valid_mask = generate_random_case(
            config=config,
            device=torch.device("cuda"),
            dtype=torch.float16,
            seed=SEED + 100_000 + num_heads,
            padding_ratio=0.0,
            input_scale=1.0,
        )
        MODELS[num_heads] = (
            baseline,
            sdpa,
            optimized,
            config,
            value,
            valid_mask,
        )
    return MODELS[num_heads]


def measure(num_heads: int, provider: str) -> tuple[float, float, float]:
    key = (num_heads, provider)
    if key not in TIMES:
        baseline, sdpa, optimized, _, value, valid_mask = models_for_heads(num_heads)
        models = {
            "megakernel": optimized,
            "sdpa": sdpa,
            "torch": baseline,
        }
        model = models[provider]

        def invoke() -> None:
            with torch.inference_mode():
                model(value, valid_mask)

        quantiles = [0.5, 0.2, 0.8]
        median, low, high = triton.testing.do_bench(
            invoke,
            warmup=WARMUP,
            rep=REPETITIONS,
            quantiles=quantiles,
        )
        TIMES[key] = float(median), float(low), float(high)
    return TIMES[key]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_heads"],
        x_vals=[1, 2, 4],
        x_log=False,
        line_arg="provider",
        line_vals=["megakernel"],
        line_names=["one-CTA megakernel"],
        styles=[("blue", "-")],
        ylabel="effective % of memory roofline",
        plot_name="step-3-h200-roofline",
        args={},
    )
)
def roofline_benchmark(num_heads: int, provider: str):
    median, low, high = measure(num_heads, provider)
    # Performance is inverse in latency, so the high-latency quantile is the
    # low performance error bar and vice versa.
    return (
        ROOFLINE.percent(median),
        ROOFLINE.percent(high),
        ROOFLINE.percent(low),
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_heads"],
        x_vals=[1, 2, 4],
        x_log=False,
        line_arg="provider",
        line_vals=["megakernel", "sdpa", "torch"],
        line_names=["one-CTA megakernel", "Torch SDPA", "explicit Torch"],
        styles=[("blue", "-"), ("orange", "-"), ("green", "-")],
        ylabel="latency (ms)",
        plot_name="step-3-h200-latency",
        args={},
    )
)
def latency_benchmark(num_heads: int, provider: str):
    return measure(num_heads, provider)


def run_text_report(mark, heads: list[int]) -> None:
    """Run a ``triton.testing.Benchmark`` without plotting dependencies."""
    benchmark = mark.benchmarks
    print(f"\n{benchmark.plot_name} ({benchmark.ylabel}):")
    print("num_heads  " + "  ".join(f"{name:>20s}" for name in benchmark.line_names))
    for x_value in benchmark.x_vals:
        if x_value not in heads:
            continue
        values = []
        for line_value in benchmark.line_vals:
            median, _, _ = mark.fn(
                **{benchmark.x_names[0]: x_value},
                **{benchmark.line_arg: line_value},
                **benchmark.args,
            )
            values.append(median)
        print(
            f"{x_value:9d}  "
            + "  ".join(f"{value:20.4f}" for value in values)
        )


def run_accuracy(
    heads: list[int],
    trials: int,
    padding_ratios: list[float],
    scales: list[float],
) -> bool:
    print("\n=== Strict numerical gate ===")
    passed = True
    with torch.inference_mode():
        for num_heads in heads:
            baseline, sdpa, optimized, config, _, _ = models_for_heads(num_heads)
            for padding_ratio in padding_ratios:
                for scale in scales:
                    failed = {"megakernel": 0, "sdpa": 0}
                    max_abs = {"megakernel": 0.0, "sdpa": 0.0}
                    repeat_failed = 0
                    for trial in range(trials):
                        value, valid_mask = generate_random_case(
                            config=config,
                            device=torch.device("cuda"),
                            dtype=torch.float16,
                            seed=(
                                SEED
                                + trial
                                + round(100 * padding_ratio)
                                + round(scale * 1000)
                            ),
                            padding_ratio=padding_ratio,
                            input_scale=scale,
                        )
                        reference = baseline(value, valid_mask)
                        candidates = {
                            "megakernel": optimized(value, valid_mask),
                            "sdpa": sdpa(value, valid_mask),
                        }
                        repeated = optimized(value, valid_mask)
                        repeat_failed += int(
                            (candidates["megakernel"] != repeated).sum().item()
                        )
                        for provider, candidate in candidates.items():
                            result = compare_outputs(
                                reference,
                                candidate,
                                rtol=0.01,
                                atol=0.001,
                            )
                            failed[provider] += result.failed_elements
                            max_abs[provider] = max(
                                max_abs[provider], result.max_abs_error
                            )
                            if provider == "megakernel":
                                passed &= result.passed
                    passed &= repeat_failed == 0
                    for provider in ("megakernel", "sdpa"):
                        provider_passed = failed[provider] == 0 and (
                            provider != "megakernel" or repeat_failed == 0
                        )
                        status = "PASS" if provider_passed else "FAIL"
                        print(
                            f"case {CASE_FOR_HEADS[num_heads]:2d} H={num_heads} "
                            f"padding={padding_ratio:.2f} scale={scale:g} "
                            f"{provider}: {status} failed={failed[provider]}/"
                            f"{trials * BATCH * SEQUENCE * MODEL} "
                            f"max_abs={max_abs[provider]:.7g}"
                            + (
                                f" repeat_diff={repeat_failed}"
                                if provider == "megakernel"
                                else ""
                            )
                        )
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--heads", nargs="+", type=int, choices=(1, 2, 4), default=(1, 2, 4)
    )
    parser.add_argument("--accuracy-trials", type=int, default=3)
    parser.add_argument(
        "--padding-ratios", nargs="+", type=float, default=(0.0,)
    )
    parser.add_argument("--scales", nargs="+", type=float, default=(1.0,))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--peak-tflops", type=float)
    parser.add_argument("--peak-memory-tbps", type=float)
    return parser.parse_args()


def main() -> int:
    global ROOFLINE, WARMUP, REPETITIONS, SEED
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("an NVIDIA GPU is required")
    properties = torch.cuda.get_device_properties(0)
    if properties.major != 9 or properties.minor != 0:
        raise RuntimeError(f"Hopper sm_90 is required, got {properties.name}")
    if "H100" not in properties.name and "H200" not in properties.name:
        raise RuntimeError(f"an H100 or H200 is required, got {properties.name}")

    if "H100" in properties.name and "NVL" in properties.name:
        default_peak = H100_NVL_DENSE_FP16_TFLOPS
        default_memory = H100_NVL_HBM_TBPS
    else:
        default_peak = (
            H200_NVL_DENSE_FP16_TFLOPS
            if "NVL" in properties.name
            else H200_SXM_DENSE_FP16_TFLOPS
        )
        default_memory = H200_HBM_TBPS
    ROOFLINE = Roofline(
        peak_compute_tflops=args.peak_tflops or default_peak,
        peak_memory_tbps=args.peak_memory_tbps or default_memory,
    )
    WARMUP = 25 if args.quick else 100
    REPETITIONS = 75 if args.quick else 500
    SEED = args.seed
    selected_heads = list(dict.fromkeys(args.heads))

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(
        f"torch={torch.__version__} triton={triton.__version__} "
        f"gpu={properties.name} sms={properties.multi_processor_count}"
    )
    print(
        f"roofline: useful_flops={useful_flops() / 1e9:.6f} GFLOP "
        f"activation_traffic={activation_traffic_bytes() / 1e6:.6f} MB "
        f"weight_footprint={packed_weight_bytes() / 1e6:.6f} MB "
        f"roofline_bytes={roofline_bytes() / 1e6:.6f} MB "
        f"intensity={ROOFLINE.arithmetic_intensity:.2f} FLOP/B "
        f"bound={ROOFLINE.bound_tflops:.1f} TFLOP/s "
        f"binding={ROOFLINE.binding_resource}"
    )
    for num_heads in selected_heads:
        tuning = resolved_megakernel_tuning(BATCH, num_heads)
        print(
            f"case {CASE_FOR_HEADS[num_heads]:2d} tuning: "
            + " ".join(f"{name}={value}" for name, value in tuning.items())
        )

    passed = run_accuracy(
        selected_heads,
        args.accuracy_trials,
        args.padding_ratios,
        args.scales,
    )
    if not passed:
        print("roofline benchmark skipped because the strict gate failed")
        return 2

    # The environment intentionally has no matplotlib.  Drive the Benchmark
    # descriptor directly and print its table without pulling a plot stack
    # into the locked performance environment.
    run_text_report(roofline_benchmark, selected_heads)
    run_text_report(latency_benchmark, selected_heads)

    print("\n=== End-to-end summary ===")
    for num_heads in selected_heads:
        optimized_ms = measure(num_heads, "megakernel")[0]
        sdpa_ms = measure(num_heads, "sdpa")[0]
        torch_ms = measure(num_heads, "torch")[0]
        print(
            f"case {CASE_FOR_HEADS[num_heads]:2d} H={num_heads}: "
            f"megakernel={optimized_ms:.6f} ms "
            f"sdpa={sdpa_ms:.6f} ms torch={torch_ms:.6f} ms "
            f"speedup_vs_sdpa={sdpa_ms / optimized_ms:.3f}x "
            f"speedup_vs_torch={torch_ms / optimized_ms:.3f}x "
            f"bandwidth={ROOFLINE.effective_tbps(optimized_ms):.3f} TB/s "
            f"memory_roofline={ROOFLINE.percent(optimized_ms):.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
