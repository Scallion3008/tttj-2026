#!/usr/bin/env python3
"""Correctness, latency, and roofline benchmark for HD8 cases 7 and 11."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel

from fused_megakernel import (
    HD8_PV_MODE,
    HD8_QK_MODE,
    resolved_megakernel_tuning,
)
from sequence_resident import SequenceResidentTransformer
from torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    compare_outputs,
    generate_random_case,
)


SEQUENCE = 128
BATCH = 64
LAYERS = 4
DTYPE_BYTES = 2
H100_NVL_DENSE_FP16_TFLOPS = 835.5
H100_NVL_HBM_TBPS = 3.9
H200_NVL_DENSE_FP16_TFLOPS = 835.5
H200_NVL_HBM_TBPS = 4.8
CASES = {7: (32, 4), 11: (128, 16)}
MODELS: dict[int, tuple[torch.nn.Module, ...]] = {}
TIMES: dict[tuple[int, str], tuple[float, float, float]] = {}
WARMUP = 100
REPETITIONS = 500
SEED = 1234


def useful_flops(model: int, heads: int) -> int:
    del heads
    return LAYERS * (
        8 * BATCH * SEQUENCE * model * model
        + 4 * BATCH * SEQUENCE * model * model
        + 2 * BATCH * SEQUENCE * (SEQUENCE + 1) * model
    )


def packed_weight_bytes(model: int) -> int:
    layer_parameters = 6 * model * model + 10 * model
    return DTYPE_BYTES * (LAYERS * layer_parameters + 2 * model)


def roofline_bytes(model: int) -> int:
    tensor_bytes = BATCH * SEQUENCE * model * DTYPE_BYTES
    return (22 * LAYERS + 2) * tensor_bytes + packed_weight_bytes(model)


@dataclass(frozen=True)
class Roofline:
    model: int
    dense_fp16_tflops: float
    hbm_tbps: float

    @property
    def intensity(self) -> float:
        return useful_flops(self.model, 1) / roofline_bytes(self.model)

    @property
    def memory_line_tflops(self) -> float:
        return self.intensity * self.hbm_tbps

    @property
    def bound_tflops(self) -> float:
        return min(self.dense_fp16_tflops, self.memory_line_tflops)

    @property
    def binding_resource(self) -> str:
        return (
            "compute"
            if self.dense_fp16_tflops <= self.memory_line_tflops
            else "memory"
        )

    def percent(self, milliseconds: float) -> float:
        achieved = useful_flops(self.model, 1) / (milliseconds * 1e9)
        return 100.0 * achieved / self.bound_tflops


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
            causal = torch.ones(
                sequence, sequence, device=x.device, dtype=torch.bool
            ).tril()
            causal_mask = causal[None, None, :, :] & valid_token_mask[
                :, None, None, :
            ]
        for layer in self.model.layers:
            normalized = layer.norm1(x)
            attention = layer.attention
            q = attention._split_heads(attention.q_proj(normalized))
            k = attention._split_heads(attention.k_proj(normalized))
            v = attention._split_heads(attention.v_proj(normalized))
            # The H200 node image does not ship cuDNN's runtime-compiled
            # engines. These are also the fastest available backends measured
            # for the two fixed shapes (efficient for D32, Flash for D128).
            backend = (
                SDPBackend.EFFICIENT_ATTENTION
                if model == 32
                else SDPBackend.FLASH_ATTENTION
            )
            with sdpa_kernel(backend):
                context = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=causal_mask,
                    is_causal=all_valid,
                )
            context = context.transpose(1, 2).contiguous().view(
                batch, sequence, model
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


def models_for_case(case_number: int):
    if case_number not in MODELS:
        model, heads = CASES[case_number]
        config = TransformerConfig(
            batch_size=BATCH,
            seq_len=SEQUENCE,
            d_model=model,
            num_heads=heads,
            ffn_dim=model,
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
        selected = {"megakernel": optimized, "sdpa": sdpa, "torch": baseline}

        def invoke() -> None:
            with torch.inference_mode():
                selected[provider](value, valid)

        median, low, high = triton.testing.do_bench(
            invoke,
            warmup=WARMUP,
            rep=REPETITIONS,
            quantiles=[0.5, 0.2, 0.8],
        )
        TIMES[key] = float(median), float(low), float(high)
    return TIMES[key]


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
                    repeat_failed = 0
                    max_abs = 0.0
                    for trial in range(trials):
                        value, valid = generate_random_case(
                            config,
                            torch.device("cuda"),
                            torch.float16,
                            SEED
                            + trial
                            + round(100 * padding_ratio)
                            + round(scale * 1000),
                            padding_ratio,
                            scale,
                        )
                        reference = baseline(value, valid)
                        candidate = optimized(value, valid)
                        repeated = optimized(value, valid)
                        result = compare_outputs(
                            reference, candidate, rtol=0.01, atol=0.001
                        )
                        failed += result.failed_elements
                        max_abs = max(max_abs, result.max_abs_error)
                        repeat_failed += int((candidate != repeated).sum().item())
                    case_passed = failed == 0 and repeat_failed == 0
                    passed &= case_passed
                    print(
                        f"case {case_number:2d} padding={padding_ratio:.2f} "
                        f"scale={scale:g}: {'PASS' if case_passed else 'FAIL'} "
                        f"failed={failed}/{trials * config.batch_size * config.seq_len * config.d_model} "
                        f"max_abs={max_abs:.7g} repeat_diff={repeat_failed}"
                    )
    return passed


def main() -> int:
    global WARMUP, REPETITIONS, SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", type=int, choices=CASES, default=CASES)
    parser.add_argument("--accuracy-trials", type=int, default=2)
    parser.add_argument("--padding-ratios", nargs="+", type=float, default=(0.0,))
    parser.add_argument("--scales", nargs="+", type=float, default=(1.0,))
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=("megakernel", "sdpa", "torch"),
        default=("megakernel", "sdpa", "torch"),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--allow-h200", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("an NVIDIA GPU is required")
    properties = torch.cuda.get_device_properties(0)
    allowed_name = "H100" in properties.name or (
        args.allow_h200 and "H200" in properties.name
    )
    if (properties.major, properties.minor) != (9, 0) or not allowed_name:
        raise RuntimeError(f"H100 sm_90 is required, got {properties.name}")
    WARMUP = 25 if args.quick else 100
    REPETITIONS = 75 if args.quick else 500
    SEED = args.seed
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if "H200" in properties.name:
        dense_fp16_tflops = H200_NVL_DENSE_FP16_TFLOPS
        hbm_tbps = H200_NVL_HBM_TBPS
    else:
        dense_fp16_tflops = H100_NVL_DENSE_FP16_TFLOPS
        hbm_tbps = H100_NVL_HBM_TBPS
    selected_cases = list(dict.fromkeys(args.cases))
    print(
        f"torch={torch.__version__} triton={triton.__version__} "
        f"gpu={properties.name} sms={properties.multi_processor_count} "
        f"hd8_qk={'simt' if HD8_QK_MODE == 0 else 'padded-tensor-core'} "
        f"hd8_pv={'native-n8' if HD8_PV_MODE == 0 else 'padded-n16'} "
        "schedule=two-role-overlap"
    )
    for case_number in selected_cases:
        model, heads = CASES[case_number]
        roofline = Roofline(model, dense_fp16_tflops, hbm_tbps)
        tuning = resolved_megakernel_tuning(BATCH, heads, model)
        print(
            f"case {case_number} tuning={tuning} "
            f"intensity={roofline.intensity:.2f} FLOP/B "
            f"binding={roofline.binding_resource}"
        )
    if not args.skip_accuracy and not run_accuracy(
        selected_cases, args.accuracy_trials, args.padding_ratios, args.scales
    ):
        print("latency benchmark skipped because the strict gate failed")
        return 2
    print("\n=== End-to-end latency ===")
    for case_number in selected_cases:
        model, _ = CASES[case_number]
        timings = {provider: measure(case_number, provider)[0] for provider in args.providers}
        fields = " ".join(f"{name}={value:.6f} ms" for name, value in timings.items())
        speedup = (
            f" speedup_vs_sdpa={timings['sdpa'] / timings['megakernel']:.3f}x"
            if "sdpa" in timings and "megakernel" in timings
            else ""
        )
        roofline = Roofline(model, dense_fp16_tflops, hbm_tbps)
        percent = (
            f" roofline={roofline.percent(timings['megakernel']):.2f}%"
            if "megakernel" in timings
            else ""
        )
        print(f"case {case_number}: {fields}{speedup}{percent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
