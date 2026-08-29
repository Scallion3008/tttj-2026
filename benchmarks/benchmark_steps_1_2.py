#!/usr/bin/env python3
"""Run initial_notes.md benchmarking steps 1 and 2 on a Hopper GPU."""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

from benchmarks.torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    benchmark_models,
    compare_outputs,
    generate_random_case,
    run_accuracy_tests,
)
from optimized_transformer import make_optimized_transformer


STRICT_ATOL = 0.001
STRICT_RTOL = 0.01
INPUT_SCALES = (1.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 0.1, 1.0, 10.0, 100.0, 1000.0)


@dataclass(frozen=True)
class StageResult:
    passed: bool
    failed: int
    total: int
    max_abs: float
    max_gate_ratio: float


def compare_stage(reference: torch.Tensor, candidate: torch.Tensor) -> StageResult:
    reference_fp32 = reference.float()
    candidate_fp32 = candidate.float()
    matching_positive_inf = torch.isposinf(reference_fp32) & torch.isposinf(candidate_fp32)
    matching_negative_inf = torch.isneginf(reference_fp32) & torch.isneginf(candidate_fp32)
    matching_inf = matching_positive_inf | matching_negative_inf
    finite = torch.isfinite(reference_fp32) & torch.isfinite(candidate_fp32)
    absolute = torch.where(
        matching_inf,
        torch.zeros_like(reference_fp32),
        (candidate_fp32 - reference_fp32).abs(),
    )
    allowed = torch.maximum(
        torch.full_like(reference_fp32, STRICT_ATOL),
        STRICT_RTOL * reference_fp32.abs(),
    )
    passed = matching_inf | (finite & (absolute <= allowed))
    failed = int((~passed).sum().item())
    finite_ratio = torch.where(
        matching_inf,
        torch.zeros_like(absolute),
        absolute / allowed,
    )
    finite_ratio = torch.nan_to_num(finite_ratio, nan=math.inf, posinf=math.inf)
    return StageResult(
        passed=failed == 0,
        failed=failed,
        total=reference.numel(),
        max_abs=float(torch.nan_to_num(absolute, nan=math.inf).max().item()),
        max_gate_ratio=float(finite_ratio.max().item()),
    )


def capture_reference_stages(
    model: BaselineTransformer,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    layer = model.layers[0]
    norm = layer.norm1(x)
    q_linear = layer.attention.q_proj(norm)
    k_linear = layer.attention.k_proj(norm)
    v_linear = layer.attention.v_proj(norm)
    q = layer.attention._split_heads(q_linear)
    k = layer.attention._split_heads(k_linear)
    v = layer.attention._split_heads(v_linear)
    scores = torch.matmul(q, k.transpose(-2, -1)) * layer.attention.scale
    causal_mask = torch.ones(
        (x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool
    ).triu(diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    scores = scores.masked_fill(~valid_mask[:, None, None, :], float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(x.dtype)
    context = torch.matmul(probs, v)
    context = (
        context.transpose(1, 2)
        .contiguous()
        .view(x.shape[0], x.shape[1], x.shape[2])
    )
    attention_output = layer.attention.out_proj(context)
    attention_output = attention_output.masked_fill(
        ~valid_mask[..., None], 0
    )
    attention_residual = x + attention_output
    ffn = layer.ffn_out(
        F.gelu(layer.ffn_in(layer.norm2(attention_residual)), approximate="none")
    )
    return {
        "layernorm": norm[0],
        "qkv.q": q_linear[0],
        "qkv.k": k_linear[0],
        "qkv.v": v_linear[0],
        "scores": scores[0],
        "softmax": probs[0],
        "pv": context[0],
        "output_projection": attention_output[0],
        "ffn": ffn[0],
    }


def make_models(
    batch_size: int,
    seed: int,
    verbose_build: bool,
    num_heads: int = 4,
) -> Tuple[BaselineTransformer, SequenceResidentTransformer, TransformerConfig]:
    config = TransformerConfig(
        batch_size=batch_size,
        seq_len=128,
        d_model=128,
        num_heads=num_heads,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    torch.manual_seed(seed)
    baseline = BaselineTransformer(config).cuda().half().eval()
    parameter_model = copy.deepcopy(baseline)
    optimized = make_optimized_transformer(
        parameter_model,
        verbose_build=verbose_build,
    ).cuda().eval()
    return baseline, optimized, config


def run_checkpoints(
    scales: Iterable[float],
    seed: int,
    verbose_build: bool,
) -> bool:
    print("\n=== Step 1: standalone FP16 correctness checkpoints ===")
    baseline, optimized, config = make_models(1, seed, verbose_build)
    all_passed = True
    for padding_ratio in (0.0, 0.25):
        for scale in scales:
            x, valid_mask = generate_random_case(
                config=config,
                device=torch.device("cuda"),
                dtype=torch.float16,
                seed=seed + round(scale * 10000) + int(padding_ratio * 100),
                padding_ratio=padding_ratio,
                input_scale=scale,
            )
            with torch.inference_mode():
                reference = capture_reference_stages(baseline, x, valid_mask)
                _, candidate = optimized.run(
                    x, valid_mask, capture_debug=True
                )
            assert candidate is not None
            case_passed = True
            print(f"scale={scale:g}, padding_ratio={padding_ratio:g}")
            for stage_name, reference_value in reference.items():
                result = compare_stage(reference_value, candidate[stage_name])
                case_passed &= result.passed
                marker = "PASS" if result.passed else "FAIL"
                print(
                    f"  {stage_name:17s} {marker} "
                    f"failed={result.failed}/{result.total} "
                    f"max_abs={result.max_abs:.6g} "
                    f"max_gate_ratio={result.max_gate_ratio:.4f}"
                )
            all_passed &= case_passed
    print(f"step 1 summary: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


def run_case(
    batch_size: int,
    accuracy_trials: int,
    warmup: int,
    repeats: int,
    rounds: int,
    seed: int,
) -> bool:
    case_number = 5 if batch_size == 128 else 6
    print(
        f"\n=== Step 2, case {case_number}: B={batch_size}, S=D=F=128, H=L=4 ==="
    )
    baseline, optimized, config = make_models(batch_size, seed, False)
    accuracy_passed = run_accuracy_tests(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=torch.device("cuda"),
        dtype=torch.float16,
        trials=accuracy_trials,
        seed=seed,
        padding_ratio=0.0,
        input_scale=1.0,
        rtol=STRICT_RTOL,
        atol=STRICT_ATOL,
    )
    benchmark_models(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=torch.device("cuda"),
        dtype=torch.float16,
        seed=seed,
        padding_ratio=0.0,
        input_scale=1.0,
        warmup=warmup,
        repeats=repeats,
        rounds=rounds,
    )
    return accuracy_passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("all", "checkpoints", "benchmarks"), default="all"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        choices=(128, 10000),
        default=(128, 10000),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("an NVIDIA GPU is required")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError(
            f"a Hopper GPU (compute capability 9.0) is required, got "
            f"{torch.cuda.get_device_name()} {torch.cuda.get_device_capability()}"
        )
    print(
        f"torch={torch.__version__}, torch_cuda={torch.version.cuda}, "
        f"gpu={torch.cuda.get_device_name()}, "
        f"sms={torch.cuda.get_device_properties(0).multi_processor_count}"
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    passed = True
    if args.mode in ("all", "checkpoints"):
        scales = (1.0e-3, 1.0, 1000.0) if args.quick else INPUT_SCALES
        passed &= run_checkpoints(scales, args.seed, args.verbose_build)
    if args.mode in ("all", "benchmarks"):
        if args.quick:
            settings = {
                128: (1, 2, 4, 1),
                10000: (1, 1, 2, 1),
            }
        else:
            settings = {
                128: (3, 10, 30, 3),
                10000: (1, 3, 10, 2),
            }
        for batch_size in args.batch_sizes:
            trials, warmup, repeats, rounds = settings[batch_size]
            passed &= run_case(
                batch_size,
                trials,
                warmup,
                repeats,
                rounds,
                args.seed,
            )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
