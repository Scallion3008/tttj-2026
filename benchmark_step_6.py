#!/usr/bin/env python3
"""Correctness and performance exploration for step 6 / benchmark case 8."""

from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel

from layerwise_hybrid import GraphedHybridTransformer, LayerwiseHybridTransformer
from case8_attention import attention_tuning, packed_causal_attention
from case8_fusions import fusion_tuning, residual_layer_norm
from torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    compare_outputs,
    generate_random_case,
)


BATCH = 64
SEQUENCE = 128
MODEL = 1024
HEADS = 4
LAYERS = 4
H200_DENSE_FP16_TFLOPS = 835.5
H200_HBM_TBPS = 4.8
SEED = 1234


def useful_flops() -> int:
    return LAYERS * (
        8 * BATCH * SEQUENCE * MODEL * MODEL
        + 4 * BATCH * SEQUENCE * MODEL * MODEL
        + 2 * BATCH * SEQUENCE * (SEQUENCE + 1) * MODEL
    )


def logical_bytes() -> int:
    tensor = BATCH * SEQUENCE * MODEL * 2
    parameters = LAYERS * (6 * MODEL * MODEL + 10 * MODEL) + 2 * MODEL
    return (22 * LAYERS + 2) * tensor + 2 * parameters


@dataclass(frozen=True)
class Roofline:
    @property
    def intensity(self) -> float:
        return useful_flops() / logical_bytes()

    @property
    def memory_line_tflops(self) -> float:
        return self.intensity * H200_HBM_TBPS

    @property
    def binding(self) -> str:
        return "compute" if H200_DENSE_FP16_TFLOPS < self.memory_line_tflops else "memory"

    @property
    def bound_tflops(self) -> float:
        return min(H200_DENSE_FP16_TFLOPS, self.memory_line_tflops)

    def achieved_tflops(self, milliseconds: float) -> float:
        return useful_flops() / (milliseconds * 1.0e9)

    def percent(self, milliseconds: float) -> float:
        return 100.0 * self.achieved_tflops(milliseconds) / self.bound_tflops


class SDPATransformer(torch.nn.Module):
    def __init__(self, model: BaselineTransformer) -> None:
        super().__init__()
        self.model = model
        self._last_valid: torch.Tensor | None = None
        self._last_valid_version: int | None = None
        self._last_all_valid = False

    def _mask_is_all_valid(self, valid: torch.Tensor) -> bool:
        try:
            version: int | None = valid._version
        except RuntimeError:
            version = None
        if valid is not self._last_valid or version != self._last_valid_version:
            self._last_valid = valid
            self._last_valid_version = version
            self._last_all_valid = bool(valid.all().item())
        return self._last_all_valid

    def forward(self, value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        all_valid = self._mask_is_all_valid(valid)
        attention_mask = None
        if not all_valid:
            causal = torch.ones(SEQUENCE, SEQUENCE, device=value.device, dtype=torch.bool).tril()
            attention_mask = causal[None, None] & valid[:, None, None, :]
        x = value
        for layer in self.model.layers:
            normalized = layer.norm1(x)
            attention = layer.attention
            q = attention._split_heads(attention.q_proj(normalized))
            k = attention._split_heads(attention.k_proj(normalized))
            v = attention._split_heads(attention.v_proj(normalized))
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                context = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attention_mask, is_causal=all_valid
                )
            context = context.transpose(1, 2).contiguous().view(BATCH, SEQUENCE, MODEL)
            branch = attention.out_proj(context)
            branch = branch.masked_fill(~valid[..., None], 0)
            x = x + branch
            x = x + layer.ffn_out(
                F.gelu(layer.ffn_in(layer.norm2(x)), approximate="none")
            )
            x = x.masked_fill(~valid[..., None], 0)
        return self.model.final_norm(x).masked_fill(~valid[..., None], 0)


def make_models():
    config = TransformerConfig(BATCH, SEQUENCE, MODEL, HEADS, MODEL, LAYERS, True)
    torch.manual_seed(SEED)
    baseline = BaselineTransformer(config).cuda().half().eval()
    sdpa = SDPATransformer(copy.deepcopy(baseline)).cuda().eval()
    variants = {
        "adaptive-candidate": LayerwiseHybridTransformer(
            copy.deepcopy(baseline),
            pack_qkv=True,
            attention_backend="custom",
            fused_norm_mask=0xFF,
            flash_attention_mask=0x8,
            adaptive_optimizations=True,
        ).cuda().eval(),
        "optimized-candidate": LayerwiseHybridTransformer(
            copy.deepcopy(baseline),
            pack_qkv=True,
            attention_backend="custom",
            fused_norm_mask=int(os.environ.get("TTTJ_STEP6_FUSED_NORM_MASK", "0"), 0),
            flash_attention_mask=int(
                os.environ.get("TTTJ_STEP6_FLASH_ATTN_MASK", "0"), 0
            ),
        ).cuda().eval(),
        "mixed-attention": LayerwiseHybridTransformer(
            copy.deepcopy(baseline),
            pack_qkv=True,
            attention_backend="custom",
            flash_attention_mask=int(
                os.environ.get("TTTJ_STEP6_FLASH_ATTN_MASK", "0"), 0
            ),
        ).cuda().eval(),
        "tanh-gelu-custom": LayerwiseHybridTransformer(
            copy.deepcopy(baseline),
            pack_qkv=True,
            attention_backend="custom",
            gelu_approximate="tanh",
        ).cuda().eval(),
        "partial-fused-custom": LayerwiseHybridTransformer(
            copy.deepcopy(baseline),
            pack_qkv=True,
            attention_backend="custom",
            fused_norm_mask=int(os.environ.get("TTTJ_STEP6_FUSED_NORM_MASK", "0"), 0),
        ).cuda().eval(),
        "fused-custom": LayerwiseHybridTransformer(
            copy.deepcopy(baseline),
            pack_qkv=True,
            attention_backend="custom",
            fuse_residual_norm=True,
        ).cuda().eval(),
        "custom-packed": LayerwiseHybridTransformer(
            copy.deepcopy(baseline), pack_qkv=True, attention_backend="custom"
        ).cuda().eval(),
        "explicit-separate": LayerwiseHybridTransformer(
            copy.deepcopy(baseline), pack_qkv=False, attention_backend="explicit"
        ).cuda().eval(),
        "explicit-packed": LayerwiseHybridTransformer(
            copy.deepcopy(baseline), pack_qkv=True, attention_backend="explicit"
        ).cuda().eval(),
        "flash-packed": LayerwiseHybridTransformer(
            copy.deepcopy(baseline), pack_qkv=True, attention_backend="flash"
        ).cuda().eval(),
        "efficient-packed": LayerwiseHybridTransformer(
            copy.deepcopy(baseline), pack_qkv=True, attention_backend="efficient"
        ).cuda().eval(),
        "math-packed": LayerwiseHybridTransformer(
            copy.deepcopy(baseline), pack_qkv=True, attention_backend="math"
        ).cuda().eval(),
    }
    for model in variants.values():
        model.prepare()
    variants["graph-flash"] = GraphedHybridTransformer(
        LayerwiseHybridTransformer(
            copy.deepcopy(baseline), pack_qkv=True, attention_backend="flash"
        ).cuda().eval()
    ).cuda().eval()
    variants["graph-explicit"] = GraphedHybridTransformer(
        LayerwiseHybridTransformer(
            copy.deepcopy(baseline), pack_qkv=True, attention_backend="explicit"
        ).cuda().eval()
    ).cuda().eval()
    variants["graph-custom"] = GraphedHybridTransformer(
        LayerwiseHybridTransformer(
            copy.deepcopy(baseline), pack_qkv=True, attention_backend="custom"
        ).cuda().eval()
    ).cuda().eval()
    variants["graph-fused-custom"] = GraphedHybridTransformer(
        LayerwiseHybridTransformer(
            copy.deepcopy(baseline),
            pack_qkv=True,
            attention_backend="custom",
            fuse_residual_norm=True,
        ).cuda().eval()
    ).cuda().eval()
    variants["graph-partial-fused-custom"] = GraphedHybridTransformer(
        LayerwiseHybridTransformer(
            copy.deepcopy(baseline),
            pack_qkv=True,
            attention_backend="custom",
            fused_norm_mask=int(os.environ.get("TTTJ_STEP6_FUSED_NORM_MASK", "0"), 0),
        ).cuda().eval()
    ).cuda().eval()
    variants["graph-tanh-gelu-custom"] = GraphedHybridTransformer(
        LayerwiseHybridTransformer(
            copy.deepcopy(baseline),
            pack_qkv=True,
            attention_backend="custom",
            gelu_approximate="tanh",
        ).cuda().eval()
    ).cuda().eval()
    variants["graph-mixed-attention"] = GraphedHybridTransformer(
        LayerwiseHybridTransformer(
            copy.deepcopy(baseline),
            pack_qkv=True,
            attention_backend="custom",
            flash_attention_mask=int(
                os.environ.get("TTTJ_STEP6_FLASH_ATTN_MASK", "0"), 0
            ),
        ).cuda().eval()
    ).cuda().eval()
    variants["graph-optimized-candidate"] = GraphedHybridTransformer(
        LayerwiseHybridTransformer(
            copy.deepcopy(baseline),
            pack_qkv=True,
            attention_backend="custom",
            fused_norm_mask=int(os.environ.get("TTTJ_STEP6_FUSED_NORM_MASK", "0"), 0),
            flash_attention_mask=int(
                os.environ.get("TTTJ_STEP6_FLASH_ATTN_MASK", "0"), 0
            ),
        ).cuda().eval()
    ).cuda().eval()
    variants["graph-adaptive-candidate"] = GraphedHybridTransformer(
        LayerwiseHybridTransformer(
            copy.deepcopy(baseline),
            pack_qkv=True,
            attention_backend="custom",
            fused_norm_mask=0xFF,
            flash_attention_mask=0x8,
            adaptive_optimizations=True,
        ).cuda().eval()
    ).cuda().eval()
    return config, baseline, sdpa, variants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--microbench", action="store_true")
    parser.add_argument("--sweep-norm-masks", action="store_true")
    parser.add_argument("--sweep-flash-masks", action="store_true")
    parser.add_argument("--accuracy-matrix", action="store_true")
    parser.add_argument("--accuracy-trials", type=int, default=1)
    parser.add_argument(
        "--matrix-provider",
        default="adaptive-candidate",
        help="variant name used by --accuracy-matrix",
    )
    parser.add_argument(
        "--padding-ratios", nargs="+", type=float, default=(0.0, 0.25, 0.75)
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=(1.0e-4, 1.0e-3, 1.0e-2, 0.1, 1.0, 10.0, 100.0, 1000.0),
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=(
            "torch",
            "sdpa",
            "custom-packed",
            "fused-custom",
            "explicit-separate",
            "explicit-packed",
            "flash-packed",
            "efficient-packed",
            "math-packed",
            "graph-flash",
            "graph-explicit",
            "graph-custom",
            "graph-fused-custom",
        ),
    )
    args = parser.parse_args()
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0) or "H200" not in properties.name:
        raise RuntimeError(f"an H200 sm_90 GPU is required, got {properties.name}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = bool(
        int(os.environ.get("TTTJ_STEP6_FP16_ACCUMULATION", "0"))
    )
    config, baseline, sdpa, variants = make_models()
    value, valid = generate_random_case(config, torch.device("cuda"), torch.float16, SEED + 100_000, 0.0, 1.0)
    models = {"torch": baseline, "sdpa": sdpa, **variants}
    variants["graph-flash"].prepare(value, valid)
    variants["graph-explicit"].prepare(value, valid)
    variants["graph-custom"].prepare(value, valid)
    variants["graph-fused-custom"].prepare(value, valid)
    variants["graph-partial-fused-custom"].prepare(value, valid)
    variants["graph-tanh-gelu-custom"].prepare(value, valid)
    variants["graph-mixed-attention"].prepare(value, valid)
    variants["graph-optimized-candidate"].prepare(value, valid)
    variants["graph-adaptive-candidate"].prepare(value, valid)
    print(f"torch={torch.__version__} triton={triton.__version__} gpu={properties.name} sms={properties.multi_processor_count} fp16_accumulation={int(torch.backends.cuda.matmul.allow_fp16_accumulation)} custom_attention={attention_tuning()} fusions={fusion_tuning()}")
    roofline = Roofline()
    print(f"useful={useful_flops()/1e9:.6f} GFLOP logical_bytes={logical_bytes()/1e9:.6f} GB intensity={roofline.intensity:.3f} FLOP/B binding={roofline.binding} bound={roofline.bound_tflops:.3f} TFLOP/s")
    if not args.skip_accuracy:
        print("\n=== accuracy ===")
        with torch.inference_mode():
            reference = baseline(value, valid)
            for name in args.providers:
                candidate = models[name](value, valid)
                result = compare_outputs(reference, candidate, 0.01, 0.001)
                print(f"{name:9s} {'PASS' if result.passed else 'FAIL'} failed={result.failed_elements}/{result.total_elements} max_abs={result.max_abs_error:.7g} mean_abs={result.mean_abs_error:.7g}")
    if args.sweep_norm_masks:
        print("\n=== fused residual/LayerNorm mask sweep ===")
        model = variants["partial-fused-custom"]
        with torch.inference_mode():
            reference = baseline(value, valid)
            for mask in range(1, 256):
                model.fused_norm_mask = mask
                result = compare_outputs(
                    reference, model(value, valid), 0.01, 0.001
                )
                if result.passed or mask.bit_count() <= 1:
                    print(
                        f"mask=0x{mask:02x} fused={mask.bit_count()} "
                        f"{'PASS' if result.passed else 'FAIL'} "
                        f"failed={result.failed_elements} "
                        f"max_abs={result.max_abs_error:.7g}"
                    )
    if args.sweep_flash_masks:
        print("\n=== Flash-attention layer mask sweep ===")
        model = variants["mixed-attention"]
        with torch.inference_mode():
            reference = baseline(value, valid)
            for mask in range(1, 16):
                model.flash_attention_mask = mask
                result = compare_outputs(
                    reference, model(value, valid), 0.01, 0.001
                )
                print(
                    f"mask=0x{mask:x} flash_layers={mask.bit_count()} "
                    f"{'PASS' if result.passed else 'FAIL'} "
                    f"failed={result.failed_elements} "
                    f"max_abs={result.max_abs_error:.7g}"
                )
    if args.accuracy_matrix:
        print(f"\n=== strict accuracy matrix ({args.matrix_provider}) ===")
        candidate_model = variants[args.matrix_provider]
        matrix_passed = True
        with torch.inference_mode():
            for padding_ratio in args.padding_ratios:
                for scale in args.scales:
                    failed = 0
                    repeat_diff = 0
                    max_abs = 0.0
                    for trial in range(args.accuracy_trials):
                        matrix_value, matrix_valid = generate_random_case(
                            config,
                            torch.device("cuda"),
                            torch.float16,
                            SEED
                            + trial
                            + round(padding_ratio * 1000)
                            + round(scale * 10000),
                            padding_ratio,
                            scale,
                        )
                        reference = baseline(matrix_value, matrix_valid)
                        candidate = candidate_model(matrix_value, matrix_valid)
                        repeated = candidate_model(matrix_value, matrix_valid)
                        result = compare_outputs(
                            reference, candidate, 0.01, 0.001
                        )
                        failed += result.failed_elements
                        max_abs = max(max_abs, result.max_abs_error)
                        repeat_diff += int((candidate != repeated).sum().item())
                    passed = failed == 0 and repeat_diff == 0
                    matrix_passed &= passed
                    print(
                        f"padding={padding_ratio:.2f} scale={scale:g} "
                        f"{'PASS' if passed else 'FAIL'} failed={failed} "
                        f"max_abs={max_abs:.7g} repeat_diff={repeat_diff}"
                    )
        if not matrix_passed:
            print("accuracy matrix failed")
            return 2
    if args.microbench:
        print("\n=== component microbenchmarks ===")
        hybrid = variants["custom-packed"]
        layer = hybrid.parameter_model.layers[0]
        normalized = layer.norm1(value)
        projected = F.linear(
            normalized, hybrid.qkv_weights[0], hybrid.qkv_biases[0]
        )
        q, k, v = projected.split(MODEL, dim=-1)
        q = q.view(BATCH, SEQUENCE, HEADS, MODEL // HEADS).transpose(1, 2)
        k = k.view(BATCH, SEQUENCE, HEADS, MODEL // HEADS).transpose(1, 2)
        v = v.view(BATCH, SEQUENCE, HEADS, MODEL // HEADS).transpose(1, 2)
        hidden = layer.ffn_in(normalized)
        branch = layer.attention.out_proj(normalized)
        components = {
            "linear-1024": (
                lambda: F.linear(normalized, layer.ffn_in.weight, layer.ffn_in.bias),
                2 * BATCH * SEQUENCE * MODEL * MODEL,
                "compute",
            ),
            "packed-qkv": (
                lambda: F.linear(normalized, hybrid.qkv_weights[0], hybrid.qkv_biases[0]),
                6 * BATCH * SEQUENCE * MODEL * MODEL,
                "compute",
            ),
            "custom-attn": (
                lambda: packed_causal_attention(projected, valid, all_valid=True),
                (
                    BATCH * SEQUENCE * MODEL * 2
                    + BATCH * HEADS * (64 + 128) * (MODEL // HEADS) * 2 * 2
                    + BATCH * SEQUENCE * MODEL * 2
                ),
                "memory",
            ),
            "context-copy": (
                lambda: q.transpose(1, 2).contiguous(),
                2 * BATCH * SEQUENCE * MODEL * 2,
                "memory",
            ),
            "residual-norm": (
                lambda: residual_layer_norm(
                    value,
                    branch,
                    layer.norm2.weight,
                    layer.norm2.bias,
                    valid,
                    all_valid=True,
                ),
                4 * BATCH * SEQUENCE * MODEL * 2,
                "memory",
            ),
            "layer-norm": (
                lambda: layer.norm1(value),
                2 * BATCH * SEQUENCE * MODEL * 2,
                "memory",
            ),
            "residual-add": (
                lambda: value + branch,
                3 * BATCH * SEQUENCE * MODEL * 2,
                "memory",
            ),
            "exact-gelu": (
                lambda: F.gelu(hidden, approximate="none"),
                2 * BATCH * SEQUENCE * MODEL * 2,
                "memory",
            ),
        }
        with torch.inference_mode():
            for name, (operation, work, resource) in components.items():
                median, low, high = triton.testing.do_bench(
                    operation,
                    warmup=20 if args.quick else 100,
                    rep=50 if args.quick else 500,
                    quantiles=[0.5, 0.2, 0.8],
                )
                ms = float(median)
                if resource == "compute":
                    achieved = work / (ms * 1.0e9)
                    percent = 100.0 * achieved / H200_DENSE_FP16_TFLOPS
                    metric = f"{achieved:.3f} TFLOP/s {percent:.2f}% peak"
                else:
                    achieved = work / (ms * 1.0e9)
                    percent = 100.0 * achieved / H200_HBM_TBPS
                    metric = f"{achieved:.3f} TB/s {percent:.2f}% peak"
                print(
                    f"{name:14s} median={ms:.6f} ms p20={float(low):.6f} "
                    f"p80={float(high):.6f} {metric}"
                )
    print("\n=== latency ===")
    warmup, rep = (10, 30) if args.quick else (100, 500)
    with torch.inference_mode():
        for name in args.providers:
            model = models[name]
            median, low, high = triton.testing.do_bench(
                lambda: model(value, valid), warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8]
            )
            ms = float(median)
            print(f"{name:9s} median={ms:.6f} ms p20={float(low):.6f} p80={float(high):.6f} achieved={roofline.achieved_tflops(ms):.3f} TFLOP/s roofline={roofline.percent(ms):.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
