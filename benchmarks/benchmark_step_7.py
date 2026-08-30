#!/usr/bin/env python3
"""Correctness and H200 tuning harness for step 7 / benchmark case 13."""

from __future__ import annotations

import argparse
import copy
import importlib.util
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmarks.torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    compare_outputs,
    generate_random_case,
)
from kernels.case13_hybrid import (
    CASE13_BATCH as BATCH,
    CASE13_HEADS as HEADS,
    CASE13_LAYERS as LAYERS,
    CASE13_MODEL as MODEL,
    CASE13_SEQUENCE as SEQUENCE,
    Case13LayerwiseHybrid,
    Case13OptimizedTransformer,
    GraphedCase13Hybrid,
)
from kernels.case13_fusions import (
    fusion_tuning,
    residual_layer_norm,
    standalone_layer_norm,
)
from kernels.case13_linear import (
    head_major_qkv,
    linear_gelu,
    linear_residual_layer_norm,
    linear_tuning,
)


H200_DENSE_FP16_TFLOPS = 835.5
H200_HBM_TBPS = 4.8
SEED = 1234


def useful_flops() -> int:
    return LAYERS * (
        8 * BATCH * SEQUENCE * MODEL * MODEL
        + 4 * BATCH * SEQUENCE * MODEL * MODEL
        + 2 * BATCH * SEQUENCE * (SEQUENCE + 1) * MODEL
    )


def base_logical_bytes() -> int:
    tensor = BATCH * SEQUENCE * MODEL * 2
    parameters = LAYERS * (6 * MODEL * MODEL + 10 * MODEL) + 2 * MODEL
    return (22 * LAYERS + 2) * tensor + 2 * parameters


def estimated_production_bytes() -> int:
    # The strict path materializes FP16 attention scores. Per layer: QK output,
    # fused scale/mask read+write, softmax read+write, and the probability read
    # by PV. This conservative HBM estimate makes the roofline representative
    # of the selected implementation instead of counting only model tensors.
    score_matrix = BATCH * HEADS * SEQUENCE * SEQUENCE * 2
    return base_logical_bytes() + LAYERS * 6 * score_matrix


@dataclass(frozen=True)
class Roofline:
    @property
    def intensity(self) -> float:
        return useful_flops() / estimated_production_bytes()

    @property
    def memory_line_tflops(self) -> float:
        return self.intensity * H200_HBM_TBPS

    @property
    def bound_tflops(self) -> float:
        return min(H200_DENSE_FP16_TFLOPS, self.memory_line_tflops)

    @property
    def binding(self) -> str:
        return (
            "compute"
            if H200_DENSE_FP16_TFLOPS < self.memory_line_tflops
            else "memory"
        )

    def achieved_tflops(self, milliseconds: float) -> float:
        return useful_flops() / (milliseconds * 1.0e9)

    def percent(self, milliseconds: float) -> float:
        return 100.0 * self.achieved_tflops(milliseconds) / self.bound_tflops


def make_models(provider_names: set[str], attention_mask: int):
    config = TransformerConfig(BATCH, SEQUENCE, MODEL, HEADS, MODEL, LAYERS, True)
    torch.manual_seed(SEED)
    baseline = BaselineTransformer(config).cuda().half().eval()
    variants: dict[str, torch.nn.Module] = {}
    for name in sorted(provider_names):
        if name == "torch":
            continue
        if name == "production":
            production = Case13OptimizedTransformer(
                copy.deepcopy(baseline)
            ).cuda().eval()
            variants[name] = production
            continue
        if name == "graph-production":
            production = Case13OptimizedTransformer(
                copy.deepcopy(baseline)
            ).cuda().eval()
            variants[name] = GraphedCase13Hybrid(production).cuda().eval()
            continue
        graph = name.startswith("graph-")
        base_name = name.removeprefix("graph-")
        family, separator, backend = base_name.partition("-")
        if not separator or family not in (
            "separate",
            "packed",
            "fused",
            "compiled",
            "clinear",
            "hlinear",
            "slinear",
            "ilinear",
            "alinear",
            "linear",
        ):
            raise ValueError(f"unknown case-13 provider {name}")
        if backend not in (
            "auto",
            "flash",
            "cudnn",
            "efficient",
            "fa3",
            "math",
        ):
            raise ValueError(f"unknown case-13 attention backend {backend}")
        if backend == "fa3" and importlib.util.find_spec("flash_attn_3") is None:
            raise RuntimeError("FlashAttention-3 is not installed")
        hybrid = Case13LayerwiseHybrid(
            copy.deepcopy(baseline),
            pack_qkv=family != "separate",
            head_major_qkv_projection=family
            in ("hlinear", "slinear", "ilinear", "alinear"),
            attention_backend=backend,
            fuse_residual_norm=family in ("fused", "compiled"),
            fuse_linear_epilogues=family
            in (
                "linear",
                "clinear",
                "hlinear",
                "slinear",
                "ilinear",
                "alinear",
            ),
            fuse_input_norm=family
            in ("hlinear", "slinear", "ilinear", "alinear"),
            streaming_attention_mask=(0 if family == "slinear" else attention_mask),
            compile_exact_attention=family
            in (
                "compiled",
                "clinear",
                "hlinear",
                "slinear",
                "ilinear",
                "alinear",
            ),
            compiled_softmax_mask=(attention_mask if family == "slinear" else 0),
            exact_score_mode=(
                "index"
                if family == "ilinear"
                else "additive"
                if family == "alinear"
                else "input"
            ),
        ).cuda().eval()
        hybrid.prepare()
        variants[name] = (
            GraphedCase13Hybrid(hybrid).cuda().eval() if graph else hybrid
        )
    return config, baseline, variants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--allow-h100", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--microbench", action="store_true")
    parser.add_argument("--accuracy-matrix", action="store_true")
    parser.add_argument("--sweep-attention-masks", action="store_true")
    parser.add_argument("--sweep-mask-scales", action="store_true")
    parser.add_argument("--error-distribution", action="store_true")
    parser.add_argument("--sweep-provider", default="packed-cudnn")
    parser.add_argument("--attention-mask", type=lambda value: int(value, 0), default=0xF)
    parser.add_argument("--accuracy-trials", type=int, default=1)
    parser.add_argument(
        "--matrix-provider", default="production"
    )
    parser.add_argument(
        "--padding-ratios", nargs="+", type=float, default=(0.0, 0.25, 0.75)
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=(1.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 0.1, 1.0, 10.0, 100.0, 1000.0),
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=(
            "torch",
            "separate-flash",
            "packed-flash",
            "packed-cudnn",
            "fused-cudnn",
            "compiled-cudnn",
            "graph-compiled-cudnn",
            "production",
        ),
    )
    args = parser.parse_args()

    properties = torch.cuda.get_device_properties(0)
    allowed_name = "H200" in properties.name or (
        args.allow_h100 and "H100" in properties.name
    )
    if (properties.major, properties.minor) != (9, 0) or not allowed_name:
        raise RuntimeError(f"an H100/H200 sm_90 GPU is required, got {properties.name}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    required_providers = set(args.providers)
    if args.accuracy_matrix:
        required_providers.add(args.matrix_provider)
    if args.microbench:
        required_providers.add("packed-flash")
        required_providers.add("compiled-cudnn")
    if args.sweep_attention_masks or args.sweep_mask_scales:
        required_providers.add(args.sweep_provider)
    config, baseline, variants = make_models(required_providers, args.attention_mask)
    value, valid = generate_random_case(
        config, torch.device("cuda"), torch.float16, SEED + 100_000, 0.0, 1.0
    )
    models = {"torch": baseline, **variants}
    for name, model in variants.items():
        if name.startswith("graph-"):
            assert isinstance(model, GraphedCase13Hybrid)
            model.prepare(value, valid)

    print(
        f"torch={torch.__version__} triton={triton.__version__} "
        f"gpu={properties.name} sms={properties.multi_processor_count} "
        f"cudnn={torch.backends.cudnn.version()}"
        f" fusions={fusion_tuning()}"
        f" linear={linear_tuning()}"
    )
    roofline = Roofline()
    print(
        f"useful={useful_flops()/1e9:.6f} GFLOP "
        f"estimated_bytes={estimated_production_bytes()/1e9:.6f} GB "
        f"intensity={roofline.intensity:.3f} FLOP/B "
        f"target=H200 binding={roofline.binding} "
        f"bound={roofline.bound_tflops:.3f} TFLOP/s"
    )

    if not args.skip_accuracy:
        print("\n=== accuracy ===")
        with torch.inference_mode():
            reference = baseline(value, valid)
            for name in args.providers:
                candidate = models[name](value, valid)
                result = compare_outputs(reference, candidate, 0.01, 0.001)
                print(
                    f"{name:24s} {'PASS' if result.passed else 'FAIL'} "
                    f"failed={result.failed_elements}/{result.total_elements} "
                    f"max_abs={result.max_abs_error:.7g} "
                    f"mean_abs={result.mean_abs_error:.7g}"
                )

    if args.sweep_attention_masks:
        print("\n=== streaming-attention layer mask sweep ===")
        model = variants[args.sweep_provider]
        assert isinstance(model, Case13LayerwiseHybrid)
        with torch.inference_mode():
            reference = baseline(value, valid)
            for mask in range(16):
                model.streaming_attention_mask = mask
                candidate = model(value, valid)
                result = compare_outputs(reference, candidate, 0.01, 0.001)
                print(
                    f"mask=0x{mask:x} streaming_layers={mask.bit_count()} "
                    f"{'PASS' if result.passed else 'FAIL'} "
                    f"failed={result.failed_elements} "
                    f"max_abs={result.max_abs_error:.7g}"
                )
                if args.error_distribution and not result.passed:
                    error = (reference.float() - candidate.float()).abs()
                    failing = ~(
                        (error <= 0.001)
                        | (error <= 0.01 * reference.float().abs())
                    )
                    per_token = failing.sum(dim=(0, 2))
                    positions = torch.nonzero(per_token).flatten()
                    thresholds = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
                    suffix = " ".join(
                        f">={threshold}:{int(per_token[threshold:].sum())}"
                        for threshold in thresholds
                    )
                    print(
                        f"  failing_token_range={int(positions[0])}:"
                        f"{int(positions[-1])} {suffix}"
                    )
            model.streaming_attention_mask = args.attention_mask

    if args.sweep_mask_scales:
        print(f"\n=== attention-mask scale sweep ({args.sweep_provider}) ===")
        model = variants[args.sweep_provider]
        assert isinstance(model, Case13LayerwiseHybrid)
        with torch.inference_mode():
            for scale in args.scales:
                matrix_value, matrix_valid = generate_random_case(
                    config,
                    torch.device("cuda"),
                    torch.float16,
                    SEED + round(scale * 10000),
                    0.0,
                    scale,
                )
                reference = baseline(matrix_value, matrix_valid)
                passing = []
                failures = []
                for mask in range(16):
                    model.streaming_attention_mask = mask
                    candidate = model(matrix_value, matrix_valid)
                    result = compare_outputs(reference, candidate, 0.01, 0.001)
                    if result.passed:
                        passing.append(f"0x{mask:x}")
                    failures.append(result.failed_elements)
                rms = float(matrix_value.float().square().mean().sqrt())
                print(
                    f"scale={scale:g} rms={rms:.7g} passing={','.join(passing)} "
                    f"failures={failures}"
                )
            model.streaming_attention_mask = args.attention_mask

    if args.accuracy_matrix:
        print(f"\n=== strict accuracy matrix ({args.matrix_provider}) ===")
        candidate_model = models[args.matrix_provider]
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
                        result = compare_outputs(reference, candidate, 0.01, 0.001)
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
        hybrid = variants["packed-flash"]
        assert isinstance(hybrid, Case13LayerwiseHybrid)
        exact_hybrid = variants["compiled-cudnn"]
        assert isinstance(exact_hybrid, Case13LayerwiseHybrid)
        layer = hybrid.parameter_model.layers[0]
        normalized = layer.norm1(value)
        assert hybrid.qkv_weights is not None
        assert hybrid.qkv_biases is not None
        projected = F.linear(normalized, hybrid.qkv_weights[0], hybrid.qkv_biases[0])
        q_linear, k_linear, v_linear = projected.split(MODEL, dim=-1)
        q = hybrid._split_heads(q_linear)
        k = hybrid._split_heads(k_linear)
        v = hybrid._split_heads(v_linear)
        contiguous_q = q.contiguous()
        contiguous_k = k.contiguous()
        contiguous_v = v.contiguous()
        hidden = layer.ffn_in(normalized)
        branch = layer.attention.out_proj(normalized)

        def attention(backend: SDPBackend):
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attention_flops = 2 * BATCH * SEQUENCE * (SEQUENCE + 1) * MODEL
        tensor_bytes = BATCH * SEQUENCE * MODEL * 2
        score_bytes = BATCH * HEADS * SEQUENCE * SEQUENCE * 2
        components = {
            "linear-128": (
                lambda: F.linear(normalized, layer.ffn_in.weight, layer.ffn_in.bias),
                2 * BATCH * SEQUENCE * MODEL * MODEL,
                "compute",
            ),
            "packed-qkv": (
                lambda: F.linear(normalized, hybrid.qkv_weights[0], hybrid.qkv_biases[0]),
                6 * BATCH * SEQUENCE * MODEL * MODEL,
                "compute",
            ),
            "head-major-qkv": (
                lambda: head_major_qkv(
                    normalized,
                    hybrid.qkv_weights[0],
                    hybrid.qkv_biases[0],
                ),
                6 * BATCH * SEQUENCE * MODEL * MODEL,
                "compute",
            ),
            "flash-attn": (
                lambda: attention(SDPBackend.FLASH_ATTENTION),
                attention_flops,
                "compute",
            ),
            "cudnn-attn": (
                lambda: attention(SDPBackend.CUDNN_ATTENTION),
                attention_flops,
                "compute",
            ),
            "exact-attn": (
                lambda: exact_hybrid._attention(
                    q,
                    k,
                    v,
                    None,
                    all_valid=True,
                    streaming=False,
                    compiled_softmax=False,
                    compiled_full_attention=False,
                ),
                6 * score_bytes + 5 * tensor_bytes,
                "memory",
            ),
            "exact-contig": (
                # Include all three copies: this is only a viable model-level
                # change if the friendlier B,H,S,D layout repays their cost.
                lambda: exact_hybrid._attention(
                    q.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    None,
                    all_valid=True,
                    streaming=False,
                    compiled_softmax=False,
                    compiled_full_attention=False,
                ),
                6 * score_bytes + 11 * tensor_bytes,
                "memory",
            ),
            "exact-qk-contig": (
                lambda: exact_hybrid._attention(
                    q.contiguous(),
                    k.contiguous(),
                    v,
                    None,
                    all_valid=True,
                    streaming=False,
                    compiled_softmax=False,
                    compiled_full_attention=False,
                ),
                6 * score_bytes + 9 * tensor_bytes,
                "memory",
            ),
            "exact-v-contig": (
                lambda: exact_hybrid._attention(
                    q,
                    k,
                    v.contiguous(),
                    None,
                    all_valid=True,
                    streaming=False,
                    compiled_softmax=False,
                    compiled_full_attention=False,
                ),
                6 * score_bytes + 7 * tensor_bytes,
                "memory",
            ),
            "exact-precontig": (
                lambda: exact_hybrid._attention(
                    contiguous_q,
                    contiguous_k,
                    contiguous_v,
                    None,
                    all_valid=True,
                    streaming=False,
                    compiled_softmax=False,
                    compiled_full_attention=False,
                ),
                6 * score_bytes + 5 * tensor_bytes,
                "memory",
            ),
            "context-copy": (
                lambda: q.transpose(1, 2).contiguous(),
                2 * tensor_bytes,
                "memory",
            ),
            "layer-norm": (
                lambda: layer.norm1(value),
                2 * tensor_bytes,
                "memory",
            ),
            "standalone-norm": (
                lambda: standalone_layer_norm(
                    value,
                    layer.norm1.weight,
                    layer.norm1.bias,
                ),
                2 * tensor_bytes,
                "memory",
            ),
            "residual-add": (
                lambda: value + branch,
                3 * tensor_bytes,
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
                    mask_branch=True,
                    mask_combined=False,
                ),
                4 * tensor_bytes,
                "memory",
            ),
            "linear-gelu": (
                lambda: linear_gelu(
                    normalized,
                    layer.ffn_in.weight,
                    layer.ffn_in.bias,
                ),
                2 * BATCH * SEQUENCE * MODEL * MODEL,
                "compute",
            ),
            "linear-res-norm": (
                lambda: linear_residual_layer_norm(
                    normalized,
                    layer.attention.out_proj.weight,
                    layer.attention.out_proj.bias,
                    value,
                    layer.norm2.weight,
                    layer.norm2.bias,
                    valid,
                    all_valid=True,
                    mask_branch=True,
                    mask_combined=False,
                ),
                2 * BATCH * SEQUENCE * MODEL * MODEL,
                "compute",
            ),
            "exact-gelu": (
                lambda: F.gelu(hidden, approximate="none"),
                2 * tensor_bytes,
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
                milliseconds = float(median)
                achieved = work / (milliseconds * 1.0e9)
                peak = (
                    H200_DENSE_FP16_TFLOPS
                    if resource == "compute"
                    else H200_HBM_TBPS
                )
                unit = "TFLOP/s" if resource == "compute" else "TB/s"
                print(
                    f"{name:14s} median={milliseconds:.6f} ms "
                    f"p20={float(low):.6f} p80={float(high):.6f} "
                    f"{achieved:.3f} {unit} {100.0*achieved/peak:.2f}% peak"
                )

        print("\n=== exact fusion tile sweep ===")

        def tune(operation):
            median, _, _ = triton.testing.do_bench(
                operation,
                warmup=10 if args.quick else 20,
                rep=30 if args.quick else 100,
                quantiles=[0.5, 0.2, 0.8],
            )
            return float(median)

        with torch.inference_mode():
            expected_residual = residual_layer_norm(
                value,
                branch,
                layer.norm2.weight,
                layer.norm2.bias,
                valid,
                all_valid=True,
                mask_branch=True,
                mask_combined=False,
            )
            for rows in (2, 4, 8, 16):
                for warps in (2, 4, 8):
                    try:
                        operation = lambda rows=rows, warps=warps: residual_layer_norm(
                            value,
                            branch,
                            layer.norm2.weight,
                            layer.norm2.bias,
                            valid,
                            all_valid=True,
                            mask_branch=True,
                            mask_combined=False,
                            _row_tile=rows,
                            _num_warps=warps,
                        )
                        actual = operation()
                        exact = all(
                            torch.equal(expected, candidate)
                            for expected, candidate in zip(expected_residual, actual)
                        )
                        milliseconds = tune(operation) if exact else float("nan")
                        print(
                            f"residual rows={rows:3d} warps={warps} "
                            f"exact={exact} median={milliseconds:.6f} ms"
                        )
                    except Exception as error:
                        print(
                            f"residual rows={rows:3d} warps={warps} "
                            f"ERROR {type(error).__name__}: {error}"
                        )

            expected_norm = layer.norm1(value)
            for rows in (2, 4, 8, 16):
                for warps in (2, 4, 8):
                    try:
                        operation = lambda rows=rows, warps=warps: standalone_layer_norm(
                            value,
                            layer.norm1.weight,
                            layer.norm1.bias,
                            _row_tile=rows,
                            _num_warps=warps,
                        )
                        actual = operation()
                        exact = torch.equal(expected_norm, actual)
                        milliseconds = tune(operation) if exact else float("nan")
                        print(
                            f"norm rows={rows:3d} warps={warps} "
                            f"exact={exact} median={milliseconds:.6f} ms"
                        )
                    except Exception as error:
                        print(
                            f"norm rows={rows:3d} warps={warps} "
                            f"ERROR {type(error).__name__}: {error}"
                        )

            expected_qkv = (q.contiguous(), k.contiguous(), v.contiguous())
            qkv_configs = {
                (rows, 64, warps, stages)
                for rows in (32, 64, 128, 256)
                for warps in (4, 8)
                for stages in (2, 3)
            }
            qkv_configs.update(((64, 32, 4, 3), (64, 128, 4, 3)))
            for rows, reduction, warps, stages in sorted(qkv_configs):
                try:
                    def operation(
                        rows=rows,
                        reduction=reduction,
                        warps=warps,
                        stages=stages,
                    ):
                        return head_major_qkv(
                            normalized,
                            hybrid.qkv_weights[0],
                            hybrid.qkv_biases[0],
                            _row_tile=rows,
                            _reduction_tile=reduction,
                            _num_warps=warps,
                            _num_stages=stages,
                        )

                    actual = operation()
                    exact = all(
                        torch.equal(expected, candidate)
                        for expected, candidate in zip(expected_qkv, actual)
                    )
                    milliseconds = tune(operation) if exact else float("nan")
                    print(
                        f"qkv rows={rows:3d} k={reduction:3d} warps={warps} "
                        f"stages={stages} exact={exact} "
                        f"median={milliseconds:.6f} ms"
                    )
                except Exception as error:
                    print(
                        f"qkv rows={rows:3d} k={reduction:3d} warps={warps} "
                        f"stages={stages} ERROR {type(error).__name__}: {error}"
                    )

            expected_gelu = linear_gelu(
                normalized,
                layer.ffn_in.weight,
                layer.ffn_in.bias,
            )
            gelu_configs = {
                (rows, 64, warps, stages)
                for rows in (64, 128, 256)
                for warps in (4, 8)
                for stages in (2, 3)
            }
            gelu_configs.update(((128, 32, 8, 3), (128, 128, 8, 3)))
            gelu_configs.add((128, 64, 8, 4))
            for rows, reduction, warps, stages in sorted(gelu_configs):
                try:
                    def operation(
                        rows=rows,
                        reduction=reduction,
                        warps=warps,
                        stages=stages,
                    ):
                        return linear_gelu(
                            normalized,
                            layer.ffn_in.weight,
                            layer.ffn_in.bias,
                            _row_tile=rows,
                            _reduction_tile=reduction,
                            _num_warps=warps,
                            _num_stages=stages,
                        )

                    actual = operation()
                    exact = torch.equal(expected_gelu, actual)
                    milliseconds = tune(operation) if exact else float("nan")
                    print(
                        f"gelu rows={rows:3d} k={reduction:3d} warps={warps} "
                        f"stages={stages} exact={exact} "
                        f"median={milliseconds:.6f} ms"
                    )
                except Exception as error:
                    print(
                        f"gelu rows={rows:3d} k={reduction:3d} warps={warps} "
                        f"stages={stages} ERROR {type(error).__name__}: {error}"
                    )

            expected_linear_norm = linear_residual_layer_norm(
                normalized,
                layer.attention.out_proj.weight,
                layer.attention.out_proj.bias,
                value,
                layer.norm2.weight,
                layer.norm2.bias,
                valid,
                all_valid=True,
                mask_branch=True,
                mask_combined=False,
            )
            norm_configs = {
                (rows, 64, warps, stages)
                for rows in (32, 64, 128)
                for warps in (4, 8)
                for stages in (2, 3)
            }
            norm_configs.update(((64, 32, 8, 3), (64, 128, 8, 3)))
            norm_configs.add((64, 64, 8, 4))
            for rows, reduction, warps, stages in sorted(norm_configs):
                try:
                    def operation(
                        rows=rows,
                        reduction=reduction,
                        warps=warps,
                        stages=stages,
                    ):
                        return linear_residual_layer_norm(
                            normalized,
                            layer.attention.out_proj.weight,
                            layer.attention.out_proj.bias,
                            value,
                            layer.norm2.weight,
                            layer.norm2.bias,
                            valid,
                            all_valid=True,
                            mask_branch=True,
                            mask_combined=False,
                            _row_tile=rows,
                            _reduction_tile=reduction,
                            _num_warps=warps,
                            _num_stages=stages,
                        )

                    actual = operation()
                    exact = all(
                        torch.equal(expected, candidate)
                        for expected, candidate in zip(
                            expected_linear_norm, actual
                        )
                    )
                    milliseconds = tune(operation) if exact else float("nan")
                    print(
                        f"linear-norm rows={rows:3d} k={reduction:3d} "
                        f"warps={warps} stages={stages} exact={exact} "
                        f"median={milliseconds:.6f} ms"
                    )
                except Exception as error:
                    print(
                        f"linear-norm rows={rows:3d} k={reduction:3d} "
                        f"warps={warps} stages={stages} "
                        f"ERROR {type(error).__name__}: {error}"
                    )

    if args.skip_latency:
        return 0

    print("\n=== latency ===")
    warmup, rep = (10, 30) if args.quick else (100, 500)
    with torch.inference_mode():
        for name in args.providers:
            model = models[name]
            median, low, high = triton.testing.do_bench(
                lambda: model(value, valid),
                warmup=warmup,
                rep=rep,
                quantiles=[0.5, 0.2, 0.8],
            )
            milliseconds = float(median)
            print(
                f"{name:24s} median={milliseconds:.6f} ms "
                f"p20={float(low):.6f} p80={float(high):.6f} "
                f"achieved={roofline.achieved_tflops(milliseconds):.3f} TFLOP/s "
                f"roofline={roofline.percent(milliseconds):.2f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
