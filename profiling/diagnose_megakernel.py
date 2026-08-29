"""Compare every fused-megakernel stage with the PyTorch reference."""

import os

import torch
import torch.nn.functional as F

from benchmarks.benchmark_steps_1_2 import compare_stage, make_models
from benchmarks.torch_transformer_benchmark import generate_random_case
from kernels.fused_megakernel import fused_megakernel_forward


batch_size = int(os.environ.get("DIAG_BATCH", "1"))
sequence_index = int(os.environ.get("DIAG_INDEX", "0"))
baseline, optimized, config = make_models(batch_size, 1234, False)
x_batch, valid_batch = generate_random_case(
    config, torch.device("cuda"), torch.float16, 1234, 0.0, 1.0
)
x = x_batch[sequence_index : sequence_index + 1]
valid = valid_batch[sequence_index : sequence_index + 1]
with torch.inference_mode():
    _, trace = fused_megakernel_forward(
        x, valid, optimized.packed_weights, capture_trace=True
    )
    reference_x = x
    references = []
    names = []
    attention_references = []
    for layer_index, layer in enumerate(baseline.layers):
        norm1 = layer.norm1(reference_x)
        q_linear = layer.attention.q_proj(norm1)
        k_linear = layer.attention.k_proj(norm1)
        v_linear = layer.attention.v_proj(norm1)
        q = layer.attention._split_heads(q_linear)
        k = layer.attention._split_heads(k_linear)
        v = layer.attention._split_heads(v_linear)
        scores = torch.matmul(q, k.transpose(-2, -1)) * layer.attention.scale
        causal = torch.ones((128, 128), device="cuda", dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal, float("-inf"))
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        probabilities_fp32 = torch.softmax(scores.float(), dim=-1)
        probabilities = probabilities_fp32.to(torch.float16)
        maximum = scores.float().max(dim=-1, keepdim=True).values
        numerator = torch.exp(scores.float() - maximum)
        lane_values = numerator.reshape(1, 4, 128, 4, 32).permute(0, 1, 2, 4, 3)
        denominator = torch.zeros_like(lane_values[..., 0])
        for item in range(4):
            denominator = denominator + lane_values[..., item]
        width = 32
        while width > 1:
            halves = denominator.reshape(1, 4, 128, 2, width // 2).permute(
                0, 1, 2, 4, 3
            )
            denominator = halves[..., 0] + halves[..., 1]
            width //= 2
        denominator = denominator.reshape(1, 4, 128)
        attention_references.append(
            (scores[0], probabilities[0], numerator[0], denominator[0])
        )
        context = torch.matmul(probabilities, v)
        context = context.transpose(1, 2).contiguous().view(1, 128, 128)
        attention_branch = layer.attention.out_proj(context).masked_fill(
            ~valid[..., None], 0
        )
        attention_residual = reference_x + attention_branch
        norm2 = layer.norm2(attention_residual)
        hidden = F.gelu(layer.ffn_in(norm2), approximate="none")
        ffn_residual = (attention_residual + layer.ffn_out(hidden)).masked_fill(
            ~valid[..., None], 0
        )
        values = (
            norm1,
            q_linear,
            k_linear,
            v_linear,
            context,
            attention_residual,
            norm2,
            hidden,
            ffn_residual,
        )
        for stage, value in enumerate(values):
            references.append(value[0])
            names.append(f"layer{layer_index}.{stage}")
        reference_x = ffn_residual
    references.append(baseline.final_norm(reference_x)[0])
    names.append("final_norm")

for index, (name, reference) in enumerate(zip(names, references)):
    candidate = trace[index]
    result = compare_stage(reference, candidate)
    exact = int((reference != candidate).sum().item())
    print(
        f"{name:12s} exact_diff={exact:6d} gate_fail={result.failed:5d} "
        f"max_abs={result.max_abs:.7f} gate_ratio={result.max_gate_ratio:.4f}"
    )

for layer_index, (scores, probabilities, numerator, denominator) in enumerate(
    attention_references
):
    for name, reference, candidate in (
        ("scores", scores, trace[37 + layer_index * 8 : 41 + layer_index * 8]),
        (
            "softmax",
            probabilities,
            trace[41 + layer_index * 8 : 45 + layer_index * 8],
        ),
        (
            "exp",
            numerator,
            trace[69 + layer_index * 4 : 73 + layer_index * 4],
        ),
        (
            "denom",
            denominator,
            trace[85 + layer_index].reshape(-1)[: 4 * 128].reshape(4, 128),
        ),
    ):
        result = compare_stage(reference, candidate)
        exact = int((reference != candidate).sum().item())
        print(
            f"layer{layer_index}.{name:7s} exact_diff={exact:6d} "
            f"gate_fail={result.failed:5d} max_abs={result.max_abs:.7f} "
            f"gate_ratio={result.max_gate_ratio:.4f}"
        )
        if exact and layer_index == 0:
            locations = torch.nonzero(reference != candidate, as_tuple=False)[:4]
            for location in locations:
                key = tuple(location.tolist())
                print(
                    f"  {key}: reference={reference[key].item():.10g} "
                    f"candidate={candidate[key].item():.10g}"
                )
