# H200 fused-megakernel results

## Environment

- Slurm job: `768744`
- GPU: NVIDIA H200 NVL, 143771 MiB, 132 SMs, compute capability 9.0
- CUDA toolkit: 12.9.86 from `/usr/local/cuda-12.9`
- Project environment: Python 3.12, PyTorch 2.13.0+cu129, Triton 3.7.1
- Accuracy gate: absolute error <= 0.001 OR relative error <= 0.01

The benchmark ran through the project lockfile with `uv run --frozen`.

## Implementation

Cases 5 and 6 now use one Triton launch per batch. Each program owns one
sequence and executes all four transformer layers plus the final LayerNorm.
The launch includes LayerNorm, Q/K/V projections, exact causal attention,
output projection and residual, exact GELU, both FFN projections and the final
valid-token mask. The megakernel body does not invoke Torch math kernels or
Torch SDPA; the Python wrapper only allocates its output and workspace.

The fixed 128x128 linear operations use 64x64 FP16/FP32 WGMMA tiles. QK and PV
use 64x32 WGMMA tiles. Generated PTX contains both expected Hopper
instructions:

```text
wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16
wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16
```

Numerical equivalence required matching the CUDA reference algorithms rather
than relying on generic reductions:

- LayerNorm uses the aligned FP16 `half4` Welford update and warp reduction
  order used by PyTorch.
- Softmax uses four values per lane followed by the CUDA shuffle-XOR reduction
  tree.
- Softmax division uses IEEE round-to-nearest FP32 division (`div.rn.f32`).
- QK scores retain the reference FP16 storage boundaries around scaling and
  masking.

## Accuracy

- Every standalone step-1 checkpoint passed at all nine input scales, both
  with no padding and with 25% padding.
- Case 5 passed all three full-model trials bit-for-bit: `max_abs=0`,
  `max_rel=0`, and 0 / 6,291,456 failed elements.
- Case 6 passed its full-model trial bit-for-bit: `max_abs=0`, `max_rel=0`, and
  0 / 163,840,000 failed elements.

## Performance

| Case | PyTorch median | Megakernel median | Speedup | Megakernel throughput |
| --- | ---: | ---: | ---: | ---: |
| 5: B128 | 1.3831 ms | 0.6937 ms | 1.994x | 23,617,326 token/s |
| 6: B10000 | 85.8820 ms | 32.3355 ms | 2.656x | 39,585,031 token/s |

Case 5 used 10 warmups, 30 repeats, and three alternating-order rounds. Case 6
used 3 warmups, 10 repeats, and two alternating-order rounds. CUDA events on
the current stream measured each call. The generated full log is
`job-scripts/run_steps_1_2-768744.out` and is ignored by git.
