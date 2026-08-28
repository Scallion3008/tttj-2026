# H200 fused-megakernel optimization results

## Environment and final validation

- GPU: NVIDIA H200 NVL, 143771 MiB, 132 SMs, compute capability 9.0
- CUDA toolkit: 12.9.86 from `/usr/local/cuda-12.9`
- Project environment: Python 3.12, PyTorch 2.13.0+cu129, Triton 3.7.1
- Accuracy gate: absolute error <= 0.001 OR relative error <= 0.01
- Full cases-5/6 job: `768869`
- Mask/determinism stress job: `768868`
- Full standalone checkpoint job: `768910`
- End-to-end stage diagnostic job: `768909`
- Final post-cleanup quick job: `768939`
- Nsight Compute attempt: `768769`

All jobs used one `gpu:h200-141` on `xgpk0` and ran through the project lockfile
with `uv run --frozen`.

## Final implementation

Cases 5 and 6 execute as one Triton launch with one four-warp CTA per sequence.
Each CTA runs all four transformer layers, exact causal attention, final
LayerNorm, and output masking. The production path does not invoke Torch SDPA
or any other Torch math kernel.

The final H200 schedule differs from the first WGMMA version in several ways:

- Linear operations use one 64x128 output tile with a 64-wide reduction tile,
  replacing four 64x64/K32 dot calls per 128x128 linear operation.
- Attention QK/PV operations retain efficient full 64x128 score tiles and use
  a 32-wide reduction tile. Pruning the causal upper-right tile was slower.
- Triton uses three pipeline stages. One stage was invalid, while four and five
  stages increased shared-memory pressure and regressed.
- Layer 0 consumes the input tensor directly instead of staging it into the
  workspace.
- The final activation resides in the output tensor throughout the four-layer
  loop, and final LayerNorm writes in place.
- Q is produced last in place over dead normalized input. Attention then
  overwrites Q head-by-head with context. Dead K and V storage is reused by the
  output projection, second LayerNorm, and FFN. This reduces scratch from six
  32 KiB slots (192 KiB per sequence) to three slots (96 KiB).
- Redundant source-level CTA barriers were removed after repeated-output and
  padded-mask stress testing. Compiler-required WGMMA/barrier instructions
  remain in the generated code.
- All-valid masks select a compile-time specialization after a one-time cached
  mask reduction. The general padded-mask specialization remains bit-exact.

Softmax uses one accurately rounded reciprocal per row. Each probability first
uses reciprocal multiplication, then computes a fused quotient residual and
applies one correction:

```text
q0 = numerator * reciprocal(denominator)
r  = fma(-q0, denominator, numerator)
q1 = q0 + r * reciprocal(denominator)
```

This recovered the same FP16 probabilities as elementwise `div.rn.f32` in all
validation while avoiding one full hardware division per probability.

The selected all-valid cubin uses 255 registers per thread and 73,728 bytes of
Triton-managed shared memory. Its PTX contains 26
`wgmma.mma_async...m64n128k16` instructions and eight
`wgmma.mma_async...m64n32k16` instructions in the statically unrolled body.

## Nsight Compute and static profiling

`/usr/local/cuda-12.9/bin/ncu` version 2026.2.1 connected to the H200 and found
the target kernel, but the node rejected access to hardware performance
counters with `ERR_NVGPUCTRPERM`. The reusable profiling entry point is
`job-scripts/profile_megakernel_h200.sh`; an administrator must enable GPU
performance-counter access before its detailed sections can be collected.

Static cubin/PTX/SASS inspection still identified useful constraints:

- the original exact kernel used 250 registers/thread and only four warps;
- WGMMA operations had frequent immediate dependency waits;
- elementwise accurate softmax division was a large scalar bottleneck;
- the final all-valid cubin uses 255 registers/thread and 72 KiB dynamic shared
  memory, fitting two CTAs within the H200's register/shared-memory limits;
- the final static SASS contains 34 HGMMA instructions, eight warp-group
  arrivals, 14 warp-group dependency waits, 97 global loads, and 74 global
  stores. Static counts describe the unrolled instruction image, not dynamic
  executed instruction totals.

## Accuracy

The full benchmark passed bit-for-bit:

- Case 5: three trials, 0 / 6,291,456 failed output elements.
- Case 6: one trial, 0 / 163,840,000 failed output elements.

Job `768868` added five trials each at 0%, 25%, and 75% padding for case 5.
All 15 trials were bit-exact, and a second launch of every input matched the
first launch exactly. The standalone step-1 checkpoints also remain within the
numerical gate through their separate diagnostic path. The end-to-end stage
trace likewise had zero gate failures at every recorded layer boundary.

## Performance

Full benchmark results from job `768869`:

| Case | PyTorch median | Final megakernel median | Speedup | Throughput |
| --- | ---: | ---: | ---: | ---: |
| 5: B128 | 1.3617 ms | 0.2375 ms | 5.733x | 68,984,102 token/s |
| 6: B10000 | 85.9093 ms | 12.2233 ms | 7.028x | 104,718,349 token/s |

Case 5 used 10 warmups, 30 repeats, and three alternating-order rounds. Case 6
used 3 warmups, 10 repeats, and two alternating-order rounds. CUDA events on
the current stream measured each call.

Relative to the first exact WGMMA megakernel from job `768744`, optimized
latency improved from 0.6937 to 0.2375 ms for case 5 (2.92x faster kernel) and
from 32.3355 to 12.2233 ms for case 6 (2.65x faster kernel).

## Rejected alternatives

- Fast reciprocal without quotient correction reached about 0.40/20.9 ms but
  failed 10 case-5 and 803 case-6 outputs.
- A 128-wide linear reduction tile was fast for case 5 but changed accumulation
  enough to miss the numerical gate and regressed case 6.
- Eight warps, smaller causal attention tiles, approximate GELU, and attention
  K16 all regressed or failed accuracy.
- A raw CUDA shared-memory WMMA prototype compiled successfully but measured
  1.1415/87.0174 ms and failed the gate; it was removed.
- A layerwise Torch SDPA comparison measured 0.7552/40.8155 ms and failed the
  gate. It was slower than the fused path and was removed.
- Initial Triton `num_ctas=2` and `num_ctas=4` attempts failed compiler
  lowering. A later exact two-CTA experimental path is documented in
  [cluster_h200_results.md](cluster_h200_results.md); it remains slower than
  the one-CTA production launch.
