# Step 8 / case 14 long-context results

Date: 2026-08-31 (Asia/Singapore)

Case 14 is `B=32, S=100000, D=1024, H=16, HD=64, L=2, F=1024`,
causal FP16. Useful reference-model work is 1.391263744 PFLOP, of which
94.21% is QK/PV attention work.

## Megakernel decision

The original rejection of a model-wide megakernel remains correct:

- one `B x S x D` activation is 6.10 GiB;
- the complete FP16 score tensor contains 5.12e12 elements and occupies
  10.24 TB (decimal);
- a 128-query attention grid already exposes 400,384 independent
  batch/head/query tasks; and
- the M=3.2M linear GEMMs fully amortize their launches.

The appropriate structure is separate large GEMMs plus memory-bounded online
attention. FA3, cuDNN SDPA, PyTorch Flash SDPA, and custom online Triton kernels
confirmed this. They all fail the original `atol=0.001, rtol=0.01` gate by
rare outliers. After the allowed gate was doubled to
`atol=0.002, rtol=0.02`, stock FA3 passed and became the production path. The
production dispatcher automatically retains the exact implementation below
input RMS 0.75; `TTTJ_STEP8_STRICT_ATTENTION=1` forces it at every RMS.

## Fast production structure

The selected path uses:

1. one packed cuBLAS QKV projection per layer;
2. stock Flash Attention 3's persistent SM90 causal FP16 forward kernel, using
   a 192x128 QK tile, FP32 online softmax state, FP16 block-local exponential
   fragments, and FP32 output accumulation; and
3. the existing fused residual/LayerNorm boundaries, standalone cuBLAS
   output/FFN projections, and exact GELU.

The dispatcher accumulates the input norm directly from FP16 into FP32, so it
does not materialize a 12.2-GiB FP32 copy of the full input. It caches the
decision for an unchanged tensor using weak object identity plus the tensor
version. Weak identity is necessary because the CUDA allocator can reuse the
same address for a new input, while retaining the tensor itself would pin a
6.10-GiB allocation. Both packed FA3 weights and the exact extension are
prepared at construction, making either branch immediately available.

The retained exact fallback materializes descending 640-query score prefixes
on H200 (512 on H100), uses cuBLAS QK, a CUDA exact-order softmax-statistics
kernel, and a fused Triton probability/PV kernel. Five long vector loops in
the exact CUDA reduction are now unrolled by 8 and two redundant entry
barriers were removed. The latest measured unroll-8 exact path took 15.716 s
on H200 and remained bit-exact in five full H100-47 B1 trials and one actual
B32/S100k trial. The final environment-controlled fallback also passed a fresh
full H100-47 B1 trial bit-for-bit (job 780506). The automatic low-RMS route
passed all tested full H100-47 B1 trials bit-for-bit (jobs 780582 and 780589).

## Numerical bottleneck and gate

H100-47 MIG stage diagnostics at B1/S4096 isolated the error (job 780374):

- FP32 PV versus the reference FP16 PV changes the layer-0 context by only
  `4.4e-8` mean absolute error, so PV accumulation is not limiting.
- Stock FA3 LSE is much closer to an FP32-score reference (`3.7e-7` mean error)
  than to the benchmark's FP16-rounded score reference (`2.18e-6`). FA3 does
  not expose the reference's FP16 QK and separate FP16 scale boundaries.
- The FA3 attention context itself passes even the original strict gate. Rare
  failures appear only after two residual/LayerNorm/FFN amplification chains.
- The larger remaining mismatch is the probability boundary: PyTorch globally
  normalizes, casts normalized probabilities to FP16, then performs PV. FA3
  casts block-local unnormalized exponentials to FP16, repeatedly rescales its
  FP32 output accumulator, and normalizes only at the end. Reproducing the
  former exactly requires a second pass and changes FA3's one-pass structure.

At the original gate, stock FA3 fails only 17--27 of 102.4M final outputs per
full B1/S100k trial. At the doubled gate it passed five full trials with zero
failures among 512M outputs through the final public factory (job 780504). It
also passed three full prefix-padded trials, zero failures among 307.2M outputs
(job 780505). cuDNN and PyTorch Flash SDPA also passed three full trials each
(jobs 780395 and 780396), but their
attention-only H200 times were slower: 1.454 s and 2.278 s versus FA3's
1.205 s.

The fast path is not scale-robust at small input magnitudes. The original
stress sweep failed at scales 1e-4 through 0.1, while scales 1, 10, 100, and
1000 passed. A finer full-length H100-47 sweep (job 780578) found 492--538
failures per 102.4M outputs at RMS about 0.25 and 11--13 at RMS about 0.50;
RMS about 0.75 passed three trials, zero failures among 307.2M outputs. The
production cutoff is therefore 0.75: smaller inputs use exact fused attention,
while the standard RMS-about-1 input uses FA3. End-to-end production routing
passed two trials each at nominal scales 0.5, 0.75, and 1 (job 780589). The
first four actual RMS values were just below the cutoff and routed exact; both
RMS-about-1 values routed FA3. Job 780582 additionally verified switching from
two exact RMS-about-0.1 inputs through two exact RMS-about-0.5 inputs to two
FA3 RMS-about-1 inputs in one model instance. All 614.4M outputs in each
production sweep passed, with the exact branch bit-for-bit. A final production
check just above the threshold at RMS about 0.80 routed FA3 and passed both
trials, zero failures among 204.8M outputs (job 780636).

A patched FA3 experiment restored both FP16 score boundaries inside the same
persistent online mainloop. It reduced LSE error against the FP16-score
reference from `2.18e-6` to `3.39e-7`, but context mean error improved only
3.6% and H200 model latency regressed from 2.905 s to 4.889 s. The experiment
was rejected and the stock dependency restored. Earlier exact-exp, score-only
rounding, split-K, backend blending, and exact-one-layer experiments likewise
did not clear the original strict gate without losing too much speed.

## Profiling and latency

H200 reference peaks used for exact-kernel analysis are 835.5 dense FP16
TFLOP/s and 4.8 TB/s HBM. The exact fallback components use a representative
B32/H16/M640/N100000 tile; the materialized-probability comparison used M512:

| Component | Time | Binding metric | Fraction of H200 reference peak |
|---|---:|---:|---:|
| QK cuBLAS, M640 | 22.559 ms | 3.197 TB/s minimum traffic | 66.60% HBM |
| Exact full softmax | 70.860 ms | 2.954 TB/s effective | 61.54% HBM |
| PV cuBLAS | 13.808 ms | 4.274 TB/s minimum traffic | 89.04% HBM |
| Exact max/sum statistics, M640 | 39.665 ms | 3.294 TB/s effective | 68.62% HBM |
| Fused exact probability/PV, M640, M128/N64/W8 | 38.474 ms | 2.556 TB/s minimum traffic plus exact exp | 53.25% HBM |

K=64 makes standalone QK and PV bandwidth-bound. The exact reduction and
fused PV are mixed memory/exact-exponential kernels. Approximate exp speeds PV
up but fails correctness. The exact score/probability boundaries force
materialization or recomputation and explain the large gap to FA3.

Nsight Compute job 780364 profiled the full B32/S100k FA3 kernel on an H100-96.
It is compute-bound: 64.90% SM throughput and 62.93% tensor-pipe activity,
versus 0.52% DRAM and 27.39% L2 throughput. The profiled kernel took 1.33 s on
H100-96. This supports retaining its persistent compute schedule rather than
adding global-memory correction passes.

| Implementation | H200 median | Useful TFLOP/s | Peak allocation |
|---|---:|---:|---:|
| Feasible exact Torch reference, q-chunk 64 | 146.449594 s | 9.500 | -- |
| Exact custom softmax + materialized PV | 19.251271 s | 72.269 | 85.453 GiB |
| Exact stats + fused probability/PV, q-chunk 640, unroll 8 | 15.716054 s | 88.525 | 97.643 GiB |
| Automatic production path, RMS 0.999995, FA3 selected | 2.895866 s | 480.431 | 36.905 GiB |

The FA3 path is **5.43x** faster than the current exact path and **50.57x**
faster than the feasible memory-bounded Torch reference. The
organizer's unchunked reference cannot run at this shape because it tries to
materialize roughly 10.24 TB of scores.

Representative jobs: 774854 (Torch reference), 775083 (materialized exact),
776165 (exact query sweep), 780364 (FA3 Nsight Compute), 780504 (final
public-factory doubled-gate accuracy), 780619 (automatic-dispatch H200 timing),
780374 (stage diagnostic), 780414 (rejected patched-FA3 timing), 780578
(threshold sweep), and 780582/780589/780636 (automatic-dispatch correctness).
H100-47 MIG jobs were used only for correctness; none of their latency data
was used.
