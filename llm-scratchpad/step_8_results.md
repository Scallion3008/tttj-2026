# Step 8 / case 14 long-context results

Date: 2026-08-31 (Asia/Singapore)

Case 14 is `B=32, S=100000, D=1024, H=16, HD=64, L=2, F=1024`,
causal FP16. Useful reference-model work is 1.391263744 PFLOP, of which
94.21% is QK/PV attention work.

## Megakernel decision

The original rejection of a model-wide megakernel remains correct:

- one `B x S x D` activation is 6.10 GiB;
- the complete FP16 score tensor would contain 5.12e12 elements and occupy
  10.24 TB (decimal);
- a 128-query attention grid already exposes 400,384 independent
  batch/head/query tasks; and
- the M=3.2M linear GEMMs fully amortize their launches.

Thus model-wide residency is impossible and launch savings are immaterial.
The appropriate structure is separate large GEMMs plus a memory-bounded
attention implementation. The initial note recommended online streaming
attention. Official FA3, cuDNN frontend SDPA, PyTorch Flash SDPA, and a custom
three-pass online Triton kernel confirmed that this is the fastest structure,
but all failed the repository's strict `atol=0.001, rtol=0.01` output gate.
The reference exposes FP16 score and probability boundaries and PyTorch's
large-row reduction order, so production must preserve those boundaries.

## Production structure

The selected layerwise path uses:

1. separate cuBLAS Q/K/V projections, immediately converted to head-major
   storage;
2. descending 640-query causal prefix tiles on H200 (512 on H100-96),
   allocating the largest score tile first so the allocator reuses it;
3. cuBLAS QK with the reference FP16 output boundary;
4. a CUDA 1024-thread, ILP-4 softmax-statistics kernel that duplicates the
   pinned PyTorch large-row max/sum reduction and FP16 scaling order; and
5. a Triton M128/N64/W8 PV mainloop which recomputes the exact exponential and
   corrected division, casts each probability to FP16, and feeds it directly
   to the tensor-core dot product. This avoids writing and rereading the
   probability tensor. An H200 sweep of query chunks 128 through 896 selected
   640; 512 is within 0.25% but uses 12.2 GiB less memory.

Residual/LayerNorm boundaries use the case-8 Triton fusion with INT64 element
offsets. D1024 output/FFN projections remain standalone cuBLAS GEMMs. The
custom extension is built during construction rather than the timed first
forward. Expandable allocator segments prevent tens-of-GiB score tiles from
fragmenting the caching allocator. A synchronized allocator-cache release is
used only on sub-120-GiB GPUs so the 512-query score allocation fits an
H100-96; H200 does not take that fallback.

## Numerical results

The exact materialized-probability path passed three full `B=1, S=100000`
trials bit-for-bit. The fused statistics/PV path also passed three such trials
bit-for-bit (0 differences among 102.4M output elements in each trial, jobs
775155 and 775213, the latter through the public factory). It passed 20 padded
`B=1, S=4096` trials through the production public class (job 775205; padding
ratio 0.25). Full-length B=1 tests at input scales 1e-4, 1e-3, 3e-3, 1e-2,
0.1, 1, 10, 100, and 1000 were bit-exact for all 921.6M tested values (job
776569). The public factory also passed the actual B32/S100000 shape
bit-for-bit: 0 differences among 3.2768B outputs at both q-chunk 512 (job
775214) and the selected q-chunk 640 (job 776582).

Rejected fast paths:

- official FA3 measured 2.897720 s but failed the strict full-length B=1 gate
  (30 failures among 102.4M tested outputs in the representative trial);
- using exact attention only in one layer still produced rare full-length
  failures;
- approximate exponential in the exact softmax failed full-length trials;
  using it only in fused PV still failed 2 of 20 short trials (job 775206);
- integrating the QK scale into GEMM, online Triton reductions, cuDNN, Flash
  SDPA, and FA3/cuDNN blends all produced strict-gate failures.

## H200 profiling and latency

H200 reference peaks used here are 835.5 dense FP16 TFLOP/s and 4.8 TB/s HBM.
Production components use a representative B32/H16/M640/N100000 tile; the
materialized-probability comparison used M512:

H100-96 Nsight Compute slots remained unavailable during the optimization
window (the reservations moved to late 2026-08-31), so the allowed H200 CUDA
event/Triton benchmarking fallback was used for binding metrics.

| Component | Time | Binding metric | Fraction of H200 reference peak |
|---|---:|---:|---:|
| QK cuBLAS, M640 | 22.559 ms | 3.197 TB/s minimum traffic | 66.60% HBM |
| Exact full softmax (materialized PV path) | 70.860 ms | 2.954 TB/s effective | 61.54% HBM |
| PV cuBLAS | 13.808 ms | 4.274 TB/s minimum traffic | 89.04% HBM |
| Exact max/sum statistics, M640 | 41.224 ms | 3.169 TB/s effective | 66.02% HBM |
| Fused exact probability/PV, M640, tile M128/N64/W8 | 38.017 ms | 2.587 TB/s minimum traffic plus exact exp | 53.90% HBM |

K=64 makes the standalone QK and PV operations bandwidth-bound rather than
dense-tensor-compute-bound. The exact reduction and fused PV are mixed
memory/exact-exponential kernels; the remaining headroom is in that combined
exact-exp/tensor mainloop. Approximate exp made fused PV 9.36% faster but was
rejected by correctness. Vectorized four-half loads improved
the full softmax from 93.122 ms / 2.248 TB/s to 70.860 ms / 2.954 TB/s.
Fusing the probability epilogue with PV removes one probability write/read
round trip. Together with final chunk/allocator tuning it reduced full-model
latency by 16.83% relative to the materialized-probability path.

| Strict-gate implementation | H200 median | Useful TFLOP/s | Peak allocation |
|---|---:|---:|---:|
| Feasible exact Torch reference, q-chunk 64 | 146.449594 s | 9.500 | -- |
| Exact custom softmax + materialized PV | 19.251271 s | 72.269 | 85.453 GiB |
| Exact stats + fused probability/PV, q-chunk 640 | 16.010808 s | 86.895 | 97.643 GiB |

The current strict implementation is **9.147x** faster than the feasible
memory-bounded Torch reference. The organizer's unchunked reference cannot be
run at this shape because it attempts to materialize roughly 10.24 TB of
scores.

Representative timing jobs: 774854 (Torch reference), 775083 (materialized
custom softmax), 776165 (query sweep), 776602 (final component profile), and
776582 (final public-factory full-shape accuracy plus timing). H100-47 MIG
jobs were used only for correctness; none of their latency data was used.
