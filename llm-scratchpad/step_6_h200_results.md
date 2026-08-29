# Step 6: D1024 layerwise hybrid results

Date: 2026-08-30 (Asia/Singapore)

## Outcome

Case 8 now dispatches to a layerwise hybrid instead of an end-to-end
megakernel.  The original suitability analysis was correct: a D1024 sequence
activation is already 256 KiB, while the flattened M8192 x K1024 GEMMs have
enough work to favor tuned cuBLAS kernels.  The production path therefore
combines:

- one packed QKV cuBLAS projection per layer rather than three projections;
- a layout-aware Triton causal-attention kernel that consumes packed B,S,3D
  output and writes B,S,D context directly;
- exact fused residual + LayerNorm kernels at all eight residual boundaries;
- exact PyTorch GELU and cuBLAS for the remaining D1024 projections;
- an accuracy-gated final-layer Flash-attention specialization for ordinary
  input scale, with the exact custom path used for small inputs and padding;
- direct-input CUDA Graph replay for the fixed steady-state input allocation.

`Case8OptimizedTransformer` is integrated into the common case dispatcher.
Different input pointers and padded calls safely use the eager hybrid path;
the fixed all-valid benchmark replays the captured graph.

## Final H200 result

All tuning used the requested 132-SM H200 NVL proxy because the H100-96 pool
was saturated.  The node ran CUDA 12.9.86, Torch 2.13.0+cu129, and Triton
3.7.1.  Hardware counters are unavailable on this node, so all timings use
Triton's CUDA-event benchmark.

The final paired 500-repetition run was job 772216:

| Provider | Median | p20 | p80 | Useful throughput | Compute roofline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Explicit Torch reference | 2.351936 ms | 2.341478 ms | 2.363808 ms | 178.990 TFLOP/s | 21.42% |
| Case-8 production hybrid | 1.088000 ms | 1.084666 ms | 1.090048 ms | 386.925 TFLOP/s | 46.31% |

The measured speedup is **2.162x**.  The integrated common dispatcher was
separately measured at 1.079136 ms over 500 repetitions in job 772215.
Post-cleanup smoke job 772218 passed all 8,388,608 ordinary-scale outputs and
measured 1.049760 ms in its 30-repetition quick run.

The useful-work model counts 420.973904 GFLOP and gives 269.792 FLOP/B.  Its
4.8 TB/s memory line is about 1,295 TFLOP/s, above the H200 NVL dense-FP16
peak of 835.5 TFLOP/s, so the complete case is compute-bound.

## Torch SDPA comparison

Job 772223 compared the current production path with two Flash-SDPA baselines
over 500 repetitions.  The drop-in baseline retains the reference's separate
Q/K/V projections; the stronger SDPA variant packs QKV and captures the whole
four-layer model in a CUDA Graph.

| Provider | Median | Useful throughput | Strict numerical gate |
| --- | ---: | ---: | ---: |
| Explicit Torch reference | 2.344992 ms | 179.520 TFLOP/s | Pass |
| Drop-in Torch Flash SDPA | 1.999264 ms | 210.564 TFLOP/s | Fail: 1,639 values |
| Packed + graphed Torch Flash SDPA | 1.197360 ms | 351.585 TFLOP/s | Fail: 1,639 values |
| Case-8 production hybrid | 1.087792 ms | 386.999 TFLOP/s | Pass |

The production hybrid is 1.838x faster than drop-in Torch SDPA and 10.1%
faster than packed/graphed SDPA.  Both all-layer SDPA variants have maximum
absolute error 0.0078125 and exceed the organizer's strict per-value gate;
the production path uses Flash only in the accuracy-safe final layer.

## Component profiling

Job 772217 measured warmed component kernels.  GEMMs are compared with dense
FP16 compute peak; bandwidth kernels use their logical activation traffic and
the 4.8 TB/s HBM peak.

| Component | Median | Binding metric | Percent of peak |
| --- | ---: | ---: | ---: |
| D1024 cuBLAS linear | 0.033024 ms | 520.224 TFLOP/s | 62.26% compute |
| Packed D1024-to-3D QKV | 0.089824 ms | 573.784 TFLOP/s | 68.68% compute |
| Custom causal attention | 0.040576 ms | 2.067 TB/s | 43.07% bandwidth |
| Fused residual + LayerNorm | 0.023360 ms | 2.873 TB/s | 59.85% bandwidth |
| Native LayerNorm | 0.026944 ms | 1.245 TB/s | 25.94% bandwidth |
| Native residual add | 0.018496 ms | 2.721 TB/s | 56.69% bandwidth |
| Exact GELU | 0.016576 ms | 2.024 TB/s | 42.17% effective bandwidth |

The custom attention executes about 3.221 GFLOP after skipping the first
causal half-tile and transfers about 83.886 MB logically, giving roughly
38.4 FLOP/B.  Its memory roofline is therefore about 184 TFLOP/s, below dense
compute peak, which is why attention tuning focused on layout traffic and
causal work elimination rather than more tensor-core parallelism.

## Numerical validation

Job 772214 ran two trials for every combination of padding 0/0.25/0.75 and
scale 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, and 1000.  All 402,653,184 values
passed the strict `atol=0.001 OR rtol=0.01` gate, and repeated calls had zero
differences.  The largest accepted absolute difference was 0.005859375.

The exact custom-only path is bit-exact over this matrix.  At ordinary input
scale the production dispatch uses Flash only in the final layer; this remains
inside the gate.  A one-time RMS classification selects the bit-exact path for
small inputs and occurs before steady-state graph timing.

## Kernel tuning and rejected paths

The final attention uses M64 query tiles, K128 reduction tiles, four warps,
two stages, an explicit PyTorch-style S128 softmax tree, corrected reciprocal
division, libdevice `exp`, and causal first-half skipping.

- M32 regressed the graph to 1.236368 ms.  M128 was slower and failed 10
  outputs.  Two warps was slower and failed 7 outputs.
- K32 and K256 regressed to 1.266800 and 1.259424 ms respectively.  K64 was
  close but slower than K128 in paired configurations.
- Approximate exponential, generic `tl.sum` softmax, and cheaper reciprocal
  divisions all failed the strict gate without a useful latency gain.
- All-layer Flash, efficient SDPA, and math SDPA failed 1,639, 1,673, and
  1,540 outputs respectively.  Only final-layer Flash passed at ordinary
  scale.

The residual/LayerNorm fusion explicitly reproduces PyTorch's vectorized
half4 Welford ordering: 128 logical lanes consume two half4 vectors, reduce
within four logical warps, then reduce the four leaders.  Triton lowers this
best with `num_warps=1`; it remains bit-exact and reached 1.049232 ms in the
quick graph comparison before final pairing.

- One-pass and generic two-pass LayerNorm variants were faster arithmetically
  but failed roughly 147 outputs.
- An exact CUDA C++ implementation compiled with CUDA 12.9 but regressed the
  graph to 1.267216 ms, so it was removed.
- A Triton FFN-in GEMM with exact-GELU epilogue was bit-exact but regressed the
  graph to at least 1.226976 ms and was removed.
- Tanh GELU failed 1,950 outputs.  FP16 GEMM accumulation failed 179,162
  outputs and was also slower.

## Reproduction

```bash
# Full correctness and paired latency benchmark on the H200 fallback
sbatch --gres=gpu:h200-141:1 job-scripts/run_step_6_h200.sh \
  --accuracy-matrix --accuracy-trials 2 \
  --matrix-provider adaptive-candidate \
  --providers graph-adaptive-candidate graph-fused-custom torch

# Common integrated dispatcher
sbatch --gres=gpu:h200-141:1 \
  --output=job-scripts/outputs/step6-integrated-%j.out \
  job-scripts/benchmark_megakernels_h100.sh \
  --cases 8 --warmup 100 --rep 500

# Component roofline proxy
sbatch --gres=gpu:h200-141:1 job-scripts/run_step_6_h200.sh \
  --skip-accuracy --microbench --providers graph-adaptive-candidate
```

Every wrapper fails unless CUDA 12.9 exists at `/usr/local/cuda-12.9`.
