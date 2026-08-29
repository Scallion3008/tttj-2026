# Benchmarking steps 1 and 2: H200 results

Date: 2026-08-29 (Asia/Singapore)

## Environment

- Node: `xgpk0`
- GPU: NVIDIA H200 NVL, 143771 MiB, 132 SMs, compute capability 9.0
- CUDA toolkit: 12.9.86 from `/usr/local/cuda-12.9`
- PyTorch: 2.8.0+cu129
- Data type: FP16, with FP32 scalar accumulation and reductions in the custom
  kernel
- Strict gate: absolute error <= 0.001 OR relative error <= 0.01

The primary full run was Slurm job `768634`. The corrected checkpoint-only run
was job `768637`.

## Step 1: FP16 correctness checkpoints

The checkpoint harness compares the first layer after:

1. LayerNorm
2. Q projection
3. K projection
4. V projection
5. scaled and masked scores
6. FP32 softmax cast back to FP16
7. PV
8. masked output projection
9. FFN branch

It sweeps input scales
`1e-4, 1e-3, 3e-3, 1e-2, 0.1, 1, 10, 100, 1000` with both all-valid inputs
and `padding_ratio=0.25`.

Result: **PASS for all nine stages in all 18 scale/padding combinations.**

The largest observed normalized gate consumption was 0.5493 at the FFN
checkpoint; all checkpoints therefore retained at least about 1.82x margin to
the strict gate in this sweep. The authoritative log is
`job-scripts/outputs/run_step_1_h200-768637.out`.

The first run incorrectly compared raw PV values for invalid query rows even
though the prototype zeroed those rows early and the reference zeroed them
after output projection. The kernel was changed to preserve raw PV values and
apply the required invalid-query mask at output projection. This made the
checkpoint location agree with the reference without changing valid outputs.

## Step 2: one-CTA sequence-resident prototype

The prototype uses one 256-thread CTA per resident sequence and 160 KiB of
dynamic shared memory for `x`, normalized scratch, Q, K, and V. It executes all
four layers before returning the sequence and obtains further sequences from a
device-side atomic global queue. Linear algebra is currently scalar FP32 FMA;
this version is intended to validate residency and queue mechanics, not to be
a competitive tensor-core implementation.

### Case 5: B128, S128, D128, H4, L4, F128

| Metric | PyTorch baseline | One-CTA prototype |
| --- | ---: | ---: |
| Median latency | 1.3777 ms | 29.8429 ms |
| Throughput | 11,892,456.85 token/s | 549,007.79 token/s |
| Reported speedup | - | 0.046x |

Strict final-output accuracy over three trials:

- FAIL: 1,701 / 6,291,456 elements (0.0270%)
- maximum absolute error: 0.0078125
- per-trial maximum absolute error: 0.0078125

### Case 6: B10000, S128, D128, H4, L4, F128

| Metric | PyTorch baseline | One-CTA prototype |
| --- | ---: | ---: |
| Median latency | 85.7265 ms | 2266.5144 ms |
| Throughput | 14,931,212.03 token/s | 564,743.82 token/s |
| Reported speedup | - | 0.038x |

Strict final-output accuracy over one trial:

- FAIL: 45,042 / 163,840,000 elements (0.0275%)
- maximum absolute error: 0.00976562

## Conclusions

1. The 160 KiB shared-memory layout launches successfully on Hopper and the
   global sequence queue covers both the single-wave B128 case and the
   multi-wave B10000 case.
2. Every isolated first-layer rounding checkpoint passes, but small differences
   in scalar accumulation order compound across four layers and fail the final
   strict gate for about 0.027% of outputs. Later-layer checkpoints are needed
   before treating the body as numerically validated.
3. Scalar FMA makes this prototype 21.7x slower than baseline for case 5 and
   26.4x slower for case 6. The next body must replace the six linear paths and
   QK/PV with tensor-core MMA/WGMMA schedules; optimizing the scalar prototype
   incrementally is not worthwhile.
