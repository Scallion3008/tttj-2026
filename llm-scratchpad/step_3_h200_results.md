# Benchmarking step 3: cases 1, 9, and 10 on H200

Date: 2026-08-29 (Asia/Singapore)

## Decision

Step 3 now uses a shape-specialized **one-CTA sequence-resident path**, not the
original two-CTA Triton cluster plan.  The earlier cluster investigation found
that Triton 3.7 PlanCTA required no-inline dot helpers, global score
materialization, cluster barriers, and duplicated dot work.  That path was
1.57--1.95x slower than the one-CTA production body for cases 5/6.  Repeating
that design for B=64 would not recover those structural costs.

The viable plan was therefore:

1. Generalize the proven one-CTA body to compile-time head counts 1, 2, and 4.
2. Emit a whole attention head (`HD=128/64/32`) from each PV WGMMA instead of
   hard-coding 32 output columns.
3. Use the extra warps available in B=64 shapes to process the full 128-row
   sequence in one linear tile and one attention tile.
4. Tune the head-specific attention reduction tile under the strict numerical
   gate.
5. Measure effective bandwidth percentage of the HBM roofline with
   `triton.testing.Benchmark` because H200 hardware counters are unavailable.

## Final implementation

`kernels/fused_megakernel.py` automatically selects the following schedule when
`B=64`, `S=D=F=128`, and `H` is 1, 2, or 4:

| Parameter | Case 9, H=1 | Case 10, H=2 | Case 1, H=4 |
| --- | ---: | ---: | ---: |
| CTAs per sequence | 1 | 1 | 1 |
| Warps per CTA | 8 | 8 | 8 |
| Linear M tile | 128 | 128 | 128 |
| Linear K tile | 64 | 64 | 64 |
| Attention M tile | 128 | 128 | 128 |
| Attention K tile | 32 | 64 | 32 |
| Pipeline stages | 3 | 3 | 3 |
| LayerNorm M tile | 64 | 64 | 64 |

The schedule is confined to B=64.  Cases 5/6 retain their previous four-warp,
M64, attention-M64 defaults.  Environment tuning switches still override the
shape defaults.

The PyTorch adapter and weight packer now accept `H in {1, 2, 4}`.  The
experimental multi-CTA path and CUDA debug capture intentionally remain H=4
only.

## Roofline method

`benchmarks/benchmark_step_3.py` owns the correctness and performance harness.  It uses a
`triton.testing.Benchmark` descriptor and `triton.testing.do_bench`; its text
report does not require the optional matplotlib package.

Useful reference-model work is:

```text
L * (8*B*S*D^2 + 4*B*S*D*F + 2*B*S*(S+1)*D)
  = 7.524581 GFLOP
```

The kernel's scratch slots are global tensors, not on-chip resident storage.
Each layer therefore transfers 22 complete activation tensors: LayerNorm1
(2), QKV (6), attention (4), output projection/residual (3), LayerNorm2 (2),
FFN input (2), and FFN output/residual (3).  Final LayerNorm adds two more.
For B=64 this is 188.743680 MB of logical activation traffic.  Adding one
797184-byte packed-weight footprint gives 189.540864 MB per call and 39.70
FLOP/byte.

On the H200 NVL, the benchmark uses 835.5 dense FP16 TFLOP/s (half the
vendor's sparsity-qualified 1671 TFLOP/s) and 4.8 TB/s HBM bandwidth.  The
ridge point is about 174.1 FLOP/byte, so **memory bandwidth is the binding
roofline**.

The byte count is logical global traffic: unique per-sequence workspace data
must pass through the memory hierarchy, while the small shared weight tensor
is counted once because it is cache-reused across CTAs.  Hardware counters are
unavailable, so the percentage is an effective-bandwidth roofline rather than
a measured DRAM-byte counter.

## Accuracy

Job 770359 tested three trials apiece at padding ratios 0, 0.25, and 0.75 for
all three head counts.  All 28,311,552 output values were bit-exact against the
PyTorch reference.  The gate is absolute error <= 0.001 OR relative error <=
0.01.

The layerwise Torch SDPA comparison in job 770387 did not pass that gate.  It
failed 358, 343, and 318 of 1,048,576 values for H=1, H=2, and H=4,
respectively, with maximum absolute error 0.005859375.  Its timing remains a
useful optimized-Torch performance reference, but not a valid submission path
under the strict gate.

## Performance

Authoritative SDPA/roofline `triton.testing` run: job 770387 on NVIDIA H200
NVL, CUDA 12.9.86, PyTorch 2.13.0+cu129, and Triton 3.7.1.  The more extensive
padded megakernel accuracy run remains job 770359.

| Case | Heads | Explicit Torch | Torch SDPA | Final one-CTA | vs. SDPA | vs. explicit | Effective BW | Memory roofline |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 1 | 0.662784 ms | 0.450912 ms | 0.158464 ms | 2.846x | 4.183x | 1.196 TB/s | 24.92% |
| 10 | 2 | 0.779664 ms | 0.533504 ms | 0.162848 ms | 3.276x | 4.788x | 1.164 TB/s | 24.25% |
| 1 | 4 | 0.906304 ms | 0.532256 ms | 0.185184 ms | 2.874x | 4.894x | 1.024 TB/s | 21.32% |

Relative to the first generalized M64/W4 baseline from job 770305, the tuned
kernel improved by 1.45x for H=1, 1.58x for H=2, and 1.38x for H=4.

Case-5 regression job 770360 remained bit-exact and measured 0.2376 ms versus
the prior 0.2375 ms result, confirming that B=128 retains its old schedule and
performance.

Only 64 independent CTAs are available on a 132-SM H200.  This limits the
fraction of full-device bandwidth that a one-CTA-per-sequence schedule can
exercise.  Nevertheless, adding a second Triton CTA was measured to cost more
than the additional occupancy returned.

## Rejected alternatives

| Alternative | Result |
| --- | --- |
| Eight warps with M64 | Helped H=1/H=2, regressed H=4; inferior to M128+W8. |
| M128 with four warps | Regressed all cases. |
| Causal upper-right tile pruning | Regressed all cases despite fewer useful FLOPs. |
| Pipeline stages 2 or 4 | Slower for every head count; stage 3 wins. |
| Pipeline stage 5 | Exceeded shared-memory capacity for H=2. |
| Linear K128 | Failed 28--40 outputs per million. |
| Attention K64 for H=1 | Slower; H=2 benefits from it. |
| Attention K128 for H=1 | Required 256 KiB shared memory, above the 232,448-byte limit. |
| Reciprocal-only softmax | Failed 1--4 outputs per million. |
| Hastings approximate exact-GELU replacement | Failed 2--3 outputs per million. |
| Simpler mean/variance LayerNorm | Failed 16--34 outputs per million. |
| LayerNorm M128 | Regressed all cases, especially H=1. |
| Sixteen warps | Regressed all cases. |

## Reproduction

```bash
sbatch --gres=gpu:h200-141:1 --partition=gpu --nodes=1 \
  job-scripts/run_step_3_h200.sh \
  --accuracy-trials 3 --padding-ratios 0 0.25 0.75
```

The job script fails immediately unless CUDA 12.9 exists at
`/usr/local/cuda-12.9`.
