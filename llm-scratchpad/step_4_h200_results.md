# Benchmarking step 4: cases 2, 3, 4, and 12 on H200

Date: 2026-08-29 (Asia/Singapore)

## Decision

Step 4 uses a single-launch, static completion DAG with **no device-wide
computation barrier**.  Dependencies are scoped to one independent sequence
or one pair of S32 sequences.  Sequence groups can therefore be in QKV,
attention, FFN, or different transformer layers at the same time.

The first phase-barrier baseline was discarded before production.  A later
global ready-queue implementation proved correctness and overlap, but atomics
for hundreds of M64/N64 tasks limited it to roughly 0.5% of the logical HBM
roofline.  The final static schedule retains the useful dependency overlap
without queue traffic.

## Final schedules

### Cases 2--4: S128

Each sequence owns three eight-warp CTAs, one for each of Q, K, and V.  Linear
tiles are M128/N128/K64 and attention uses M128/K128 query/key tiles with a
K32 head-dimension reduction.

1. The Q role computes two M64 LayerNorm row groups and publishes normalized
   input to the K/V roles.
2. Q, K, and V projections execute concurrently as three M128/N128 WGMMA
   tasks.
3. After the sequence-local QKV counter completes, the K and V roles each
   compute two attention heads concurrently.
4. Those same roles each execute one independent 64-row output-projection,
   LayerNorm2, FFN-in, FFN-out, and (on the last layer) final-norm tail.
5. The Q role releases the next layer only after both row tails complete.

There are 3, 12, and 48 CTAs for cases 2, 3, and 4.  Synchronization is only
among the three CTAs belonging to the same sequence.  Other sequences drift
freely through the DAG.

### Case 12: S32

Two independent sequences are paired into one M64 linear row.  Each pair owns
three four-warp CTAs for concurrent Q/K/V M64/N128/K64 projection.  The Q and
K roles then compute the two sequences' attention concurrently; the Q role
joins them for the paired M64 output/FFN tail.  There are 32 pairs and 96 CTAs.

PyTorch's S32 strided-batched PV GEMM exposes a two-fragment K16 accumulation
boundary.  A fused K32 WGMMA differed by rare FP16 ULPs that compounded at
padded, low input scales.  The production S32 PV explicitly computes two K16
dot products and adds them in FP32.  The isolated checkpoint and final model
are then bit-exact.

Both schedules use four global FP16 activation slots (norm/context, Q/residual,
K, and V/FFN).  Cross-CTA stores use release/acquire atomics plus explicit GPU
memory fences; operations chained inside one CTA use local barriers.  A
host-side epoch value lets each sequence role initialize its own scheduler
state inside the megakernel, avoiding a separate scheduler-memset launch.

## H200 tuning

| Parameter | Cases 2--4 | Case 12 |
| --- | ---: | ---: |
| CTAs per sequence/group | 3 | 3 per sequence pair |
| Total CTAs | 3 / 12 / 48 | 96 |
| Warps per CTA | 8 | 4 |
| Pipeline stages | 3 | 3 |
| Linear M/N/K | 128 / 128 / 64 | 64 / 128 / 64 |
| Attention query rows | 128 | 32 |
| Head-dimension K | 32 | 32 |
| PV accumulation | fused K128 | two K16 fragments |

## Roofline method

The benchmark uses `triton.testing.Benchmark` and `triton.testing.do_bench`.
Hardware counters are unavailable on the node.

Logical bytes count 22 complete activation transfers per layer, two for final
LayerNorm, and one packed FP16 weight footprint.  The resulting arithmetic
intensities are 31.38, 37.34, 39.20, and 35.01 FLOP/byte for cases 2, 3, 4,
and 12.  All are below the H200 NVL ridge point of about 174 FLOP/byte, so the
logical roofline is memory-bound.  The low-batch cases remain latency and
parallelism limited well below that line.

The H200 NVL reference peaks are 835.5 dense FP16 TFLOP/s and 4.8 TB/s HBM.
Percentages below are effective logical-bandwidth percentages, not measured
DRAM-counter values.

## Correctness

Authoritative job 771033 tested two trials for every combination of:

- cases 2, 3, 4, and 12;
- padding ratios 0, 0.25, and 0.75;
- input scales 1e-4, 1e-3, 3e-3, 1e-2, 0.1, 1, 10, 100, and 1000.

All **32,735,232 output values were bit-exact** against the explicit PyTorch
reference.  The gate was absolute error <= 0.001 OR relative error <= 0.01.

The S32 attention diagnostic in job 771026 separately verified exact scores,
softmax probabilities, and PV context after selecting the K16-fragment PV
path.

Shared-helper regression job 771036 reran case 1 with all-valid and padded
masks.  The earlier step-3 path remained bit-exact and retained its 0.185 ms
latency.

## Performance

Authoritative job 771033 used 100 ms warmup and 500 ms repetition windows per
provider through `triton.testing.do_bench` on NVIDIA H200 NVL, CUDA 12.9.86,
PyTorch 2.13.0+cu129, and Triton 3.7.1.

| Case | Torch explicit | Torch SDPA | Final DAG | vs. SDPA | vs. explicit | Effective BW | Memory roofline |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.7339 ms | 0.4572 ms | 0.127424 ms | 3.588x | 5.760x | 0.029 TB/s | 0.61% |
| 3 | 0.7393 ms | 0.4604 ms | 0.124224 ms | 3.706x | 5.951x | 0.101 TB/s | 2.11% |
| 4 | 0.7372 ms | 0.4550 ms | 0.123552 ms | 3.683x | 5.966x | 0.388 TB/s | 8.09% |
| 12 | 0.7335 ms | 0.4552 ms | 0.113408 ms | 4.014x | 6.468x | 0.423 TB/s | 8.81% |

## Optimization history and rejected alternatives

| Alternative | H200 result |
| --- | --- |
| Device-wide phase barriers | Kept only as the initial structural baseline; not a production approach. |
| Global ready queue, M64/N64 tasks | Exact, but 0.218/0.508/1.950/2.292 ms for cases 2/3/4/12 due to queue overhead. |
| Coarsened global queue | Improved cases 4/12 to 0.519/0.489 ms, still scheduler-bound. |
| Static M64 S128 schedule | Exact at about 0.247/0.241/0.239 ms. |
| M128 S128, serial attention/tail | Improved to about 0.181/0.178/0.176 ms. |
| Parallel attention heads | Improved cases 2--4 to about 0.154/0.149/0.152 ms. |
| Parallel 64-row tails | Final S128 path, about 0.129/0.126/0.125 ms before the epoch cleanup. |
| S32 attention on one CTA | Exact PV issue remained and latency was 0.115 ms; splitting two sequences improves all-valid latency. |
| Fused K32 S32 PV | Failed a few padded low-scale outputs after four layers; explicit two-K16 PV is exact. |
| Eight warps for S32 | Regressed from about 0.121 to 0.127 ms before later optimizations. |
| Four or sixteen warps for S128 | Regressed to about 0.290 and 0.214 ms. |
| Pipeline stages 2 or 4 | Regressed relative to stage 3. |
| Linear K32 | Regressed all cases; K64 wins. |
| Attention K16 | Regressed all cases; K32 wins. |

## Reproduction

```bash
sbatch --gres=gpu:h200-141:1 job-scripts/run_step_4_h200.sh \
  --accuracy-trials 2 \
  --padding-ratios 0 0.25 0.75 \
  --scales 0.0001 0.001 0.003 0.01 0.1 1 10 100 1000
```

The job script fails immediately unless CUDA 12.9 exists at
`/usr/local/cuda-12.9`.
