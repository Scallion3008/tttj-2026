# H100 hardware-counter optimization results

Date: 2026-08-30 (Asia/Singapore)

## Outcome

All nine working cases were profiled with Nsight Compute 2026.2.1 on an
NVIDIA H100 NVL 96 GB (132 SMs), using CUDA 12.9.86.  The final paths remain
single-launch megakernels.  Every cross-CTA dependency is scoped to one
sequence (or one pair of S32 sequences); no grid-wide computation barrier was
introduced.

The two production families are now:

- cases 5/6/9: one resident CTA per sequence;
- cases 1/10: a resident CTA plus one attention/projection helper per sequence;
- cases 2/3/4: four-role static sequence DAG;
- case 12: four-role static two-sequence DAG with split attention and tails.

## Final latency and fastest PyTorch SDPA baseline

`benchmarks/benchmark_sdpa_backends.py` measured auto-selection and forced cuDNN, Flash,
memory-efficient, and math SDPA.  Memory-efficient attention was fastest for
all cases except case 6, where auto-selection won.  cuDNN 9.24 was available
but was not the fastest backend for these shapes.

| Case | Final megakernel | Fastest SDPA | Backend | Speedup |
| ---: | ---: | ---: | --- | ---: |
| 1 | 0.167776 ms | 0.853184 ms | efficient | 5.085x |
| 2 | 0.110592 ms | 0.879488 ms | efficient | 7.953x |
| 3 | 0.107328 ms | 0.873600 ms | efficient | 8.140x |
| 4 | 0.108768 ms | 0.878784 ms | efficient | 8.079x |
| 5 | 0.263936 ms | 0.865536 ms | efficient | 3.279x |
| 6 | 13.453952 ms | 41.811073 ms | auto | 3.108x |
| 9 | 0.153152 ms | 0.782080 ms | efficient | 5.107x |
| 10 | 0.154944 ms | 0.885040 ms | efficient | 5.712x |
| 12 | 0.103040 ms | 0.873344 ms | efficient | 8.476x |

The case-6 timing is clock-sensitive.  In a paired stage experiment on one
node, stage 2 improved 13.644432 to 13.435120 ms (1.016x); Nsight duration
improved from 12.77 to 12.27 ms while dynamic shared memory fell from 73.73 to
49.15 KiB.

## Final Nsight Compute profile

Reports are `job-scripts/outputs/ncu_h100_case*_771916.ncu-rep`; text exports use the
same names with `.txt`.  All percentages are Nsight's peak-normalized values.

| Case | Duration | Grid | SM compute | Memory | DRAM | Achieved occupancy | No eligible |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 262.82 us | 128 | 26.87% | 25.67% | 0.53% | 6.28% | 71.72% |
| 6 | 12.27 ms | 10,000 | 44.37% | 39.00% | 9.58% | 12.65% | 55.51% |
| 1 | 178.56 us | 128 | 20.72% | 18.21% | 0.43% | 12.36% | 78.13% |
| 9 | 166.56 us | 64 | 12.37% | 16.41% | 0.46% | 12.58% | 74.15% |
| 10 | 166.82 us | 128 | 16.80% | 19.57% | 0.50% | 12.58% | 82.26% |
| 2 | 120.93 us | 4 | 0.49% | 0.49% | 0.20% | 12.53% | 82.47% |
| 3 | 118.59 us | 16 | 2.10% | 2.07% | 0.23% | 12.42% | 82.14% |
| 4 | 119.30 us | 64 | 8.55% | 8.43% | 0.31% | 12.34% | 82.23% |
| 12 | 106.75 us | 128 | 8.75% | 14.15% | 0.35% | 6.27% | 90.74% |

Every kernel has a 97--99% L2 hit rate except the tiny case 2 (73.6%).  DRAM
never exceeds 9.6% of peak.  Thus the low-batch paths are latency/dependency
limited, while case 6 is the only path with substantial sustained compute and
memory utilization.  All paths still use 255 registers/thread.  Cases
1/2/3/4/9/10 use eight warps; cases 5/6/12 use four.

Relative to the initial profiles, Nsight duration improved by 8.6% for case 1,
4.1% for case 10, 6.7--7.5% for cases 2--4, 13.9% for case 12, and 3.9% for
case 6.  Cases 5/9 were intentionally left on their original schedules.

## Accepted algorithmic changes

### Cases 1 and 10

The original B64 kernel launched only 64 blocks.  A second CTA now assists each
sequence, producing K concurrently with the resident role's V projection and
then computing disjoint attention heads.  Case 10 also splits the non-aliased
Q projection into two N64 halves.  A per-sequence epoch protocol releases only
Q/K/V, attention, and layer-local successors.  The resident role alone owns
the output projection and FFN tail, so sequences can occupy different phases
and layers simultaneously.

Case 9 retains one CTA: splitting its single HD128 attention head into M64
query halves cost more than the added parallelism saved.

### Cases 2--4

The static DAG now uses four CTAs per sequence.  The fourth role produces half
of LayerNorm1 and half of Q, then owns one attention head.  The other roles own
the remaining Q/K/V work and attention heads; two roles continue into the two
independent M64 tails.  This grows the grids from 3/12/48 to 4/16/64 CTAs and
shortens both LayerNorm and Q critical paths.

### Case 12

Each pair of S32 sequences now uses four rather than three roles.  Attention
heads 0--1 and 2--3 are split between roles for each sequence, and two roles
run the two S32 output/FFN/final-norm tails concurrently.  The grid grows from
96 to 128 CTAs.

### Case 6

Two Triton pipeline stages reduce shared-memory pressure and improve the
steady-state large-batch path.  All smaller resident shapes retain stage 3.

## Correctness

- Cases 1/9/10: two trials at every combination of padding 0/0.25/0.75 and
  scale 1e-4, 1e-3, 3e-3, 1e-2, 0.1, 1, 10, 100, and 1000.  All 169,869,312
  values were bit-exact and every repeated launch was exact.
- Cases 2/3/4/12: the same matrix, totaling 32,735,232 values.  Every value and
  repeated launch was bit-exact.
- Case 5: five trials at each of three padding ratios, all bit-exact and
  deterministic.
- Case 6: two trials at each padding ratio, all within the strict gate and
  deterministic.  One padded trial had max absolute error 0.001953125 but no
  gate failures.

## Rejected experiments

- Removing the GPU memory fence or its post-fence CTA barrier made the
  four-role case-12 path nondeterministic.  The safe barrier / `membar.gl` /
  barrier publication remains.
- Replacing acquire atomics with inline acquire loads was neutral or slower.
- Fusing final LayerNorm into FFN-out was neutral/slower.
- Dispatching B64 through the full static DAG regressed cases 1/9/10 because
  projection and tail hand-offs outweighed added parallelism.
- Register caps (168, 160, 128), eight warps for cases 5/6, B64 resident M64,
  and stages 4 all regressed.
- Four roles without split LayerNorm/Q did not help S128; the combined schedule
  is required.
- Splitting the single H1 attention head was slower; splitting Q was useful
  only for H2.

## Reproduction

```bash
sbatch job-scripts/benchmark_megakernels_h100.sh
sbatch job-scripts/benchmark_sdpa_backends_h100.sh
sbatch job-scripts/profile_megakernels_h100.sh
```

Every wrapper requests `--gres=gpu:h100-96:1` and fails unless CUDA 12.9 is
installed under `/usr/local/cuda-12.9`.
