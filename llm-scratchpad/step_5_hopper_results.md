# Step 5: head-dimension-8 megakernel results

Date: 2026-08-30 (Asia/Singapore)

## Outcome

Cases 7 and 11 now use a single-launch, two-role sequence-resident megakernel.
The final production path uses padded K16 tensor-core QK and padded N16 PV.
It does not contain a grid-wide barrier or device-wide synchronization.

The fixed grid has 128 CTAs: two roles for each of the 64 independent
sequences. Within each sequence, the roles overlap/split:

- the two 64-row halves of LayerNorm1;
- K and V production, followed by the two non-aliased halves of Q;
- half of the attention heads per role;
- the two 64-row halves of output projection, LayerNorm2, both FFN GEMMs, and
  final LayerNorm.

Dependencies are published only through per-sequence release/acquire epochs.
Different sequences and layers can therefore remain in different phases; no
batch-wide phase boundary was introduced.

## H200 NVL tuning proxy

The H100-96 pool was saturated after the first full-device run, so final
tuning used the available 132-SM H200 NVL, as permitted by `AGENTS.md`. CUDA
12.9.86, Torch 2.13.0+cu129, and Triton 3.7.1 were used. Hardware counters are
unavailable on that node, so these are 500-repetition Triton medians.

| Case | Megakernel | Fastest available SDPA | Backend | Speedup |
| ---: | ---: | ---: | --- | ---: |
| 7 | 0.094400 ms | 0.484608 ms | efficient | 5.134x |
| 11 | 0.219872 ms | 0.597248 ms | Flash | 2.716x |

These are paired measurements from job 772118. A separate exhaustive SDPA
backend sweep measured efficient SDPA for case 7 as low as 0.430560 ms, which
still gives a conservative 4.561x megakernel speedup. Flash was the fastest
case-11 backend in both runs.

The useful-work model gives 14.25 FLOP/B for case 7 and 39.70 FLOP/B for case
11, so both bind on HBM rather than dense FP16 tensor throughput. Against the
H200 NVL 4.8 TB/s memory roofline, the final medians reach 10.42% and 17.96%
of the corresponding useful-work roofline. These percentages are model-based
because Nsight hardware counters cannot be collected on this node.

The last available full H100 NVL run, before splitting LayerNorm1, Q, and the
tail across both roles, measured 0.097344 ms and 0.254528 ms. On H200, adding
those overlaps improved the comparable serial-tail schedule from 0.098112 to
0.094400 ms for case 7 (1.039x) and from 0.256208 to 0.219872 ms for case 11
(1.165x).

## SIMT versus padded tensor-core QK

The requested comparison strongly favors padding the reduction from K8 to
K16 and using tensor cores:

| Case | Padded tensor-core QK | True SIMT K8 QK | SIMT slowdown |
| ---: | ---: | ---: | ---: |
| 7 | 0.094400 ms | 0.517056 ms | 5.477x |
| 11 | 0.219872 ms | 1.883360 ms | 8.565x |

The SIMT implementation also failed the strict gate at ordinary scale: 164 of
262,144 outputs failed for case 7 and 339 of 1,048,576 failed for case 11.
The padded tensor-core path is both faster and numerically preferable.

## Correctness

- H100 NVL MIG job 772072 ran two trials at every combination of padding
  0/0.25/0.75 and scale 1e-4, 1e-3, 3e-3, 1e-2, 0.1, 1, 10, 100, and 1000.
  All 70,778,880 checked values were bit-exact, and repeat differences were
  zero.
- H200 NVL job 772089 repeated the complete matrix for 35,389,440 values. All
  values and repeated launches were bit-exact.
- A native N8 PV tensor-core result was slightly faster in early experiments,
  but failed padded low-scale cases. Production therefore keeps the padded
  N16 PV boundary.

## Final tuning and rejected experiments

Both cases use eight warps, three stages, M128 linear/attention tiles, K16 QK,
and two roles. Case 7 uses K16 linear reductions to match the reference's two
K16 D32 accumulation fragments; case 11 uses K64 linear reductions.

- Four roles with two internal sequence waves regressed case 7 to 0.129760 ms
  and case 11 to 0.281728 ms, so that alternate implementation was removed.
- Four warps was effectively tied for case 7 (0.094592 ms) but regressed case
  11 to 0.335456 ms. Eight warps also won the earlier H100 comparison and is
  retained for both fixed shapes.
- Two stages regressed case 11 to 0.237088 ms; K32 linear reductions regressed
  it to 0.229248 ms.
- M64 linear/attention tiles, causal-tile skipping, serial LayerNorm/tails,
  native N8 PV, approximate softmax/exp, and unsplit Q were all slower or
  failed the numerical gate.

## Reproduction

```bash
# H100-96 target
sbatch job-scripts/run_step_5_h100.sh

# H200-141 fallback/proxy
sbatch --gres=gpu:h200-141:1 job-scripts/run_step_5_h100.sh --allow-h200

# Required QK alternative
sbatch --gres=gpu:h200-141:1 \
  --export=ALL,TTTJ_HD8_QK_MODE=0 \
  job-scripts/run_step_5_h100.sh --allow-h200
```

The wrapper fails unless `/usr/local/cuda-12.9` is present.
