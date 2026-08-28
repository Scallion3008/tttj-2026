# H200 multi-CTA cluster investigation

## Scope

This note records the two-CTA experiment for the fused `S=D=F=128`, `H=L=4`
transformer megakernel. The production default remains one CTA; set
`TTTJ_NUM_CTAS=2` to compile the clustered path.

The measurements below used an NVIDIA H200 NVL (`sm_90`, 132 SMs), CUDA
12.9.86, PyTorch 2.13.0+cu129, and Triton 3.7.1 on `xgpk0`.

## What blocked the original path

Several independent Triton 3.7 limitations had to be worked around:

1. Rank-changing `tl.split` reduction trees failed multi-CTA layout lowering
   with `LLVM ERROR: Invalid bases passed to LinearLayout`. LayerNorm now uses
   a split-free half4-per-lane Welford tree and clustered softmax uses an
   explicit lane tree. Both reproduce the PyTorch reduction order exactly.
2. PlanCTA asserted in `markTiled` when one IR function contained multiple
   `tt.dot` operations. Clustered linears and the three attention phases are
   isolated in no-inline helpers, with at most one dot per helper.
3. Splitting a QK tile along the key axis required an unsupported cross-CTA
   reduction in softmax. QK is instead tiled as `128x64`, which makes PlanCTA
   split complete query rows between the CTAs. Scores/probabilities use one
   materialized 128x128 workspace slot between no-inline QK, softmax, and PV
   helpers.
4. Triton's one-CTA buffer aliases are not generally legal after PlanCTA
   repartitions a stage. In particular, the Q projection wrote into its own
   normalized input. At B=10,000 this produced nondeterministic corruption
   beginning at row 64 even with a software global-memory barrier. Giving Q a
   distinct live location fixed the race. The final LayerNorm was similarly
   changed so clustered input and output are distinct.

The fourth item was the runtime correctness blocker. Native Hopper
`barrier.cluster.arrive.release` / `barrier.cluster.wait.acquire` is reliable
once the intra-stage alias hazards are removed; an experimental release-atomic
software barrier did not fix the aliased dot and was removed.

## Final four-slot schedule

The corrected cluster schedule avoids a fifth per-sequence workspace slot:

- LayerNorm preserves the residual in the fourth workspace slot while it has
  the source values loaded.
- Q is written to the output buffer, separate from normalized input.
- The normalized-input slot becomes the materialized attention score and
  probability scratch after Q/K/V complete.
- Attention context overwrites Q one head at a time only after that head's QK
  phase has completed.
- The output projection reads context and the preserved residual.
- The last FFN residual is placed in a dead workspace slot and final LayerNorm
  writes directly to the output, keeping its input/output distinct.

The whole transformer is still one Triton launch and does not call Torch SDPA.

## Accuracy

The final split-free math is bit-exact against the project PyTorch reference:

- Job 769236: all 15 B=128 stress trials passed for padding ratios 0, 0.25,
  and 0.75; every repeated launch was exact.
- Job 769232: B=10,000 passed all 163,840,000 output elements and the repeated
  launch was exact.
- Job 769238: the final stage diagnostic matched all 37 transformer stages
  plus score, exponential, denominator, and probability traces exactly.

## H200 performance

Final full-benchmark results (cluster job 769234; production job 768869):

| Path | Case 5, B=128 | Case 6, B=10,000 | vs. Torch case 5 | vs. Torch case 6 |
| --- | ---: | ---: | ---: | ---: |
| Torch baseline | 1.3841 ms | 85.8897 ms | 1.000x | 1.000x |
| Two-CTA cluster | 0.3736 ms | 23.8459 ms | 3.705x | 3.602x |
| One-CTA production | 0.2375 ms | 12.2233 ms | 5.733x | 7.028x |

The clustered path is therefore about 1.57x slower than one CTA in case 5 and
1.95x slower in case 6, despite comfortably beating Torch.

Useful negative/positive experiments:

- The first corrected five-slot path was 0.4299 / 27.7601 ms.
- Reusing four slots, preserving the residual during LayerNorm, and eliminating
  the final copy improved the path to 0.3736 / 23.8459 ms.
- M128 clustered linears regressed to 0.4527 / 38.7323 ms. PlanCTA partitions
  result ownership but does not turn the two CTA programs into one cooperative
  WGMMA; enlarging M caused more duplicated tensor-core work.
- `num_stages=2`, `num_stages=4`, and linear K32 gave case-6 medians of
  24.5683, 25.0078, and 25.4054 ms. The K64/stages3 defaults win.
- Eight warps improved the noisy case-5 quick median to 0.3843 ms but regressed
  case 6 to 27.2659 ms, so four warps remains the joint default.

## Profiling result and remaining bottleneck

Nsight Compute 2026.2.1 at `/usr/local/cuda-12.9/ncu` was attempted for both
paths in jobs 769205--769208. The H200 driver denies performance-counter access
to this Slurm user with `ERR_NVGPUCTRPERM`, including the LaunchStats-only
attempt, so no replay-based hardware counters could be collected.

Static cubin/compiler data still exposes the cost of Triton's clustered
lowering:

| Resource | One CTA | Two CTAs |
| --- | ---: | ---: |
| Registers per thread | 255 | 255 |
| Dynamic shared memory | 24,576 B | 65,536 B |
| Stack frame | 0 B | 1,280 B |

The cluster path needs no-inline calls to evade PlanCTA's multiple-dot
assertion, materializes attention scores in global memory, executes many
cluster barriers, and retains PlanCTA's duplicated dot work. Those costs
explain why it cannot beat the one-CTA kernel for cases 5 and 6. A competitive
cluster implementation would require explicit CTA ownership in CuTe/CUTLASS
or raw CUDA rather than Triton 3.7 PlanCTA. Since B=128 already fills 128 of
the H200's 132 SMs with independent one-CTA sequences, two-CTA clustering also
has no occupancy deficit to solve in these cases.
