# Initial H100 FP16 megakernel notes

These notes analyze the 14 fixed cases in the [project README](../README.md)
under the following assumptions:

- The target is one NVIDIA H100 GPU.
- Inputs and weights are FP16. Reductions and tensor-core accumulators use FP32
  where needed for correctness.
- Each case is run independently, with warmed weights, and steady-state forward
  latency is the quantity being optimized.
- Attention is exact, dense, causal self-attention. No sparse or approximate
  attention is allowed.
- The implementation may dispatch to multiple shape-specialized kernels. A
  "megakernel" does not mean that all 14 cases must use one launch
  configuration or one static schedule.

## Executive recommendation

| Cases | Recommendation | Primary reason |
| --- | --- | --- |
| 1--7, 9--12 | End-to-end megakernel | Launch cost and/or cross-operation activation traffic can be removed while preserving useful tile parallelism. |
| 8 | Layerwise hybrid | Large D=1024 GEMMs dominate and a sequence cannot reside in one CTA. |
| 13 | Conditional hybrid | Streaming attention dominates enough that matching a tuned Hopper attention kernel matters more than eliminating launches. |
| 14 | Standalone long-context path | The work is overwhelmingly quadratic attention; model-wide launch overhead is immaterial. |

The most important distinction is whether a sequence can remain resident. H100
supports up to 227 KiB of shared memory in one thread block. For S=128 and
D=128, `x` occupies 32 KiB and QKV occupies 96 KiB, leaving approximately 99
KiB for aliased scratch and staged tiles. This is tight but feasible. For
S=128, D=1024 or S=1024, D=128, `x` alone occupies 256 KiB, so a single-CTA
sequence-resident design is impossible.

Hopper's Tensor Memory Accelerator (TMA), warp specialization, thread-block
clusters, and distributed shared memory (DSM) are useful for these schedules.
The resource limits and mechanisms are described in the
[NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html).

## Why FP16 is the primary path

FP16 has 11 bits of significand precision, including the implicit leading bit,
versus 8 bits for BF16. Its machine epsilon is `2^-10`, compared with `2^-7`
for BF16, so FP16 has eight times finer representational spacing at the same
magnitude. NVIDIA advertises the same peak FP16 and BF16 Tensor Core throughput
on H100, making FP16 the better initial choice for this strict numerical gate:

- [CUDA floating-point format properties](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
- [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/)

For the README gate, an output near 1 has an allowed relative error of 0.01.
That is only about 1.28 BF16 ULPs but about 10.24 FP16 ULPs. At the
absolute/relative crossover `|reference| = 0.1`, the 0.001 absolute allowance
is about 2 BF16 ULPs but about 16 FP16 ULPs. FP16 therefore gives substantially
more tolerance for differences caused by reassociation, fusion, and rounding.
It does not make an approximate GELU, incorrect softmax, or missing mask
acceptable.

The tradeoff is dynamic range. FP16's maximum finite value is 65,504 and its
smallest normal value is `2^-14`, whereas BF16 has approximately the exponent
range of FP32. All GEMM accumulation, LayerNorm statistics, softmax state, and
other reductions must remain FP32. Accuracy tests must include input scales
large enough to expose FP16 overflow and small enough to expose underflow or
LayerNorm-epsilon behavior.

## Cost model

Ignoring LayerNorm, GELU, masking, and softmax scalar work, one transformer
layer performs approximately

```text
Q/K/V/out projections:  8 * B * S * D^2
two FFN projections:    4 * B * S * D * F
causal QK and PV:        2 * B * S^2 * D
```

The exact attention estimate used below substitutes `S * (S + 1)` for `S^2`
to include the causal diagonal. Multiply the sum by the number of layers.
These figures are comparative estimates, not predicted runtimes.

| Case | Tokens `B*S` | One `x` buffer | Approx. total work | Attention share | Head dim |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8,192 | 2.000 MiB | 7.525 GFLOP | 14.4% | 32 |
| 2 | 128 | 0.031 MiB | 0.118 GFLOP | 14.4% | 32 |
| 3 | 512 | 0.125 MiB | 0.470 GFLOP | 14.4% | 32 |
| 4 | 2,048 | 0.500 MiB | 1.881 GFLOP | 14.4% | 32 |
| 5 | 16,384 | 4.000 MiB | 15.049 GFLOP | 14.4% | 32 |
| 6 | 1,280,000 | 312.500 MiB | 1,175.716 GFLOP | 14.4% | 32 |
| 7 | 8,192 | 0.500 MiB | 0.673 GFLOP | 40.2% | 8 |
| 8 | 8,192 | 16.000 MiB | 420.974 GFLOP | 2.1% | 256 |
| 9 | 8,192 | 2.000 MiB | 7.525 GFLOP | 14.4% | 128 |
| 10 | 8,192 | 2.000 MiB | 7.525 GFLOP | 14.4% | 64 |
| 11 | 8,192 | 2.000 MiB | 7.525 GFLOP | 14.4% | 8 |
| 12 | 2,048 | 0.500 MiB | 1.680 GFLOP | 4.1% | 32 |
| 13 | 65,536 | 16.000 MiB | 120.326 GFLOP | 57.2% | 32 |
| 14 | 3,200,000 | 6,250.000 MiB | 1.391 PFLOP | 94.2% | 64 |

Changing the head count at fixed D does not materially change the arithmetic
count, but it changes task granularity, tensor-core utilization, and the number
of independently schedulable attention jobs.

## Scheduling families

### A. Fine-grained static DAG

Use an offline, shape-specialized task graph executed by a persistent set of
workers. A task becomes runnable as soon as its input tiles are ready:

```text
LayerNorm tile
  -> Q, K, and V projection tiles in parallel
  -> attention query tiles as their required K/V region becomes ready
  -> output projection + residual
  -> LayerNorm + FFN in/out + residual
  -> next layer
```

Readiness counters avoid forcing every worker through a full-grid barrier after
each logical operator. This schedule is intended for low-batch cases where one
CTA per sequence would leave most of the H100 idle.

### B. Sequence-resident CTA or cluster

Exploit the fact that batches do not interact. A worker owns a sequence, or a
two-CTA cluster owns two 64-row halves of one sequence, and runs it through all
layers before releasing its shared-memory allocation. The design should:

- retain the residual activation across projections and layers;
- compute attention with online FP32 softmax rather than materializing scores;
- alias normalized input, attention output, and FFN scratch aggressively;
- stream weight tiles with TMA and overlap them with WGMMA consumers;
- use `cluster.sync()` only at true cross-CTA dependencies, principally after
  producing K/V and before reusing shared-memory regions.

For a two-CTA S=128 cluster, each CTA owns 64 query rows. That is a natural
WGMMA M tile and cuts the local `x` and QKV footprint in half. Each CTA reads
the other half of K/V through DSM during attention.

### C. Layerwise hybrid

Use separate, highly tuned WGMMA and Hopper attention kernels, but fuse
bandwidth-bound epilogues and capture the fixed sequence of launches in a CUDA
Graph. Candidate fusions are:

- LayerNorm + QKV preparation;
- projection bias + output layout conversion;
- output projection + residual;
- LayerNorm + FFN input projection;
- exact GELU in the FFN epilogue or following mainloop;
- FFN output projection + residual + padding mask;
- final LayerNorm + padding mask.

This path is preferable when a standalone GEMM or attention kernel has enough
work to amortize launch overhead and an end-to-end kernel cannot retain its
working set.

### D. Long-context streaming attention

Assign work by `(batch, head, query_tile)`. Each task streams causal K/V tiles,
updates FP32 online-softmax state, and never writes an S-by-S score matrix.
Causal query tasks have unequal lengths, so issue later query blocks first
(longest-processing-time order) to reduce the tail. The official Hopper
FlashAttention implementation uses persistent scheduling and causal head/tile
swizzling patterns worth studying:

- [FlashAttention repository](https://github.com/Dao-AILab/flash-attention)
- [Hopper scheduling source](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_api.cpp)

## Detailed case analysis

### Case 1: B64, D128, H4, S128, L4, F128

**Suitability: strong end-to-end megakernel candidate.**

- The base per-sequence working set is only 32 KiB for `x` plus 96 KiB for
  QKV, so the model can stay resident with carefully aliased scratch.
- One CTA per sequence would launch only 64 CTAs, which leaves a substantial
  portion of an H100 unused. A two-CTA cluster per sequence produces 128 CTAs,
  close to a full wave on an H100 SXM and still portable to an H100 with fewer
  SMs through normal cluster scheduling.
- Splitting S=128 into two 64-query halves maps directly to WGMMA's useful M
  granularity. Each half performs its own output and FFN work while K/V is
  visible across the cluster.
- The complete activation is only 2 MiB, so launch elimination is important,
  but the stronger megakernel benefit is retaining each sequence through four
  layers rather than repeatedly reading and writing intermediates.

**Schedule:** two-CTA sequence cluster, local 64-row ownership, DSM K/V access,
and full-model execution before the cluster accepts another sequence.

### Case 2: B1, D128, H4, S128, L4, F128

**Suitability: strong for latency, despite unavoidable low utilization.**

- The entire model is only about 0.118 GFLOP, making a conventional chain of
  kernel launches disproportionately expensive.
- A sequence-resident design would occupy only one CTA, or two CTAs if
  clustered. It would remove launches but serialize too much tensor work.
- After LayerNorm, Q, K, and V projections are independent. Their M64/N64
  output tiles, followed by separate attention heads and query tiles, provide
  the principal available parallelism.
- Filling every SM with persistent workers would add queue polling and barrier
  overhead without creating more useful work. The worker count should be
  autotuned around the actual ready-task count, likely in the tens rather than
  the full device.

**Schedule:** small static-DAG grid, critical-path task ordering, concurrent
Q/K/V branches, and readiness counters instead of phase-wide grid barriers.

### Case 3: B4, D128, H4, S128, L4, F128

**Suitability: strong fine-grained DAG candidate.**

- At approximately 0.470 GFLOP, launch cost remains important.
- Four sequence-owned CTAs would be badly under-parallel. Treating the combined
  token dimension as M=512 creates eight M64 tiles; with two N64 tiles and
  three Q/K/V branches, those projections alone expose 48 tasks.
- Attention adds `B * H * 2 = 32` natural 64-query jobs, with more tasks from
  output and FFN projections. This is enough parallel work for a moderate
  persistent grid even though no single operator is large.
- Fine-grained dependencies allow early batches and query tiles to advance
  while the tail of Q/K/V production completes.

**Schedule:** 64x64 projection tasks in a static DAG, with batch/head/query
attention tasks released through readiness counters.

### Case 4: B16, D128, H4, S128, L4, F128

**Suitability: strong full-device DAG candidate.**

- Approximately 1.881 GFLOP supplies enough tile work for a device-sized
  persistent worker grid, while the baseline still consists of many small
  launches.
- An eight-CTA cluster per sequence could nominally create 128 CTAs, but each
  would own only 16 query rows. Padding those partitions to tensor-core tile
  sizes and synchronizing eight blocks would squander the apparent occupancy.
- Flattening batch and sequence for projection tiles provides regular M64/N64
  work. Attention then expands into independent batch/head/query jobs.
- The DAG should fuse or immediately chain bandwidth-bound work: LayerNorm into
  QKV production, projection epilogues into bias/layout handling, and residuals
  into output stores.

**Schedule:** approximately one persistent worker per SM consuming
operation-specific queues, with fine-grained readiness rather than large
sequence clusters.

### Case 5: B128, D128, H4, S128, L4, F128

**Suitability: excellent sequence-resident candidate.**

- B=128 is close to the physical SM count of common H100 variants, giving
  approximately one wave of independent sequences.
- A single sequence supplies M=128, so each linear operation decomposes cleanly
  into two M64 WGMMA tiles without cross-CTA communication.
- Retaining `x`, QKV, and aliased scratch avoids intermediate global traffic
  across all four layers. No batch-level synchronization is required.
- The shared-memory allocation will likely restrict occupancy to one CTA per
  SM. That is intended: the design seeks tensor-core and TMA pipeline
  utilization inside each CTA rather than multiple resident blocks.
- On an H100 with fewer than 128 SMs, a short second wave is acceptable and can
  be handled by the ordinary grid or a sequence counter.

**Schedule:** one CTA per sequence, producer/consumer warp specialization, and
full-transformer execution before storing the final normalized output.

### Case 6: B10000, D128, H4, S128, L4, F128

**Suitability: excellent persistent sequence-queue candidate.**

- At approximately 1.176 TFLOP, launch overhead is no longer the main issue.
  The opportunity is avoiding repeated materialization of a 312.5 MiB
  activation between every logical operator.
- Each of the 10,000 sequences still has the same compact, tensor-core-friendly
  128x128 working set as case 5.
- A fixed pool of resident CTAs provides natural load balancing. Each CTA
  atomically claims a sequence, runs all four layers, writes its result, and
  claims the next sequence.
- All tasks have the same nominal length, so a simple monotonic counter is
  sufficient; an elaborate priority queue would only add overhead.
- The D128/F128 weight set is small relative to the total work and is shared by
  all workers, allowing strong cache reuse. One queue atomic per roughly 0.118
  GFLOP sequence is negligible.
- The main risk is that a sequence-local GEMM body might underperform the
  equivalent very-large-M GEMM. Both paths must be benchmarked; the megakernel
  should be retained only if its 128x128 WGMMA tiles stay efficient.

**Schedule:** approximately one resident CTA per SM pulling whole-sequence jobs
from a global counter, with no phase-wide synchronization across sequences.

### Case 7: B64, D32, H4, S128, L4, F32

**Suitability: strong, but it needs a tiny-dimension implementation.**

- Total work is only about 0.673 GFLOP, and the full activation is 0.5 MiB, so
  framework and launch overhead are substantial.
- One sequence needs only 8 KiB for `x` and 24 KiB for QKV. Residency is easy;
  the difficulty is efficient arithmetic at D=32 and head dimension 8.
- LayerNorm over 32 features maps naturally to a warp reduction.
- QK has reduction dimension 8, below normal FP16 tensor-core K granularity.
  Padding to K=16 wastes half the arithmetic but may still beat scalar code;
  a warp-level SIMT dot product is a credible alternative and must be measured.
- One task per head is too small. Each CTA should bundle multiple heads and
  query tiles so scheduling and synchronization do not dominate.
- Two CTAs per sequence restore approximately 128-way block parallelism while
  keeping each CTA's query ownership at 64 rows.

**Schedule:** two-CTA sequence clusters, warp-level normalization, grouped head
tasks, and an autotuned padded-WGMMA-versus-SIMT attention path.

### Case 8: B64, D1024, H4, S128, L4, F1024

**Suitability: poor for an end-to-end megakernel; use a layerwise hybrid.**

- The case performs approximately 421 GFLOP, of which only about 2.1% is
  attention. Large D1024 projections and FFNs determine performance.
- One sequence's `x` is already 256 KiB, exceeding the H100 per-block
  shared-memory limit before allocating QKV or GEMM staging buffers.
- A cluster could distribute the activation, but that does not eliminate the
  need for highly tuned, large WGMMA GEMMs. Remote DSM traffic and general task
  dispatch risk losing more than launch fusion can save.
- The flattened projection shape has M=8192 and K/N=1024, enough work for
  standalone GEMMs to amortize launches and approach tensor-core throughput.
- Attention should use a Hopper implementation specialized for head dimension
  256; it is too small a fraction of total time to justify redesigning the
  entire model around it.

**Schedule:** tuned layerwise WGMMA GEMMs with fused norm, bias, GELU, and
residual epilogues; Hopper FlashAttention for attention; CUDA Graph replay for
launch orchestration.

### Case 9: B64, D128, H1, S128, L4, F128

**Suitability: strong clustered sequence candidate.**

- The total arithmetic and sequence working set match case 1. Only attention's
  task decomposition changes.
- A single head reduces head-level parallelism, but two 64-query tiles per
  sequence still create 128 attention jobs across B=64.
- Head dimension 128 is favorable for tensor-core QK/PV operations and makes
  each attention task substantial enough to amortize its setup.
- Each CTA in a two-block cluster owns one query half and reads the complete K/V
  through local plus remote DSM.

**Schedule:** two-CTA sequence clusters with one head and 64 query rows per CTA,
using online FP32 softmax over both K/V halves.

### Case 10: B64, D128, H2, S128, L4, F128

**Suitability: strong clustered sequence candidate.**

- Head dimension 64 is a natural tensor-core attention size with little padding
  or scalar cleanup.
- B=64 and two query halves already provide approximately a full block wave;
  the two heads add useful warpgroup-level parallelism inside each CTA.
- TMA producer warps can load K/V tiles while consumer warpgroups compute QK,
  online softmax, and PV.
- The same 128 KiB base sequence working set as case 1 permits all four layers
  to remain resident with aliasing.

**Schedule:** two-CTA sequence clusters; distribute heads across consumer
warpgroups and overlap K/V movement with WGMMA.

### Case 11: B64, D128, H16, S128, L4, F128

**Suitability: strong, with attention head packing.**

- Projection and FFN work are unchanged from cases 1, 9, and 10, so the
  sequence-resident argument still holds.
- Sixteen heads create abundant apparent attention parallelism, but each head
  has dimension 8. A CTA-per-head schedule would create tiny tasks dominated by
  dispatch, synchronization, and softmax setup.
- Assign several heads to each CTA or warpgroup and reuse loaded query/key rows
  across the grouped work where layouts permit.
- QK should compare K=16 padded tensor-core execution with SIMT dot products.
  Packing must not introduce cross-head products; heads remain mathematically
  independent.
- PV reduces over S rather than head dimension and may still use tensor cores
  efficiently, with a narrow N=8 output and appropriate output packing.

**Schedule:** two-CTA sequence clusters with multiple heads per CTA, specialized
head-dimension-8 QK, and a packed narrow-N PV path.

### Case 12: B64, D128, H4, S32, L4, F128

**Suitability: strong phase/DAG megakernel candidate.**

- The entire model is approximately 1.680 GFLOP and attention contributes only
  about 4.1%, so a conventional implementation spends disproportionate time in
  small linear and elementwise launches.
- One sequence provides M=32 for linear layers, underfilling a natural M64
  WGMMA tile. Pairing two independent sequences along M fills the tile without
  mixing their attention domains.
- Flattening all tokens gives M=2048. With M64 and N64 tiling, each projection
  exposes 64 tiles, and the three Q/K/V branches provide ample persistent-grid
  work.
- The global activation is only 0.5 MiB. Saving its traffic is less valuable
  than preserving good tile occupancy, so a phase/DAG design is preferable to
  assigning one underfilled CTA to every sequence.
- Attention can use one warp or compact CTA per sequence/head because S=32 fits
  in a single key tile.

**Schedule:** pair sequences for linear tiles, separate them for attention, and
use a fine-grained phase/DAG megakernel with fused epilogues.

### Case 13: B64, D128, H4, S1024, L4, F128

**Suitability: conditional; prefer a hybrid until a megakernel matches the
standalone attention body.**

- The case performs approximately 120 GFLOP, with attention accounting for
  about 57%. Launch removal is useful but secondary to streaming-attention
  quality.
- One sequence's `x` is 256 KiB, already too large for a single CTA. QKV adds
  another 768 KiB, so sequence residency would require a sizable cluster and
  substantial DSM traffic.
- With 128-query tiles there are `64 * 4 * 8 = 2,048` attention tasks. That is
  enough parallelism without splitting the K dimension.
- Causal tasks have increasing K-loop lengths. A persistent scheduler should
  issue later query tiles first so long jobs begin early and short jobs fill the
  tail.
- A 128x128 query/key tile is a sensible starting point on H100, but it must be
  autotuned against register pressure, shared-memory use, head dimension 32,
  and the actual padding-mask distribution.
- An end-to-end persistent wrapper is justified only if its attention task body
  remains close to the standalone Hopper kernel. Otherwise use a CUDA Graph
  around fused layerwise kernels.

**Schedule:** FA3-style persistent attention with FP32 online softmax and
longest-query-first ordering, plus separate fused projection/FFN kernels.

### Case 14: B32, D1024, H16, S100000, L2, F1024

**Suitability: unsuitable for an end-to-end megakernel.**

- The approximate forward cost is 1.391 PFLOP, with causal attention
  contributing about 94.2%. Even at idealized H100 tensor-core throughput this
  is a long-running call, so saving several host launches cannot materially
  change the result.
- A single `x` buffer is approximately 6.10 GiB, and one sequence alone is
  approximately 195 MiB. No useful model-level activation residency is
  possible.
- Materializing scores would require
  `32 * 16 * 100000 * 100000 = 5.12e12` FP16 values, or 10.24 TB in decimal
  units. Streaming online softmax is mandatory.
- With 128-query tiles, the attention grid contains
  `32 * 16 * ceil(100000 / 128) = 400,384` tasks. This is already far more than
  enough parallelism for one H100.
- Because causal tasks have widely varying lengths, map block IDs so the latest
  query tiles, which traverse the most K/V tiles, are claimed first. A
  persistent monotonic counter over a longest-first task ordering avoids a few
  very long jobs forming the final tail.
- Do not begin with split-K attention. It adds partial max/sum/output storage and
  a merge step, while the query dimension already supplies hundreds of
  thousands of tasks. Add split-K only if profiling demonstrates a remaining
  long-task occupancy or tail problem.
- The linear path has M=3.2 million and D=1024, which is also large enough for
  standalone tensor-core GEMMs to amortize launches fully.

**Schedule:** layerwise large-M GEMMs and a standalone streaming causal
attention kernel using approximately 128x128 query/key tiles, FP32 online
softmax, and longest-query-first persistent scheduling. Use CUDA Graph replay
for orchestration rather than a model-wide kernel.

## Correctness constraints

The optimized implementation must match the boundaries in
[`torch_transformer_benchmark.py`](../benchmarks/torch_transformer_benchmark.py), not just
the transformer equation in abstract.

- Accumulate LayerNorm statistics and online-softmax state in FP32.
- Preserve the reference's observable FP16 rounding points. In particular, the
  baseline materializes FP16 Q/K/V projections, forms attention scores before
  explicitly converting scores to FP32 for softmax, casts probabilities back
  to FP16, and then computes PV. A fused implementation that retains every
  intermediate in FP32 can be more mathematically accurate and still disagree
  with the reference.
- Store or explicitly cast LayerNorm outputs, projection outputs, scaled
  attention scores, softmax probabilities, GELU outputs, and residual sums to
  FP16 at the corresponding PyTorch-visible boundaries. In a sequence-resident
  kernel these boundaries can be preserved in shared memory without adding HBM
  traffic.
- Implement `gelu(..., approximate="none")`; a tanh-approximate GELU is not
  equivalent.
- Apply both causal and invalid-key masks before softmax. Invalid query outputs
  must be zeroed after attention, after each transformer block, and after final
  LayerNorm as required by the reference.
- Use the actual valid prefix length to bound attention work when padding is
  present. A fast all-valid specialization is useful only if the judge permits
  detecting or assuming that condition.
- Test accumulation order and rounding after every fused stage, not only at the
  final model output. Four layers can amplify a small per-stage discrepancy.

The comparison uses
`max(atol, rtol * abs(reference))` as its effective per-element tolerance.
Because the model ends in LayerNorm, valid outputs generally remain order one
rather than scaling in proportion to `input_scale`. Larger input scales do not
automatically loosen the gate.

The generated input is a dtype-quantized approximation to
`N(0, input_scale^2)`. For the default LayerNorm epsilon `1e-5`, the first
normalization changes regime around `input_scale ~= sqrt(1e-5) ~= 0.00316`:

- above that scale, normalization largely removes input scale from the
  attention and FFN branches, although the raw residual still scales;
- near that scale, epsilon affects the denominator and small implementation
  differences can be amplified;
- at large scales, the residual dominates but may overflow FP16 or absorb
  order-one branch updates when they fall below an FP16 ULP;
- at very small scales, FP16 input/internal values may underflow and the model's
  linear biases become comparatively important.

Accuracy validation should sweep `input_scale` over at least
`1e-4, 1e-3, 3e-3, 1e-2, 0.1, 1, 10, 100, 1000` and report
`abs_error / max(atol, rtol * abs(reference))` in addition to raw error.

There is currently a specification discrepancy to resolve: the README states
absolute/relative thresholds of 0.001/0.01, while the benchmark CLI defaults to
0.002/0.02. Development should target the stricter README values until the
organizer clarifies the actual judge.

## Toolchain notes

- The current shell exposes CUDA 12.0 through `/usr/bin/nvcc`.
- The official FlashAttention-3 Hopper path requires CUDA 12.3 or newer and
  recommends CUDA 12.8. CUDA 12.8 should therefore be the project baseline for
  H100 work.
- The current shell does not have PyTorch installed and cannot communicate with
  an NVIDIA driver, so it cannot validate these scheduling hypotheses locally.
  Profiling must run in the actual H100 environment.
- Determine the H100 SM count at runtime instead of hard-coding an SXM or PCIe
  count. Resident worker and cluster grids should be derived from the device
  properties and checked with CUDA occupancy APIs.

## Initial benchmarking order

1. Establish standalone FP16 correctness checkpoints after LayerNorm, QKV,
   scores, softmax, PV, output projection, and FFN.
2. Implement cases 5 and 6 first to validate the one-CTA sequence-resident body
   and its global sequence queue.
3. Extend the body to two-CTA clusters for cases 1, 9, and 10.
4. Add fine-grained DAG scheduling for cases 2--4 and 12.
5. Add the head-dimension-8 variants for cases 7 and 11, benchmarking SIMT
   against padded tensor-core QK.
6. Build the layerwise D1024 hybrid for case 8.
7. Integrate and tune streaming attention for case 13.
8. Treat case 14 as a separate long-context project after memory capacity and
   reference-runtime feasibility are confirmed on the judge H100.
