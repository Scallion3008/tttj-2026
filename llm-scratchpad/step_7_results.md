# Step 7 / case 13 optimization record

Shape: `B=64, S=1024, D=128, H=4, HD=32, F=128, L=4`, causal FP16.

## Structure decision

The hybrid plan from `initial_notes.md` is the right first implementation. The
attention grid has thousands of independent query tiles and a single sequence
cannot remain in one CTA. A whole-model megakernel would either serialize this
parallelism or need multi-CTA synchronization while also matching the
standalone attention library. Projection and FFN GEMMs already have ample
parallelism. The selected structure is therefore:

- exact Triton QKV projection writing contiguous head-major tensors;
- exact materialized attention for the strict ordinary-input path;
- cuDNN SDPA for the numerically safe large-input path;
- exact Triton linear+GELU and linear+residual+LayerNorm epilogues around the
  remaining cuBLAS projections;
- direct-input CUDA Graph replay for the fixed all-valid steady state, with
  eager fallback for changed input pointers, mutated inputs, or padding.

## Attention library comparison

The official FlashAttention-3 CUTLASS repository was built at commit
`ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820` with CUDA 12.9, forward-only sm_90
HD64 specialization (the repository pads HD32 into its minimum Hopper tile).
The build is CPU-only; `job-scripts/build_fa3.sh` does not reserve a GPU.

The cuDNN loader failure was fixed by pinning the complete cuDNN 9.24 wheel and
putting its library directory first in `LD_LIBRARY_PATH`. This supplies
`libcudnn_engines_runtime_compiled.so.9`, which the earlier environment lacked.

On H200, all-valid streaming attention alone measured 0.1221 ms for cuDNN and
0.1428 ms for Torch Flash. The requested full H100-96 library comparison is
job 774121: the packed official-FA3 hybrid measured 1.860 ms and packed cuDNN
measured 1.780 ms, making cuDNN 4.5% faster. H200 job 774124 reproduced the
ranking at 1.791 versus 1.722 ms (4.0%). Both full streaming paths fail the
ordinary model gate (1,264 values for FA3 and 1,255 for cuDNN). H100-47
measurements were used only for correctness and not for performance selection.

Using streaming attention in every layer is far outside the strict gate: the
ordinary H200 case had 1,255 failed final elements with cuDNN and 1,298 with
FA3. Layer-mask sweeps found layer 4 alone passed the single ordinary case, but
a 20-seed strict sweep still found rare failures. Inputs with RMS at least 3
passed all layer masks in the sampled sweep; production uses the more
conservative RMS >= 4 threshold and otherwise retains exact attention. Padded
inputs always use the exact path.

## Exact-path optimizations

The organizer exposes two FP16 boundaries: after scaled/masked QK and after the
FP32 softmax. Replacing either with a streaming reduction changes enough ulps
to fail after four layers. The optimized exact path retains both boundaries
using only PyTorch library operations:

1. `torch.compile` fuses score scaling and causal masking around the QK GEMM.
2. CUDA FP16 softmax for fixed S1024 accumulates in FP32 and writes exactly the
   same FP16 probabilities as `softmax(scores.float()).half()`. Calling it
   directly removes the 1 GiB FP32 probability allocation.
3. The compiled score mask is not applied a second time.

H100-47 correctness probes confirmed that each transformation remains
bit-exact. Latency and roofline claims are intentionally deferred to full
H100-96 or H200 runs.

The paired H100-96 Nsight profiles show that direct FP16 softmax removes a
439.8 us FP32-to-FP16 probability cast and reduces one attention call from
about 1.821 ms to 1.331 ms (26.9%). In the updated path, score masking reaches
88.8% DRAM throughput, QK reaches 74.2%, and PV reaches 85.7%; softmax is the
only compute-bound stage at 84.0% SM throughput. The benchmark roofline
therefore counts estimated materialized-score traffic in addition to model
tensors.

On the full H100-96, the first final-candidate pass measured 6.476 ms for exact
attention plus residual/norm fusion and 6.410 ms for the exact linear-epilogue
hybrid. CUDA Graph replay only changed the latter to 6.402 ms, a 0.1% effect
that is too small to select without H200 confirmation. A bit-exact tile sweep
then improved the fused GELU microkernel by about 24% and the fused
linear/residual/norm microkernel by about 5%; those settings were retained in
the subsequent candidate runs.
The next full-H100 iteration removed the three per-layer Q/K/V layout
conversions by writing head-major tensors from the QKV projection. In the same
run it reduced 7.040 ms to 6.733 ms (4.4%) versus the packed-QKV linear-epilogue
hybrid. Its 64-row, 4-warp, 2-stage projection configuration is bit-exact to
the packed cuBLAS values in the tuning sweep. Absolute H100 timings varied with
node contention, so only the within-run change is used; H200 remains
useful as a cross-check.

The final surrounding-kernel iteration reuses the exact Welford kernel for the
one standalone input LayerNorm. It measured 0.0181 ms versus 0.0842 ms for
PyTorch LayerNorm; an 8-row, 2-warp specialization reduced it further to about
0.0158 ms. Together, head-major QKV and input normalization reduced the
within-run exact hybrid from 6.967 ms to 6.539 ms on H100-96. CUDA Graph replay
was slower at 6.576 ms for that older kernel sequence. After exact attention
was shortened, graph replay was retested and became beneficial.

The final head-major Nsight capture contains only four attention kernels (the
three layout conversions are gone): QK at 74.1% DRAM, scale/mask at 88.5% DRAM,
softmax at 84.4% SM, and PV at 85.5% DRAM. Their durations sum to 1.306 ms on
H100-96.

## Validation

- Five strict public-regression trials at padding 0 and 0.25: bit-exact,
  0 failures in 83,886,080 outputs.
- Production scale/padding matrix (three trials each at scales 1e-4, 0.1, 1,
  10, and 1000; padding 0 and 0.25): all pass and deterministic.
- cuDNN large-scale route: all sampled scale 10/1000 cases pass the strict OR
  gate; exact padded route remains bit-exact.
- The promoted linear-epilogue production candidate passed five public
  regression trials at padding 0 and 0.25 and the same three-trial
  scale/padding matrix. H100-47 was used only for these correctness checks.
- The subsequently promoted head-major-QKV candidate also passed five public
  regression trials at both padding ratios and the full scale/padding matrix.

The initial all-case H100 comparison in `torch_compile_comparison.md` measured
the then-current production path at 6.318 ms and whole-model max-autotune at
5.614 ms, making case 13 the only implemented case behind the compiler.
Applying Inductor's faster compiled FP16 softmax only in layer 4 came closest
to preserving accuracy, but still failed 1 value over 100 standard-scale seeds
on the correctness-only H100-47 MIG and 7 values over 120 full-H100 trials
spanning scales 0.5--3. Fully compiling only layer-4 materialized attention
failed 18 values over the same 100-seed standard-scale MIG sweep. Both
shortcuts remain rejected; production retains the exact native softmax.

## Closing the compiler gap

The accepted compiler/library optimizations preserve both observable FP16
boundaries rather than compiling the reduction:

1. max-autotune selects the fastest exact QK GEMM and fuses FP16 scaling with
   an additive causal mask;
2. native FP16 softmax retains the exact FP32 reduction and FP16 output;
3. max-autotune selects a bit-identical PV GEMM;
4. the public dispatcher replays the fixed launch sequence through a
   direct-input CUDA Graph.

The input-mask, generated-index-mask, and additive-mask variants all produced
bit-identical 268-million-element score tensors and bit-identical attention
outputs. On H100, additive max-autotune reduced isolated exact attention from
1.396 ms (default compiler mode) to 1.125 ms; on H200 it reduced 1.216 ms to
0.994 ms. Exact tuned PV was also bit-identical and reduced its H200 component
from 0.154 to 0.147 ms. The complete H200 eager hybrid measured 4.521 ms and
graph replay measured 4.449 ms.

Job 774107 is the final paired H100-96 comparison. Across six alternating
500-ms rounds, graph production measured 5.534 ms versus 5.855 ms for
whole-model max-autotune: production is **1.058x faster**. A less contended
single-round comparison of the preceding exact-score candidate measured 5.205
versus 5.463 ms and reached the same compiler-gap conclusion. Component and
H200 comparisons then selected the additive mask, tuned PV, and graph replay.

Final Nsight job 774112 profiles three exact-attention kernels on H100-96:

| Kernel | Duration | Binding metric | Peak utilization |
| --- | ---: | --- | ---: |
| fused QK + scale + additive mask | 234.34 us | memory/L1 | 74.65% memory |
| native exact FP16 softmax | 632.13 us | compute | 84.17% SM |
| tuned PV | 162.94 us | DRAM | 89.01% DRAM |

They sum to 1.029 ms, down 21.2% from the prior 1.306 ms profile. Softmax is
now the only substantial local opportunity, but every faster reduction tested
failed the numerical gate, including a layer-4-only specialization.

The finalized additive-mask/tuned-PV production path passed the H100 and H200
three-trial scale/padding matrices (scales 1e-4, 0.1, 1, 10, and 1000; padding
0 and 0.25) and five public-regression trials at both padding ratios. Ordinary
and padded outputs were bit-exact; the adaptive large-scale cuDNN route stayed
within the strict gate and deterministic.
