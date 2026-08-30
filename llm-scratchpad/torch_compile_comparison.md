# Production kernels versus `torch.compile`

Date: 2026-08-30 (Asia/Singapore)

Job 774077 ran the common construction API on a 132-SM NVIDIA H100 NVL 96 GB
with CUDA 12.9.86, Torch 2.13.0+cu129, and Triton 3.7.1. Each row used 25 ms
warmup and a 100 ms repetition window through `triton.testing.do_bench`.
Production and eager outputs were checked against the strict repository gate;
as requested, `torch.compile` is a timing-only reference and its numerical
output was not validated.

| Case | Kernel family | Eager | Production | Fastest compile | Compile mode | Production vs compile |
| ---: | --- | ---: | ---: | ---: | --- | ---: |
| 1 | two-role resident | 1.552064 ms | 0.167904 ms | 0.286112 ms | max-autotune | 1.704x faster |
| 2 | four-role static DAG | 1.598784 ms | 0.112672 ms | 0.152096 ms | max-autotune | 1.350x faster |
| 3 | four-role static DAG | 1.610144 ms | 0.106048 ms | 0.154272 ms | max-autotune | 1.455x faster |
| 4 | four-role static DAG | 1.612224 ms | 0.107872 ms | 0.186528 ms | max-autotune | 1.729x faster |
| 5 | one-CTA resident | 1.647840 ms | 0.263232 ms | 0.436560 ms | max-autotune | 1.658x faster |
| 6 | one-CTA resident | 96.262367 ms | 12.533920 ms | 27.892096 ms | max-autotune | 2.225x faster |
| 7 | two-role resident | 1.587664 ms | 0.094976 ms | 0.207808 ms | max-autotune | 2.188x faster |
| 8 | D1024 layerwise hybrid | 2.511440 ms | 1.361248 ms | 1.541120 ms | reduce-overhead | 1.132x faster |
| 9 | one-CTA resident | 1.479392 ms | 0.157664 ms | 0.204864 ms | max-autotune | 1.299x faster |
| 10 | two-role resident | 1.581152 ms | 0.154720 ms | 0.284032 ms | reduce-overhead | 1.836x faster |
| 11 | two-role resident | 1.919488 ms | 0.219200 ms | 0.539008 ms | max-autotune | 2.459x faster |
| 12 | four-role paired-S32 DAG | 1.590480 ms | 0.104480 ms | 0.198368 ms | reduce-overhead | 1.899x faster |
| 13 | exact-attention hybrid + graph | 21.733888 ms | 5.534488 ms | 5.854984 ms | max-autotune | 1.058x faster |

The initial sweep found cases 1--12 already ahead and case 13 as the sole gap:
6.318 ms production versus 5.614 ms max-autotune. Subsequent exact-score, PV,
and graph work closed it. Job 774107 alternated provider order for six 500-ms
rounds; the median-of-rounds result above has production 5.8% ahead. The node
drifted during the run, but a cleaner earlier paired measurement also had
production ahead, 5.205 versus 5.463 ms (4.9%).

The harness resets Torch's compiler state before every case/mode. This is
required because all shape variants share the same Python `forward` code
object; without the reset, Dynamo reaches its recompile limit after eight
variants and later rows are not valid compiled measurements.
