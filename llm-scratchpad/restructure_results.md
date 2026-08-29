# Repository restructure regression

Date: 2026-08-30 (Asia/Singapore)

## Layout and public API

The repository now separates executable concerns into `benchmarks/`,
`kernels/`, and `profiling/`. CUDA sources live under `kernels/csrc/`.
`job-scripts/` contains only Slurm scripts; all historical and future logs,
Nsight reports, and text exports live under `job-scripts/outputs/`.

`optimized_transformer.make_optimized_transformer(parameter_model)` is the
single production constructor. It detects cases 1--12 from `model.config`,
selects the sequence-resident, DAG, or D1024 hybrid implementation, and fully
prepares it before returning.

All Python modules compile, all safe module imports succeed, every shell script
passes `bash -n`, all 17 Slurm scripts target `job-scripts/outputs/`, and the
factory's case mapping was checked for every implemented case.

## H200 regression

Job 772262 used CUDA 12.9.86 on the 132-SM H200 NVL. Correctness covered every
implemented case with both all-valid and 25%-padded inputs. All 358,776,832
values passed the strict `atol=0.001 OR rtol=0.01` gate, and every repeated
launch was identical.

Performance used the same 25-warmup/100-repeat command immediately before and
after restructuring. Job 772253 is the before snapshot; job 772262 is after.

| Case | Before | After | Change |
| ---: | ---: | ---: | ---: |
| 1 | 0.168928 ms | 0.169120 ms | +0.114% |
| 2 | 0.108000 ms | 0.108128 ms | +0.119% |
| 3 | 0.108640 ms | 0.107680 ms | -0.884% |
| 4 | 0.110560 ms | 0.110272 ms | -0.260% |
| 5 | 0.263232 ms | 0.263456 ms | +0.085% |
| 6 | 12.392352 ms | 12.380208 ms | -0.098% |
| 7 | 0.094528 ms | 0.095104 ms | +0.609% |
| 8 | 1.066160 ms | 1.070944 ms | +0.449% |
| 9 | 0.158576 ms | 0.160608 ms | +1.281% |
| 10 | 0.155712 ms | 0.156768 ms | +0.678% |
| 11 | 0.220576 ms | 0.220960 ms | +0.174% |
| 12 | 0.105056 ms | 0.104976 ms | -0.076% |

Checksums and tuning selections were unchanged. The maximum observed movement
was +1.281%, within the normal timing spread for these sub-millisecond kernels;
there is no restructuring performance regression.
