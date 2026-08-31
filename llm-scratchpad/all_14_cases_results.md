# All-14-case construction and validation

Date: 2026-08-31 (Asia/Singapore)

## Fix

`IMPLEMENTED_CASE_LAYERS` in `optimized_transformer.py` is now the single
source of truth for benchmark layer counts. Cross-case construction in the
performance, regression, Torch compile comparison, SDPA comparison, and
profiling entrypoints uses it. This removes the four-layer assumption that made
`benchmark_compile_comparison.make_config(14)` incompatible with
`make_optimized_transformer`.

The regression and performance harnesses include all 14 cases by default.
Case 14 uses its memory-bounded exact reference for correctness and reduced
long-context timing windows. The compile comparison reports the eager and
compiled case-14 baselines as `SKIP`, rather than attempting their approximately
9.3 TiB materialized `[B,H,S,S]` score tensor; production is still measured.

## Environment and commands

Both final jobs ran on `xgpk0`, one NVIDIA H200 NVL with 132 SMs, CUDA
12.9.86, Torch 2.13.0+cu129, and Triton 3.7.1.

```bash
sbatch --gres=gpu:h200-141:1 job-scripts/validate_all_cases_h200.sh
# job 782018, COMPLETED 0:0

sbatch --gres=gpu:h200-141:1 job-scripts/compare_torch_compile_h200.sh
# job 782017, COMPLETED 0:0
```

Static checks also passed:

```bash
python -m py_compile optimized_transformer.py \
  benchmarks/benchmark_megakernels.py benchmarks/regression.py \
  benchmarks/benchmark_compile_comparison.py \
  benchmarks/benchmark_sdpa_backends.py profiling/profile_megakernel.py
git diff --check
bash -n job-scripts/validate_all_cases_h200.sh \
  job-scripts/compare_torch_compile_h200.sh
```

A CPU construction audit instantiated the organizer model config for every
case, confirmed the centralized layer count, and confirmed that
`case_number_for_model` dispatched all 14 cases correctly.

## Correctness and determinism

All 28 rows passed: every case at padding ratios 0.00 and 0.25. Every row had
zero gate failures and `repeat_diff=0`. Cases 1--13 used `atol=0.001` and
`rtol=0.01`; case 14 used the previously authorized doubled gate,
`atol=0.002` and `rtol=0.02`.

Case 14 correctness used `B=1, S=100000, D=1024`, the full sequence with a
query-chunked exact reference, because the exact batch-32 reference is not
memory-feasible. Its two rows each compared 102,400,000 elements. Production
performance below uses the actual `B=32` case.

| Case | padding 0.00 max abs | padding 0.25 max abs | Result |
| ---: | ---: | ---: | --- |
| 1 | 0 | 0 | PASS |
| 2 | 0 | 0 | PASS |
| 3 | 0 | 0 | PASS |
| 4 | 0 | 0 | PASS |
| 5 | 0 | 0 | PASS |
| 6 | 0.002929688 | 0 | PASS |
| 7 | 0 | 0 | PASS |
| 8 | 0.00390625 | 0 | PASS |
| 9 | 0 | 0 | PASS |
| 10 | 0 | 0 | PASS |
| 11 | 0 | 0 | PASS |
| 12 | 0 | 0 | PASS |
| 13 | 0 | 0 | PASS |
| 14 | 0.005859375 | 0.0078125 | PASS |

## Production performance

Cases 1--13 used a 25 ms warmup and 100 ms measurement window. Case 14 used a
3000 ms warmup and 15000 ms window, yielding multiple full forwards. Its RMS
gate selected FA3.

| Case | Family | Median | p20 | p80 |
| ---: | --- | ---: | ---: | ---: |
| 1 | resident | 0.168864 ms | 0.168064 | 0.169792 |
| 2 | DAG | 0.108032 ms | 0.107264 | 0.109440 |
| 3 | DAG | 0.107616 ms | 0.106816 | 0.113536 |
| 4 | DAG | 0.110240 ms | 0.109600 | 0.111008 |
| 5 | resident | 0.262960 ms | 0.261485 | 0.264627 |
| 6 | resident | 12.302160 ms | 12.270003 | 12.316115 |
| 7 | resident | 0.094656 ms | 0.094048 | 0.095232 |
| 8 | hybrid | 1.068272 ms | 1.045421 | 1.073888 |
| 9 | resident | 0.160336 ms | 0.159456 | 0.161408 |
| 10 | resident | 0.156832 ms | 0.156064 | 0.157933 |
| 11 | resident | 0.220576 ms | 0.219712 | 0.221952 |
| 12 | DAG | 0.104896 ms | 0.104192 | 0.106118 |
| 13 | hybrid | 4.447024 ms | 4.444736 | 4.452160 |
| 14 | long context / FA3 | 2942.844482 ms | 2941.677051 | 2943.259033 |

## `torch.compile` comparison

The compile harness resets Dynamo between every case and mode. As in the
existing comparison methodology, eager and production are checked for
correctness while compiled providers are timing-only references. Production
passed cases 1--13 and beat the fastest of `default`, `reduce-overhead`, and
`max-autotune` in every feasible comparison.

| Case | Production | Fastest compile | Mode | Compile / production |
| ---: | ---: | ---: | --- | ---: |
| 1 | 0.169280 ms | 0.277312 ms | max-autotune | 1.638x |
| 2 | 0.114272 ms | 0.142848 ms | max-autotune | 1.250x |
| 3 | 0.109408 ms | 0.145792 ms | max-autotune | 1.333x |
| 4 | 0.109664 ms | 0.166208 ms | max-autotune | 1.516x |
| 5 | 0.262880 ms | 0.399840 ms | max-autotune | 1.521x |
| 6 | 12.374400 ms | 22.932496 ms | max-autotune | 1.853x |
| 7 | 0.095456 ms | 0.198080 ms | max-autotune | 2.075x |
| 8 | 1.065472 ms | 1.230720 ms | reduce-overhead | 1.155x |
| 9 | 0.158224 ms | 0.200288 ms | max-autotune | 1.266x |
| 10 | 0.155680 ms | 0.250368 ms | max-autotune | 1.608x |
| 11 | 0.220128 ms | 0.471552 ms | max-autotune | 2.142x |
| 12 | 0.104352 ms | 0.156192 ms | max-autotune | 1.497x |
| 13 | 4.443712 ms | 5.017024 ms | max-autotune | 1.129x |
| 14 | 2927.799072 ms | infeasible | materialized 9.3 TiB scores | N/A |

Case 14's production timing in this independent sweep had p20 2924.476270 ms
and p80 2932.586523 ms. Its numerical result is supplied by the full-sequence
memory-bounded regression above.

Raw logs:

- `job-scripts/outputs/validate_all_cases_h200-782018.out`
- `job-scripts/outputs/compare_torch_compile_h200-782017.out`
