# Results

Measured on a Tesla T4 (sm75, 15.6 GB, CUDA 12.8) by a top to bottom run of
`notebooks/bench_t4.ipynb` on 2026-08-27, model Qwen/Qwen2.5-1.5B-Instruct in float16,
after the correctness gates in that notebook passed on the same hardware. The build host
(macOS, no CUDA GPU) produced no performance number. Raw CSVs and figures are attached to
the release on GitHub.

The clockwork matrix ran three times in this session. Every clockwork cell below is the
median of those three repetitions; throughput agreed to within 1.3 percent across them
(median spread 0.3 percent). The vLLM matrix ran once.

## Evaluation protocol

`scripts/run_bench.py` drives a running server over HTTP with streaming requests at
temperature 0.0 and writes one CSV row per request plus a `summary.csv` row per workload.
Workload generation is seeded and deterministic (`clockwork/bench/workloads.py`); arrivals
follow the workload's Poisson or Pareto process at the configured rate. Metrics, as
computed by `clockwork/bench/metrics.py`:

- TTFT: ms from client-side request send to the first content chunk; nearest-rank
  percentiles over requests.
- ITL: ms gaps between successive content chunks, percentiles over all gaps.
- output tok/s: completion tokens divided by the span from first request start to last
  request end.
- hit rate: for clockwork, `usage.prompt_tokens_details.cached_tokens` over prompt tokens;
  for vLLM, its Prometheus prefix-cache counters sampled around each workload, since vLLM
  does not report cached tokens in the usage payload.

Errored requests are counted and excluded from latency statistics. Multi-turn sessions
send each turn after the previous turn completes, so cross-engine latency comparisons on
the same trace are schedule dependent.

## Clockwork workload matrix

| workload | kind | req/s | radix | ttft p50 (ms) | ttft p99 (ms) | itl p50 (ms) | itl p99 (ms) | out tok/s | hit rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharegpt_r1 | sharegpt | 1 | on | 113.2 | 539.8 | 31.3 | 95.9 | 111.7 | 0.052 |
| sharegpt_r2 | sharegpt | 2 | on | 121.6 | 485.5 | 32.8 | 136.4 | 195.9 | 0.048 |
| sharegpt_r4 | sharegpt | 4 | on | 170.9 | 966.0 | 45.0 | 248.1 | 317.5 | 0.045 |
| sharegpt_r8 | sharegpt | 8 | on | 260.4 | 3641.2 | 65.7 | 310.7 | 466.2 | 0.052 |
| sharegpt_r16 | sharegpt | 16 | on | 1647.9 | 11883.4 | 65.8 | 416.4 | 499.0 | 0.052 |
| agent_p1024_t4to8_pois_r2 | agent | 2 | on | 226.9 | 441.7 | 75.4 | 252.7 | 281.2 | 0.893 |
| agent_p1024_t4to8_pois_r4 | agent | 4 | on | 334.8 | 3811.1 | 122.2 | 366.9 | 298.9 | 0.874 |
| agent_p1024_t4to8_pois_r8 | agent | 8 | on | 2287.0 | 19557.5 | 143.2 | 1134.0 | 326.9 | 0.833 |
| agent_p1024_t4to8_pareto_r4 | agent | 4 | on | 246.7 | 711.5 | 83.4 | 317.7 | 230.8 | 0.875 |
| agent_p1536_t6to12_pois_r2 | agent | 2 | on | 304.5 | 1847.3 | 90.2 | 324.3 | 183.6 | 0.884 |
| agent_p1536_t6to12_pois_r4 | agent | 4 | on | 529.0 | 8249.8 | 155.2 | 481.9 | 250.5 | 0.902 |
| agent_p1536_t6to12_pois_r8 | agent | 8 | on | 18721.1 | 45126.6 | 102.5 | 740.3 | 211.6 | 0.881 |
| agent_p1536_t6to12_pareto_r4 | agent | 4 | on | 996.6 | 13965.4 | 180.5 | 667.3 | 259.8 | 0.897 |
| agent_p1536_t6to12_pareto_r8 | agent | 8 | on | 971.2 | 18584.2 | 181.7 | 739.5 | 261.6 | 0.893 |
| agent_p2048_t8to16_pois_r4 | agent | 4 | on | 273.0 | 1166.0 | 75.3 | 388.0 | 162.0 | 0.903 |
| agent_p2048_t8to16_pois_r8 | agent | 8 | on | 882.6 | 8861.0 | 185.3 | 854.5 | 237.6 | 0.887 |
| agent_p2048_t8to16_pareto_r8 | agent | 8 | on | 601.8 | 15003.1 | 202.2 | 632.7 | 237.6 | 0.907 |
| ablation_p1536_radix_on_r2 | ablation | 2 | on | 195.7 | 912.6 | 79.7 | 244.6 | 176.3 | 0.912 |
| ablation_p1536_radix_on_r8 | ablation | 8 | on | 2397.6 | 25469.2 | 183.4 | 1008.6 | 249.0 | 0.857 |
| ablation_p1536_radix_off_r2 | ablation | 2 | off | 7895.3 | 26973.6 | 196.9 | 1211.2 | 123.7 | 0.000 |
| ablation_p1536_radix_off_r8 | ablation | 8 | off | 25450.1 | 74227.1 | 201.4 | 1195.6 | 122.9 | 0.000 |

Agent workload names encode shared prefix tokens (p), turns per session (t), and the
arrival process. Each ablation pair shares a seed, so the on and off runs replay an
identical trace.

## Radix ablation

Identical traces, server relaunched with the prefix cache off. The cache buys throughput
and, far more dramatically, tail latency:

| rate | out tok/s on | out tok/s off | ratio | ttft p50 on | ttft p50 off | p50 cut | p99 cut |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 req/s | 176.3 | 123.7 | 1.42x | 196 ms | 7895 ms | 98% | 97% |
| 8 req/s | 249.0 | 122.9 | 2.03x | 2398 ms | 25450 ms | 91% | 66% |

A cold versus warm experiment isolates the same effect on one trace with no config change:
the workload runs once against an empty cache, then replays identically against the warm
cache. TTFT p50 falls from 346.2 ms to 178.1 ms
(48.6 percent) and p99 from 2227.6 ms to
512.6 ms (77.0 percent).

## Paged decode kernel

Triton kernel versus the torch fallback on identical fp16 tensors at model shapes, CUDA
events with warmup:

| batch | ctx len | torch (ms) | triton (ms) | speedup |
| --- | --- | --- | --- | --- |
| 1 | 128 | 0.3946 | 0.0475 | 8.31x |
| 1 | 512 | 0.4048 | 0.1461 | 2.77x |
| 1 | 2048 | 0.3916 | 0.5671 | 0.69x |
| 4 | 128 | 0.4241 | 0.0671 | 6.32x |
| 4 | 512 | 0.5457 | 0.1808 | 3.02x |
| 4 | 2048 | 1.8549 | 0.7115 | 2.61x |
| 16 | 128 | 0.436 | 0.0952 | 4.58x |
| 16 | 512 | 1.8662 | 0.5007 | 3.73x |
| 16 | 2048 | 7.7712 | 1.5403 | 5.04x |
| 64 | 128 | 1.823 | 0.4769 | 3.82x |
| 64 | 512 | 7.9072 | 1.6187 | 4.88x |
| 64 | 2048 | 32.4626 | 6.1396 | 5.29x |

The Triton kernel wins 11 of 12 shapes, peaking at 8.31x
(batch 1, ctx 128), so `resolve_backend("auto")` selects it.

## vLLM baseline

Same model, same dtype, same seeded traces, same harness; vLLM v1 on its default settings
as printed in the notebook log. vLLM enables prefix caching by default and its measured hit
rates (0.899 to 0.928 on agent traces) are slightly higher than clockwork's.
Its two radix-off rows are not a vLLM ablation: those traces replay against the same default
vLLM server, so that label describes the clockwork arm only.

| workload | kind | req/s | radix | ttft p50 (ms) | ttft p99 (ms) | itl p50 (ms) | itl p99 (ms) | out tok/s | hit rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharegpt_r1 | sharegpt | 1 | on | 96.7 | 1153.9 | 16.7 | 50.4 | 92.4 | 0.052 |
| sharegpt_r2 | sharegpt | 2 | on | 100.7 | 591.7 | 17.3 | 81.0 | 156.2 | 0.048 |
| sharegpt_r4 | sharegpt | 4 | on | 138.5 | 1428.3 | 20.1 | 163.6 | 288.1 | 0.045 |
| sharegpt_r8 | sharegpt | 8 | on | 162.6 | 541.1 | 28.7 | 266.0 | 502.4 | 0.052 |
| sharegpt_r16 | sharegpt | 16 | on | 834.5 | 3024.1 | 53.7 | 508.6 | 646.4 | 0.052 |
| agent_p1024_t4to8_pois_r2 | agent | 2 | on | 202.2 | 579.1 | 31.1 | 236.3 | 176.5 | 0.905 |
| agent_p1024_t4to8_pois_r4 | agent | 4 | on | 232.5 | 1251.6 | 28.0 | 253.8 | 215.1 | 0.906 |
| agent_p1024_t4to8_pois_r8 | agent | 8 | on | 1285.4 | 2788.1 | 55.8 | 958.0 | 278.7 | 0.902 |
| agent_p1024_t4to8_pareto_r4 | agent | 4 | on | 234.4 | 642.0 | 32.3 | 219.3 | 140.2 | 0.899 |
| agent_p1536_t6to12_pois_r2 | agent | 2 | on | 502.5 | 2006.5 | 31.0 | 469.4 | 105.0 | 0.917 |
| agent_p1536_t6to12_pois_r4 | agent | 4 | on | 392.9 | 1371.9 | 33.1 | 436.1 | 177.8 | 0.923 |
| agent_p1536_t6to12_pois_r8 | agent | 8 | on | 3725.4 | 6621.5 | 53.6 | 1724.5 | 173.0 | 0.920 |
| agent_p1536_t6to12_pareto_r4 | agent | 4 | on | 498.1 | 1573.2 | 38.9 | 774.0 | 179.7 | 0.922 |
| agent_p1536_t6to12_pareto_r8 | agent | 8 | on | 1378.2 | 2555.9 | 61.9 | 1152.3 | 199.4 | 0.917 |
| agent_p2048_t8to16_pois_r4 | agent | 4 | on | 435.9 | 1587.7 | 25.7 | 425.3 | 68.1 | 0.920 |
| agent_p2048_t8to16_pois_r8 | agent | 8 | on | 862.5 | 3854.4 | 62.8 | 643.2 | 137.7 | 0.928 |
| agent_p2048_t8to16_pareto_r8 | agent | 8 | on | 745.3 | 1818.6 | 65.2 | 1002.8 | 172.7 | 0.922 |
| ablation_p1536_radix_on_r2 | ablation | 2 | on | 230.9 | 1825.1 | 27.8 | 305.2 | 102.7 | 0.922 |
| ablation_p1536_radix_on_r8 | ablation | 8 | on | 2995.7 | 4811.2 | 57.9 | 1698.7 | 176.7 | 0.918 |
| ablation_p1536_radix_off_r2 | ablation | 2 | off | 94.2 | 176.1 | 26.1 | 66.7 | 105.0 | 0.996 |
| ablation_p1536_radix_off_r8 | ablation | 8 | off | 138.9 | 236.7 | 45.7 | 85.9 | 271.9 | 0.997 |

Mean output tok/s over the 12 agent workloads: clockwork 245.2 versus vLLM 168.7,
with clockwork faster on 12 of 12 by 1.17x to 2.38x. On the five
single-turn sharegpt rates the order reverses: clockwork 318.0 versus vLLM 337.1, vLLM
ahead by 6 percent, and vLLM's ITL p50 is lower nearly everywhere. Clockwork
trades per-token pacing for admitting prefix-cached sequences sooner.

A fixed-length session experiment (ignore_eos, 64 tokens per turn, so both engines emit
identical token counts) measures end-to-end multi-turn session completion on three agent
traces. vLLM finishes sessions faster: pooled mean session time is
87 percent higher for clockwork. Reported because
it was measured, not because it flatters this engine. Clockwork's advantage in this benchmark
set is aggregate throughput under concurrent agent load, not single-session wall clock.

SGLANG: not-run. Its server did not come up in this environment; the notebook prints the
reason and continues, and no substitute numbers are reported.

## Reproducing

`notebooks/bench_t4.ipynb` reproduces every number above on a free Colab T4 without code
changes. The underlying commands:

```
uv run python scripts/serve.py --config configs/qwen2.5-1.5b-instruct.yaml
uv run python scripts/run_bench.py --configs all --base-url http://127.0.0.1:8000 \
  --out bench_results --plots docs/figures
```

The radix-off ablations need the server relaunched with `--set enable_prefix_cache=false`.
Figures for both engines are in `docs/figures/`.
