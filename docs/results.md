# Results

All numbers in this file were measured on a Tesla T4 (sm75, 15.6 GB, CUDA 12.8) by a
top to bottom run of `notebooks/bench_t4.ipynb` on 2026-08-27, model
Qwen/Qwen2.5-1.5B-Instruct in float16, after the correctness gates in that notebook
passed on the same hardware. The build host (macOS, no CUDA GPU) produced no
performance number.

## Evaluation protocol

`scripts/run_bench.py` drives a running clockwork server over HTTP with streaming
requests at temperature 0.0 and writes one CSV row per request plus a `summary.csv`
row per workload. Workload generation is seeded and deterministic
(`clockwork/bench/workloads.py`); arrivals follow the workload's Poisson or Pareto
process at the configured rate. Metrics, as computed by `clockwork/bench/metrics.py`:

- TTFT: ms from client-side request send to the first content chunk of the stream;
  p50 and p99 are nearest-rank percentiles over requests.
- ITL: ms gaps between successive content chunks; percentiles over all gaps in the
  workload.
- output tok/s: total completion tokens divided by the span from the first request
  start to the last request end.
- hit rate: total `usage.prompt_tokens_details.cached_tokens` reported by the server
  divided by total prompt tokens, so radix hits are measured from the API surface, not
  from engine internals.

Requests that error are counted and excluded from the latency statistics. Multi-turn
sessions send each turn after the previous turn completes, so cross-run latency
comparisons on the same trace are schedule dependent: a faster server pulls later
turns earlier, which changes the load it then faces.

## Clockwork workload matrix

| workload | kind | req/s | radix | ttft p50 (ms) | ttft p99 (ms) | itl p50 (ms) | itl p99 (ms) | out tok/s | hit rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharegpt_r1 | sharegpt | 1 | on | 105.4 | 637.5 | 28.7 | 90.0 | 112.3 | 0.052 |
| sharegpt_r2 | sharegpt | 2 | on | 115.0 | 480.9 | 30.2 | 115.7 | 197.3 | 0.048 |
| sharegpt_r4 | sharegpt | 4 | on | 159.6 | 885.9 | 41.8 | 246.5 | 321.2 | 0.045 |
| sharegpt_r8 | sharegpt | 8 | on | 229.8 | 3022.5 | 61.3 | 278.1 | 489.0 | 0.052 |
| sharegpt_r16 | sharegpt | 16 | on | 1450.5 | 11117.7 | 62.4 | 491.3 | 523.6 | 0.052 |
| agent_p1024_t4to8_pois_r2 | agent | 2 | on | 203.6 | 427.5 | 68.3 | 207.1 | 284.5 | 0.895 |
| agent_p1024_t4to8_pois_r4 | agent | 4 | on | 279.6 | 2560.9 | 116.1 | 336.4 | 307.0 | 0.880 |
| agent_p1024_t4to8_pois_r8 | agent | 8 | on | 1581.4 | 17742.4 | 139.5 | 629.5 | 344.2 | 0.861 |
| agent_p1024_t4to8_pareto_r4 | agent | 4 | on | 218.6 | 773.2 | 81.6 | 312.7 | 233.7 | 0.878 |
| agent_p1536_t6to12_pois_r2 | agent | 2 | on | 266.2 | 1233.2 | 83.0 | 313.8 | 186.4 | 0.898 |
| agent_p1536_t6to12_pois_r4 | agent | 4 | on | 497.8 | 7281.6 | 148.8 | 462.6 | 255.9 | 0.904 |
| agent_p1536_t6to12_pois_r8 | agent | 8 | on | 2005.5 | 43044.9 | 125.8 | 714.8 | 219.6 | 0.907 |
| agent_p1536_t6to12_pareto_r4 | agent | 4 | on | 864.7 | 13125.3 | 176.7 | 668.7 | 266.5 | 0.902 |
| agent_p1536_t6to12_pareto_r8 | agent | 8 | on | 940.2 | 17926.0 | 177.8 | 657.8 | 267.9 | 0.897 |
| agent_p2048_t8to16_pois_r4 | agent | 4 | on | nd | nd | nd | nd | nd | nd |
| agent_p2048_t8to16_pois_r8 | agent | 8 | on | nd | nd | nd | nd | nd | nd |
| agent_p2048_t8to16_pareto_r8 | agent | 8 | on | nd | nd | nd | nd | nd | nd |
| ablation_p1536_radix_on_r2 | ablation | 2 | on | 198.8 | 983.6 | 78.0 | 247.0 | 177.2 | 0.910 |
| ablation_p1536_radix_on_r8 | ablation | 8 | on | 1880.0 | 23410.7 | 178.5 | 705.2 | 261.6 | 0.880 |
| ablation_p1536_radix_off_r2 | ablation | 2 | off | 891.4 | 2554.4 | 32.9 | 416.9 | 46.8 | 0.000 |
| ablation_p1536_radix_off_r8 | ablation | 8 | off | 736.2 | 1846.8 | 33.3 | 447.0 | 65.7 | 0.000 |

Agent workload names encode shared prefix tokens (p), turns per session (t), and the
arrival process; each ablation pair shares a seed, so the on and off runs replay an
identical trace. nd: the three p2048 workloads returned no data because their opening
prompts (2129 to 2308 tokens) exceed the shipped config's `max_num_batched_tokens`
of 2048; the engine prefills per sequence without chunking, so such prompts are never
admitted and the server answers with an empty completion instead of an error. This is
a real limitation of the shipped config, kept in the table rather than hidden.

## Radix ablation

Identical traces, server relaunched with the prefix cache off: radix caching gives
3.78x output tok/s at 2 req/s (177.2 vs 46.8) and 3.98x at 8 req/s (261.6 vs 65.7),
with hit rates of 0.910 and 0.881. All 96 requests completed in both arms of both
pairs. Per-request latency columns are not comparable across arms per the schedule
dependence note above; throughput over the whole trace is the primary ablation metric.

## Paged decode kernel

Triton kernel versus the torch fallback on identical fp16 tensors at model shapes,
CUDA events with warmup:

| batch | ctx len | torch (ms) | triton (ms) | speedup |
| --- | --- | --- | --- | --- |
| 1 | 128 | 0.4381 | 0.0467 | 9.38x |
| 1 | 512 | 0.3805 | 0.1598 | 2.38x |
| 1 | 2048 | 0.3837 | 0.6213 | 0.62x |
| 4 | 128 | 0.3962 | 0.0733 | 5.41x |
| 4 | 512 | 0.6757 | 0.2723 | 2.48x |
| 4 | 2048 | 1.6877 | 0.5401 | 3.12x |
| 16 | 128 | 0.3991 | 0.0867 | 4.60x |
| 16 | 512 | 1.5386 | 0.3946 | 3.90x |
| 16 | 2048 | 6.2156 | 1.338 | 4.64x |
| 64 | 128 | 1.4966 | 0.389 | 3.85x |
| 64 | 512 | 6.2455 | 1.386 | 4.51x |
| 64 | 2048 | 24.9193 | 5.0714 | 4.91x |


The Triton kernel wins 11 of 12 shapes on this GPU, so `resolve_backend("auto")`
selects it; the one loss (batch 1, ctx 2048) is the shape where a single program
iterates the longest block table alone.

## vLLM baseline

Same model, same dtype, same seeded traces, same harness; vLLM v1 on its default
settings as printed verbatim in the notebook log. vLLM does not surface cached token
counts in its usage payload, so its hit rate is not observable (nr). Its radix on/off
rows replay the ablation traces against the same default server, so they differ only
by trace and scheduling, not by any vLLM setting.

| workload | kind | req/s | radix | ttft p50 (ms) | ttft p99 (ms) | itl p50 (ms) | itl p99 (ms) | out tok/s | hit rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharegpt_r1 | sharegpt | 1 | on | 83.3 | 1197.5 | 16.5 | 48.1 | 92.4 | nr |
| sharegpt_r2 | sharegpt | 2 | on | 100.3 | 588.8 | 17.1 | 79.9 | 155.2 | nr |
| sharegpt_r4 | sharegpt | 4 | on | 121.5 | 1390.0 | 19.7 | 163.6 | 289.9 | nr |
| sharegpt_r8 | sharegpt | 8 | on | 165.2 | 551.2 | 29.3 | 274.8 | 509.2 | nr |
| sharegpt_r16 | sharegpt | 16 | on | 864.7 | 3015.1 | 52.7 | 512.4 | 656.8 | nr |
| agent_p1024_t4to8_pois_r2 | agent | 2 | on | 207.8 | 774.1 | 33.5 | 309.5 | 177.8 | nr |
| agent_p1024_t4to8_pois_r4 | agent | 4 | on | 223.1 | 1233.1 | 28.7 | 262.2 | 223.5 | nr |
| agent_p1024_t4to8_pois_r8 | agent | 8 | on | 1240.7 | 2717.6 | 55.5 | 946.2 | 278.2 | nr |
| agent_p1024_t4to8_pareto_r4 | agent | 4 | on | 236.0 | 635.6 | 32.4 | 239.5 | 140.8 | nr |
| agent_p1536_t6to12_pois_r2 | agent | 2 | on | 449.3 | 2552.8 | 31.1 | 423.9 | 102.5 | nr |
| agent_p1536_t6to12_pois_r4 | agent | 4 | on | 406.3 | 1213.5 | 31.5 | 437.1 | 175.4 | nr |
| agent_p1536_t6to12_pois_r8 | agent | 8 | on | 3376.0 | 6970.7 | 51.6 | 1712.0 | 168.2 | nr |
| agent_p1536_t6to12_pareto_r4 | agent | 4 | on | 549.6 | 1468.5 | 38.6 | 579.0 | 183.2 | nr |
| agent_p1536_t6to12_pareto_r8 | agent | 8 | on | 1466.2 | 2709.6 | 61.9 | 1288.3 | 198.7 | nr |
| agent_p2048_t8to16_pois_r4 | agent | 4 | on | 416.5 | 1608.8 | 25.3 | 404.5 | 67.8 | nr |
| agent_p2048_t8to16_pois_r8 | agent | 8 | on | 847.5 | 3709.9 | 60.9 | 829.1 | 135.4 | nr |
| agent_p2048_t8to16_pareto_r8 | agent | 8 | on | 662.5 | 1686.0 | 62.2 | 813.0 | 172.9 | nr |
| ablation_p1536_radix_on_r2 | ablation | 2 | on | 236.9 | 1756.7 | 27.8 | 298.1 | 104.9 | nr |
| ablation_p1536_radix_on_r8 | ablation | 8 | on | 3159.8 | 4761.7 | 61.1 | 1701.2 | 184.3 | nr |
| ablation_p1536_radix_off_r2 | ablation | 2 | off | 93.6 | 184.9 | 25.9 | 66.0 | 105.2 | nr |
| ablation_p1536_radix_off_r8 | ablation | 8 | off | 138.4 | 210.8 | 45.6 | 86.7 | 281.4 | nr |

Mean output tok/s over the 18 workloads both engines completed: clockwork 253.4,
vLLM 223.8. By category: agent traces 262.9 vs 183.1 (clockwork 1.24x to 1.82x per
workload, from prefix reuse at 0.86 to 0.91 hit rates), sharegpt 328.7 vs 340.7
(vLLM ahead 3.5%, widest at 16 req/s where its deeper batching dominates and shared
prefixes are scarce). vLLM's ITL p50 is consistently lower; clockwork trades
per-token pacing for admission of prefix-cached sequences.

SGLANG: not-run. Its server did not come up in this environment; the notebook prints
the reason and continues, and no substitute numbers are reported.

## Reproducing

`notebooks/bench_t4.ipynb` reproduces every number above on a free Colab T4 without
code changes. The underlying commands, usable directly on any CUDA host:

```
uv run python scripts/serve.py --config configs/qwen2.5-1.5b-instruct.yaml
uv run python scripts/run_bench.py --configs all --base-url http://127.0.0.1:8000 \
  --out bench_results --plots docs/figures
```

`--configs` also accepts comma-separated workload names. The radix-off ablations need
the server relaunched with the prefix cache disabled, since the harness cannot toggle
it per request:

```
uv run python scripts/serve.py --config configs/qwen2.5-1.5b-instruct.yaml \
  --set enable_prefix_cache=false
```

Figures for both engines are in `docs/figures/`.
