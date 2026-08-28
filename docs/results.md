# Results

Measured on a Tesla T4 (sm75, 15.6 GB, CUDA 12.8) by a top to bottom run of
`notebooks/bench_t4.ipynb`, model Qwen/Qwen2.5-1.5B-Instruct in float16, after the
correctness gates in that notebook passed on the same hardware. Raw CSVs and figures are
attached to the GitHub release.

## What these numbers are, and are not

1. The single-turn workloads draw prompts from the real ShareGPT trace
   (`anon8231489123/ShareGPT_Vicuna_unfiltered`, 672837942 bytes, sha256 `35f0e213ce091ed9`),
   sampling 12000 conversations. Their configuration names keep a
   `synthetic_` prefix from an earlier run made before the dataset was wired in.
2. The agent traces are self-designed by this project to model agent-loop traffic. Their
   long shared prefixes are the structure the prefix cache exists to exploit, so the
   agent-workload comparison favors this engine by construction.
3. Both engines are driven by this repository's own harness. No independent benchmark tool
   has corroborated the comparison.

The clockwork matrix ran 2 times; every clockwork cell is the median. vLLM ran once.

## Evaluation protocol

`scripts/run_bench.py` drives a running server over HTTP with streaming requests at
temperature 0.0, writing one CSV row per request plus a `summary.csv` row per workload.
Generation is seeded and deterministic; arrivals follow each workload's Poisson or Pareto
process. TTFT is client-side send to first content chunk, ITL is the gaps between chunks,
both as nearest-rank percentiles. Hit rate is `cached_tokens` over prompt tokens for
clockwork, and Prometheus prefix-cache counters for vLLM, which does not report cached
tokens in its usage payload. Errored requests are excluded from latency statistics.

## Clockwork workload matrix

| workload | kind | req/s | radix | ttft p50 (ms) | ttft p99 (ms) | itl p50 (ms) | itl p99 (ms) | out tok/s | hit rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_singleturn_r1 | singleturn | 1 | on | 111.8 | 1088.1 | 34.0 | 98.3 | 195.2 | 0.059 |
| synthetic_singleturn_r2 | singleturn | 2 | on | 131.2 | 945.6 | 44.7 | 188.1 | 340.8 | 0.059 |
| synthetic_singleturn_r4 | singleturn | 4 | on | 272.7 | 4303.0 | 68.4 | 298.9 | 554.8 | 0.125 |
| synthetic_singleturn_r8 | singleturn | 8 | on | 547.2 | 12623.8 | 66.2 | 371.5 | 590.5 | 0.175 |
| synthetic_singleturn_r16 | singleturn | 16 | on | 676.4 | 17512.0 | 63.3 | 426.5 | 641.1 | 0.116 |
| agent_p1024_t4to8_pois_r2 | agent | 2 | on | 237.6 | 656.3 | 81.0 | 264.5 | 278.0 | 0.891 |
| agent_p1024_t4to8_pois_r4 | agent | 4 | on | 317.8 | 4414.9 | 131.8 | 371.6 | 292.5 | 0.874 |
| agent_p1024_t4to8_pois_r8 | agent | 8 | on | 2507.2 | 20506.6 | 148.0 | 1209.4 | 316.0 | 0.832 |
| agent_p1024_t4to8_pareto_r4 | agent | 4 | on | 252.3 | 711.1 | 86.8 | 339.4 | 228.6 | 0.876 |
| agent_p1536_t6to12_pois_r2 | agent | 2 | on | 325.3 | 1625.0 | 99.8 | 371.8 | 182.5 | 0.889 |
| agent_p1536_t6to12_pois_r4 | agent | 4 | on | 557.5 | 9119.6 | 163.4 | 470.1 | 243.9 | 0.905 |
| agent_p1536_t6to12_pois_r8 | agent | 8 | on | 2858.8 | 28149.8 | 184.6 | 1440.5 | 246.1 | 0.851 |
| agent_p1536_t6to12_pareto_r4 | agent | 4 | on | 973.1 | 15633.1 | 186.6 | 786.1 | 249.2 | 0.893 |
| agent_p1536_t6to12_pareto_r8 | agent | 8 | on | 1800.3 | 21268.4 | 189.0 | 1138.0 | 245.8 | 0.879 |
| agent_p2048_t8to16_pois_r4 | agent | 4 | on | 316.7 | 10917.9 | 72.2 | 354.2 | 154.4 | 0.904 |
| agent_p2048_t8to16_pois_r8 | agent | 8 | on | 2041.9 | 11930.1 | 192.4 | 1334.0 | 225.1 | 0.873 |
| agent_p2048_t8to16_pareto_r8 | agent | 8 | on | 670.4 | 15922.1 | 208.4 | 551.3 | 230.0 | 0.906 |
| ablation_p1536_radix_on_r2 | ablation | 2 | on | 210.2 | 914.2 | 81.8 | 276.1 | 175.4 | 0.914 |
| ablation_p1536_radix_on_r8 | ablation | 8 | on | 2720.8 | 26654.4 | 188.6 | 1246.2 | 240.8 | 0.852 |
| ablation_p1536_radix_off_r2 | ablation | 2 | off | 8809.9 | 30140.8 | 206.5 | 1284.9 | 119.0 | 0.000 |
| ablation_p1536_radix_off_r8 | ablation | 8 | off | 26704.3 | 77989.3 | 209.6 | 1253.6 | 117.9 | 0.000 |

## Radix ablation

Identical traces, server relaunched with the prefix cache off:

| rate | tok/s on | tok/s off | ratio | ttft p50 on (ms) | ttft p50 off (ms) | p50 cut | gpu util on | gpu util off |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 req/s | 175.4 | 119.0 | 1.47x | 210 | 8810 | 98% | 76.8 | 91.3 |
| 8 req/s | 240.8 | 117.9 | 2.04x | 2721 | 26704 | 90% | 88.2 | 94.2 |

Utilization falls when the cache is on because the cache removes redundant prefill: the
device does less work and produces more output.

A cold versus warm experiment isolates the same effect on one trace with no config change.
TTFT p50 falls from 355.1 ms to 202.6 ms
(42.9 percent) and p99 from 2314.6 ms to
438.7 ms (81.0 percent).

## Paged decode kernel

| batch | ctx len | torch (ms) | triton (ms) | speedup |
| --- | --- | --- | --- | --- |
| 1 | 128 | 0.5172 | 0.0607 | 8.52x |
| 1 | 512 | 0.5148 | 0.1602 | 3.21x |
| 1 | 2048 | 0.5175 | 0.6219 | 0.83x |
| 4 | 128 | 0.5566 | 0.0737 | 7.55x |
| 4 | 512 | 0.6764 | 0.2735 | 2.47x |
| 4 | 2048 | 1.6161 | 0.5802 | 2.79x |
| 16 | 128 | 0.6249 | 0.0979 | 6.38x |
| 16 | 512 | 1.6053 | 0.4182 | 3.84x |
| 16 | 2048 | 6.5619 | 1.4058 | 4.67x |
| 64 | 128 | 1.5818 | 0.4079 | 3.88x |
| 64 | 512 | 6.6408 | 1.4359 | 4.62x |
| 64 | 2048 | 26.5465 | 5.3783 | 4.94x |

The Triton kernel wins 11 of 12 shapes, peaking at 8.52x
(batch 1, ctx 128), so `resolve_backend("auto")` selects it.
`results/roofline_decode.csv` records the measured arithmetic intensity behind this.

## vLLM baseline

Same model, dtype, traces, and harness; vLLM v1 on default settings.

| workload | kind | req/s | radix | ttft p50 (ms) | ttft p99 (ms) | itl p50 (ms) | itl p99 (ms) | out tok/s | hit rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_singleturn_r1 | singleturn | 1 | on | 72.3 | 2132.6 | 17.9 | 45.4 | 185.2 | 0.059 |
| synthetic_singleturn_r2 | singleturn | 2 | on | 98.5 | 1473.0 | 21.4 | 74.7 | 318.5 | 0.059 |
| synthetic_singleturn_r4 | singleturn | 4 | on | 139.4 | 857.1 | 32.3 | 218.3 | 649.9 | 0.125 |
| synthetic_singleturn_r8 | singleturn | 8 | on | 286.7 | 2392.3 | 44.3 | 566.9 | 813.2 | 0.175 |
| synthetic_singleturn_r16 | singleturn | 16 | on | 197.3 | 7220.2 | 63.0 | 673.4 | 908.5 | 0.116 |
| agent_p1024_t4to8_pois_r2 | agent | 2 | on | 218.2 | 785.5 | 33.3 | 367.8 | 175.1 | 0.905 |
| agent_p1024_t4to8_pois_r4 | agent | 4 | on | 236.7 | 1332.9 | 30.7 | 294.9 | 218.9 | 0.906 |
| agent_p1024_t4to8_pois_r8 | agent | 8 | on | 1491.1 | 2930.5 | 57.6 | 1136.4 | 273.8 | 0.902 |
| agent_p1024_t4to8_pareto_r4 | agent | 4 | on | 247.4 | 646.9 | 32.1 | 246.2 | 137.3 | 0.899 |
| agent_p1536_t6to12_pois_r2 | agent | 2 | on | 524.3 | 2471.2 | 32.0 | 457.9 | 104.6 | 0.917 |
| agent_p1536_t6to12_pois_r4 | agent | 4 | on | 427.1 | 1315.1 | 35.1 | 472.7 | 177.9 | 0.923 |
| agent_p1536_t6to12_pois_r8 | agent | 8 | on | 3380.9 | 7232.3 | 53.0 | 1719.5 | 164.7 | 0.920 |
| agent_p1536_t6to12_pareto_r4 | agent | 4 | on | 576.6 | 1673.8 | 40.6 | 782.7 | 176.6 | 0.922 |
| agent_p1536_t6to12_pareto_r8 | agent | 8 | on | 1476.8 | 2869.4 | 62.5 | 1222.6 | 192.2 | 0.917 |
| agent_p2048_t8to16_pois_r4 | agent | 4 | on | 427.6 | 1608.4 | 26.2 | 439.6 | 67.8 | 0.920 |
| agent_p2048_t8to16_pois_r8 | agent | 8 | on | 865.5 | 4081.2 | 64.0 | 674.7 | 139.1 | 0.928 |
| agent_p2048_t8to16_pareto_r8 | agent | 8 | on | 823.4 | 1834.1 | 65.5 | 1029.9 | 167.9 | 0.922 |
| ablation_p1536_radix_on_r2 | ablation | 2 | on | 246.9 | 1832.4 | 28.8 | 316.8 | 102.2 | 0.922 |
| ablation_p1536_radix_on_r8 | ablation | 8 | on | 3183.5 | 4970.5 | 60.3 | 1802.6 | 176.8 | 0.918 |
| ablation_p1536_radix_off_r2 | ablation | 2 | off | 101.3 | 212.1 | 26.8 | 70.0 | 103.8 | 0.996 |
| ablation_p1536_radix_off_r8 | ablation | 8 | off | 156.7 | 217.2 | 47.8 | 88.2 | 270.6 | 0.997 |

Mean output tok/s over the 12 agent workloads: clockwork 241.0 versus vLLM
166.3, clockwork faster on 12 of 12 by 1.15x to 2.28x. On the
5 ShareGPT-backed single-turn rates the order reverses: clockwork 464.5 versus
vLLM 575.0, vLLM ahead by 24 percent, and vLLM's ITL p50 is lower nearly
everywhere. A fixed-length session experiment (ignore_eos, 64 tokens per turn) shows vLLM
completing multi-turn sessions faster: pooled mean session time is
89 percent higher for clockwork. Within
clockwork, the prefix cache cuts mean agent-session time 79 percent at 2 req/s and 58
percent at 8 req/s on identical replayed traces.

SGLANG: not-run. Its install failed building the `outlines_core` wheel; the notebook prints
the reason and continues, and no substitute numbers are reported.

## Reproducing

```
uv run python scripts/serve.py --config configs/qwen2.5-1.5b-instruct-cuda.yaml
uv run python scripts/run_bench.py --configs all --base-url http://127.0.0.1:8000 \
  --out bench_results --plots docs/figures
```

The radix-off ablations need `--set enable_prefix_cache=false` on the server.
