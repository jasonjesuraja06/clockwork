# Results

Every performance cell below is the literal TBD. The build host (macOS, Apple M4 Pro)
has no CUDA GPU, and CPU timings are not engine performance. `notebooks/bench_t4.ipynb`
runs the commands in this file unchanged on a CUDA host and fills the tables.

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

Requests that error are counted and excluded from the latency statistics.

## Workload matrix

Model Qwen/Qwen2.5-1.5B-Instruct, one workload per row, names from
`clockwork/bench/configs.py`.

| workload | kind | req/s | radix | ttft p50 (ms) | ttft p99 (ms) | itl p50 (ms) | itl p99 (ms) | out tok/s | hit rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sharegpt_r1 | sharegpt | 1 | on | TBD | TBD | TBD | TBD | TBD | TBD |
| sharegpt_r2 | sharegpt | 2 | on | TBD | TBD | TBD | TBD | TBD | TBD |
| sharegpt_r4 | sharegpt | 4 | on | TBD | TBD | TBD | TBD | TBD | TBD |
| sharegpt_r8 | sharegpt | 8 | on | TBD | TBD | TBD | TBD | TBD | TBD |
| sharegpt_r16 | sharegpt | 16 | on | TBD | TBD | TBD | TBD | TBD | TBD |
| agent_p1024_t4to8_pois_r2 | agent | 2 | on | TBD | TBD | TBD | TBD | TBD | TBD |
| agent_p1024_t4to8_pois_r4 | agent | 4 | on | TBD | TBD | TBD | TBD | TBD | TBD |
| agent_p1024_t4to8_pois_r8 | agent | 8 | on | TBD | TBD | TBD | TBD | TBD | TBD |
| agent_p1024_t4to8_pareto_r4 | agent | 4 | on | TBD | TBD | TBD | TBD | TBD | TBD |
| agent_p1536_t6to12_pois_r2 | agent | 2 | on | TBD | TBD | TBD | TBD | TBD | TBD |
| agent_p1536_t6to12_pois_r4 | agent | 4 | on | TBD | TBD | TBD | TBD | TBD | TBD |
| agent_p1536_t6to12_pois_r8 | agent | 8 | on | TBD | TBD | TBD | TBD | TBD | TBD |
| agent_p1536_t6to12_pareto_r4 | agent | 4 | on | TBD | TBD | TBD | TBD | TBD | TBD |
| agent_p1536_t6to12_pareto_r8 | agent | 8 | on | TBD | TBD | TBD | TBD | TBD | TBD |
| agent_p2048_t8to16_pois_r4 | agent | 4 | on | TBD | TBD | TBD | TBD | TBD | TBD |
| agent_p2048_t8to16_pois_r8 | agent | 8 | on | TBD | TBD | TBD | TBD | TBD | TBD |
| agent_p2048_t8to16_pareto_r8 | agent | 8 | on | TBD | TBD | TBD | TBD | TBD | TBD |
| ablation_p1536_radix_on_r2 | ablation | 2 | on | TBD | TBD | TBD | TBD | TBD | TBD |
| ablation_p1536_radix_off_r2 | ablation | 2 | off | TBD | TBD | TBD | TBD | TBD | TBD |
| ablation_p1536_radix_on_r8 | ablation | 8 | on | TBD | TBD | TBD | TBD | TBD | TBD |
| ablation_p1536_radix_off_r8 | ablation | 8 | off | TBD | TBD | TBD | TBD | TBD | TBD |

Agent workload names encode shared prefix tokens (p), turns per session (t), and the
arrival process; each ablation pair shares a seed, so the on and off runs replay an
identical trace.

## Reproducing

Serve, then run the full matrix against the server:

```
uv run python scripts/serve.py --config configs/qwen2.5-1.5b-instruct.yaml
uv run python scripts/run_bench.py --configs all --base-url http://127.0.0.1:8000 \
  --out bench_results --plots docs/figures
```

Every table in the docs derives from the resulting `bench_results/summary.csv`.
`--configs` also accepts comma-separated workload names. A workload's radix column
describes the server it must run against: the radix-off ablations need the server
relaunched with the prefix cache disabled, since the harness cannot toggle it per
request:

```
uv run python scripts/serve.py --config configs/qwen2.5-1.5b-instruct.yaml \
  --set enable_prefix_cache=false
```

## Baselines

vLLM and SGLang were not run on the build host: neither targets a macOS machine with
no CUDA device, and no number would be comparable anyway. `notebooks/bench_t4.ipynb`
produces the baseline rows on the same hardware as the clockwork rows.

| engine | hardware | ttft p50 (ms) | ttft p99 (ms) | itl p50 (ms) | itl p99 (ms) | out tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| clockwork | TBD | TBD | TBD | TBD | TBD | TBD |
| vLLM | TBD | TBD | TBD | TBD | TBD | TBD |
| SGLang | TBD | TBD | TBD | TBD | TBD | TBD |

Fairness protocol: same model (Qwen/Qwen2.5-1.5B-Instruct), same dtype, same seeded
workload traces driven by the same harness, each engine on its default settings, with
engine versions and the exact launch commands recorded in the notebook output.
