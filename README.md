# clockwork

An LLM inference engine with continuous batching, a paged KV cache, and a radix prefix
cache behind an OpenAI-compatible API.

[![ci](https://github.com/jasonjesuraja06/clockwork/actions/workflows/ci.yml/badge.svg)](https://github.com/jasonjesuraja06/clockwork/actions/workflows/ci.yml)

## Motivation

Agent loops resend the same system prompt, tool schemas, and conversation history on
every turn, so most prompt tokens reaching the server carry KV that was already computed.
Continuous batching keeps the device busy across requests of very different lengths,
paged KV storage turns cache memory into a pool of fixed-size blocks that can be shared
and reclaimed per block, and a radix prefix cache resolves the repeated prefix to a
block-table lookup instead of a prefill. The combination targets time to first token on
exactly the traffic agents generate.

## Architecture

```
client
  |  POST /v1/chat/completions (SSE streaming)
  v
server (FastAPI) --> AsyncLLMEngine --> LLMEngine.step()
                                            |
        +-----------------------------------+
        v
    scheduler <----- match / release ------> radix prefix cache
        |  decode first, FCFS admission,         |
        |  preemption by recompute               |  incref + lock matched blocks
        v                                        v
    block manager ------------------------> block allocator (refcounts, CoW, LIFO free list)
        |  block tables, slot mappings
        v
    model runner -------------------------> paged KV cache
        |  per-sequence prefill,                 [num_blocks, block_size, kv_heads, head_dim]
        |  batched decode
        v
    Qwen2 model --> attention kernels (torch reference, Triton paged decode)
```

## Measured results

Correctness, measured on the build host (float32, cpu):

| check | value |
| --- | --- |
| tiny-model max abs logit diff vs Hugging Face | 2.98e-07 |
| paged prefill vs dense attention, max abs diff | 0.0 |
| paged decode vs dense attention, max abs diff | 2.05e-07 |
| Qwen2.5-1.5B-Instruct greedy decoding vs Hugging Face | PASS, exact token match |

Performance, measured on a Tesla T4 (float16, CUDA 12.8) by a top to bottom run of
`notebooks/bench_t4.ipynb`:

| measurement | value |
| --- | --- |
| Triton vs torch paged decode | faster on 11 of 12 shapes, up to 9.4x |
| radix ablation, identical agent trace | 3.78x to 3.98x output tok/s |
| agent-trace prefix hit rate | 0.86 to 0.91 |
| mean output tok/s vs vLLM, 9 agent workloads | 262.9 vs 183.1 (1.24x to 1.82x each) |
| mean output tok/s vs vLLM, 5 sharegpt workloads | 328.7 vs 340.7 |
| peak output tok/s | 523.6 (sharegpt at 16 req/s) |

Full tables, the fairness protocol, and known limits are in `docs/results.md`.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run pytest -q -m "not slow"
uv run pytest tests/test_hf_equivalence.py -q -m slow
uv run clockwork-serve --config configs/qwen2.5-1.5b-instruct.yaml
```

The first pytest command downloads nothing and covers what CI runs (CI additionally
deselects gpu-marked tests, which skip without CUDA); the second downloads
Qwen2.5-1.5B-Instruct and runs the real-model exact-match gate. With the server up:

```
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Name the largest planet in the solar system."}],
    "max_tokens": 32
  }'
```

## Evaluation protocol and limitations

The equivalence gate loads identical weights into clockwork and Hugging Face
transformers, runs both in float32 on cpu, and requires greedy decoding to produce the
same token at every step; on the tiny config, full logits must also agree within atol
and rtol 1e-5. CI runs the gate on a tiny random-weight Qwen2 config, and the slow
marker runs it on the real Qwen2.5-1.5B-Instruct checkpoint. The bench harness drives seeded workloads against the
server over HTTP and measures TTFT and inter-token latency percentiles, output tokens
per second, and radix hit rate from the server's own usage accounting; the protocol and
workload matrix are in `docs/results.md`, the internals in `docs/design.md`.

Limitations: one model per process on a single GPU, no speculative decoding, no tensor
parallelism, no quantization. Prefill is per sequence without chunking, so admissible
prompts are capped at `max_num_batched_tokens` (4096 in the shipped config); the server
rejects longer prompts with a 400. The T4 tables in `docs/results.md` were measured
with the earlier 2048 budget, where three p2048 workloads were inadmissible. On the build host
(macOS, no CUDA) the Triton kernel is validated by a line-for-line torch
transliteration; on the T4 the gpu-marked kernel tests pass against the torch reference.

## License

MIT
