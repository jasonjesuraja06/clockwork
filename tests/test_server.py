"""Server gates: OpenAI endpoints over the async engine with the tiny model."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from clockwork.config import EngineConfig
from clockwork.engine.async_engine import AsyncLLMEngine
from clockwork.engine.llm_engine import LLMEngine
from clockwork.engine.loader import build_tiny_qwen2
from clockwork.engine.sequence import RequestOutput, SamplingParams
from clockwork.server.app import build_app
from tests.test_llm_engine import BLOCK_SIZE, IntTokenizer, make_prompts

MODEL = "tiny-qwen2"
TIMEOUT = 120.0
ROLE_IDS = {"system": 1, "user": 2, "assistant": 3}


class ChatTokenizer(IntTokenizer):
    """IntTokenizer plus a minimal chat template: a role tag token before each message."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        ids: list[int] = []
        for message in messages:
            ids.append(ROLE_IDS.get(message["role"], 4))
            ids.extend(self.encode(message["content"]))
        if add_generation_prompt:
            ids.append(ROLE_IDS["assistant"])
        if tokenize:
            return ids
        return self.decode(ids)


@pytest.fixture(scope="module")
def tiny():
    model, config = build_tiny_qwen2(seed=0)
    return model, config


def make_config(**overrides) -> EngineConfig:
    overrides.setdefault("block_size", BLOCK_SIZE)
    overrides.setdefault("num_blocks", 128)
    overrides.setdefault("attention_backend", "torch")
    return EngineConfig.defaults(MODEL, **overrides)


@asynccontextmanager
async def serve(tiny, **overrides):
    model, config = tiny
    cfg = make_config(**overrides)
    engine = AsyncLLMEngine(cfg, model=model, hf_config=config, tokenizer=ChatTokenizer())
    app = build_app(cfg, engine=engine)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=TIMEOUT
        ) as client:
            yield client


def sync_chat(tiny, messages, params: SamplingParams) -> RequestOutput:
    model, config = tiny
    tokenizer = ChatTokenizer()
    engine = LLMEngine(make_config(), model=model, hf_config=config, tokenizer=tokenizer)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return engine.generate([ids], params)[0]


def sync_completion(tiny, prompt_ids, params: SamplingParams) -> RequestOutput:
    model, config = tiny
    engine = LLMEngine(make_config(), model=model, hf_config=config, tokenizer=ChatTokenizer())
    return engine.generate([list(prompt_ids)], params)[0]


def make_messages(vocab_size: int, lengths: list[int], seed: int) -> list[list[dict]]:
    prompts = make_prompts(vocab_size, lengths, seed=seed)
    return [[{"role": "user", "content": " ".join(str(t) for t in prompt)}] for prompt in prompts]


def parse_sse(body: str) -> list[dict]:
    parts = [part for part in body.split("\n\n") if part]
    assert all(part.startswith("data: ") for part in parts), parts
    assert parts[-1] == "data: [DONE]"
    return [json.loads(part[len("data: ") :]) for part in parts[:-1]]


def stream_text(events: list[dict]) -> str:
    return "".join(e["choices"][0]["delta"].get("content", "") for e in events)


async def test_chat_completion_matches_sync_engine(tiny):
    _, config = tiny
    messages = make_messages(config.vocab_size, [9], seed=101)[0]
    expected = sync_chat(tiny, messages, SamplingParams(max_tokens=8))
    async with serve(tiny) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": MODEL, "messages": messages, "max_tokens": 8, "temperature": 0.0},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert body["model"] == MODEL
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == expected.text
    assert choice["finish_reason"] == "length"
    usage = body["usage"]
    assert usage["prompt_tokens"] == expected.num_prompt_tokens
    assert usage["completion_tokens"] == 8
    assert usage["total_tokens"] == expected.num_prompt_tokens + 8
    assert usage["prompt_tokens_details"]["cached_tokens"] == 0


async def test_chat_stream_chunk_discipline(tiny):
    _, config = tiny
    messages = make_messages(config.vocab_size, [11], seed=202)[0]
    expected = sync_chat(tiny, messages, SamplingParams(max_tokens=8))
    async with serve(tiny) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": 8,
                "temperature": 0.0,
                "stream": True,
            },
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.text)
    assert len(events) >= 3

    first = events[0]
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["role"] == "assistant"
    assert first["choices"][0]["delta"].get("content", "") == ""

    last = events[-1]
    assert last["choices"][0]["finish_reason"] == "length"
    assert last["choices"][0]["delta"].get("content") is None
    assert last["usage"]["completion_tokens"] == 8
    assert last["usage"]["prompt_tokens"] == expected.num_prompt_tokens

    for event in events[:-1]:
        assert event["choices"][0].get("finish_reason") is None
        assert "usage" not in event
    for event in events[1:-1]:
        assert event["choices"][0]["delta"]["content"] != ""
    assert stream_text(events) == expected.text
    assert len({e["id"] for e in events}) == 1
    assert all(e["model"] == MODEL for e in events)


async def test_eight_concurrent_clients_match_solo_runs(tiny):
    _, config = tiny
    lengths = [5, 9, 12, 16, 7, 11, 3, 14]
    budgets = [4, 9, 6, 11, 5, 8, 7, 10]
    all_messages = make_messages(config.vocab_size, lengths, seed=55)
    expected = [
        sync_chat(tiny, messages, SamplingParams(max_tokens=budget))
        for messages, budget in zip(all_messages, budgets, strict=True)
    ]

    async with serve(tiny) as client:

        async def one(i: int):
            payload = {
                "model": MODEL,
                "messages": all_messages[i],
                "max_tokens": budgets[i],
                "temperature": 0.0,
            }
            if i % 2:
                payload["stream"] = True
                resp = await client.post("/v1/chat/completions", json=payload)
                assert resp.status_code == 200
                events = parse_sse(resp.text)
                last = events[-1]["choices"][0]
                return stream_text(events), last["finish_reason"], events[-1]["usage"]
            resp = await client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200
            body = resp.json()
            choice = body["choices"][0]
            return choice["message"]["content"], choice["finish_reason"], body["usage"]

        results = await asyncio.wait_for(
            asyncio.gather(*(one(i) for i in range(8))), timeout=TIMEOUT
        )

    for i, (text, finish_reason, usage) in enumerate(results):
        assert text == expected[i].text, f"client {i} diverged from its solo run"
        assert finish_reason == "length"
        assert usage["completion_tokens"] == budgets[i]
        assert usage["prompt_tokens"] == expected[i].num_prompt_tokens


async def test_completion_stop_sequence(tiny):
    _, config = tiny
    prompt_ids = make_prompts(config.vocab_size, [10], seed=13)[0]
    prompt = " ".join(str(t) for t in prompt_ids)
    base = sync_completion(tiny, prompt_ids, SamplingParams(max_tokens=10))
    stop_str = f"{base.token_ids[2]} {base.token_ids[3]}"
    expected = sync_completion(tiny, prompt_ids, SamplingParams(max_tokens=10, stop=[stop_str]))
    assert expected.finish_reason == "stop"

    async with serve(tiny) as client:
        for stop in (stop_str, [stop_str]):
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "max_tokens": 10,
                    "temperature": 0.0,
                    "stop": stop,
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["id"].startswith("cmpl-")
            assert body["object"] == "text_completion"
            choice = body["choices"][0]
            assert choice["text"] == expected.text
            assert stop_str not in choice["text"]
            assert choice["finish_reason"] == "stop"
            assert body["usage"]["completion_tokens"] == expected.num_generated_tokens


async def test_completion_max_tokens_and_stream(tiny):
    _, config = tiny
    prompt_ids = make_prompts(config.vocab_size, [8], seed=71)[0]
    expected = sync_completion(tiny, prompt_ids, SamplingParams(max_tokens=5))
    async with serve(tiny) as client:
        resp = await client.post(
            "/v1/completions",
            json={
                "model": MODEL,
                "prompt": list(prompt_ids),
                "max_tokens": 5,
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["text"] == expected.text
        assert body["choices"][0]["finish_reason"] == "length"
        assert body["usage"]["completion_tokens"] == 5
        assert body["usage"]["prompt_tokens"] == len(prompt_ids)
        assert body["usage"]["total_tokens"] == len(prompt_ids) + 5

        streamed = await client.post(
            "/v1/completions",
            json={
                "model": MODEL,
                "prompt": list(prompt_ids),
                "max_tokens": 5,
                "temperature": 0.0,
                "stream": True,
            },
        )
        assert streamed.status_code == 200
        events = parse_sse(streamed.text)
        assert all(e["object"] == "text_completion" for e in events)
        assert "".join(e["choices"][0]["text"] for e in events) == expected.text
        assert events[-1]["choices"][0]["finish_reason"] == "length"
        assert events[-1]["usage"]["completion_tokens"] == 5
        for event in events[:-1]:
            assert event["choices"][0].get("finish_reason") is None


async def test_usage_reports_cached_tokens_on_repeat(tiny):
    _, config = tiny
    # Content of 14 tokens plus two template tokens gives a 16 token prompt; the
    # radix match never covers the whole prompt, so the repeat hits 12 tokens.
    messages = make_messages(config.vocab_size, [14], seed=88)[0]
    payload = {"model": MODEL, "messages": messages, "max_tokens": 6, "temperature": 0.0}
    async with serve(tiny) as client:
        first = await client.post("/v1/chat/completions", json=payload)
        second = await client.post("/v1/chat/completions", json=payload)
        metrics = (await client.get("/metrics")).json()
    body1 = first.json()
    body2 = second.json()
    assert body1["usage"]["prompt_tokens"] == 16
    assert body1["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert body2["usage"]["prompt_tokens_details"]["cached_tokens"] == 12
    assert body2["choices"][0]["message"]["content"] == body1["choices"][0]["message"]["content"]
    assert metrics["prefix_cache_queries"] == 2
    assert metrics["prefix_cache_hit_tokens"] == 12
    assert metrics["total_cached_tokens"] == 12


async def test_seeded_sampling_reproducible_across_requests(tiny):
    _, config = tiny
    messages = make_messages(config.vocab_size, [10], seed=3)[0]
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 10,
        "temperature": 0.8,
        "top_p": 0.9,
        "seed": 1234,
    }
    async with serve(tiny) as client:
        first = await client.post("/v1/chat/completions", json=payload)
        second = await client.post("/v1/chat/completions", json=payload)
    assert (
        first.json()["choices"][0]["message"]["content"]
        == (second.json()["choices"][0]["message"]["content"])
    )
    assert first.json()["usage"]["completion_tokens"] == 10


async def test_health_models_and_metrics_shape(tiny):
    async with serve(tiny) as client:
        health = await client.get("/health")
        models = await client.get("/v1/models")
        metrics = await client.get("/metrics")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    body = models.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["id"] == MODEL
    assert body["data"][0]["object"] == "model"

    stats = metrics.json()
    assert stats["model"] == MODEL
    assert stats["attention_backend"] == "torch"
    for key in (
        "num_waiting",
        "num_running",
        "num_preempted_total",
        "num_scheduled_batches",
        "total_prompt_tokens",
        "total_cached_tokens",
        "prefix_cache_queries",
        "prefix_cache_hit_tokens",
        "prefix_cache_prompt_tokens",
        "prefix_cache_hit_rate",
        "num_free_blocks",
    ):
        assert key in stats, f"metrics missing {key}"


async def test_bad_model_and_n_return_400(tiny):
    messages = [{"role": "user", "content": "1 2 3"}]
    async with serve(tiny) as client:
        wrong_model_chat = await client.post(
            "/v1/chat/completions", json={"model": "gpt-4", "messages": messages}
        )
        wrong_model_completion = await client.post(
            "/v1/completions", json={"model": "gpt-4", "prompt": "1 2 3"}
        )
        multi_chat = await client.post(
            "/v1/chat/completions", json={"model": MODEL, "messages": messages, "n": 2}
        )
        multi_completion = await client.post(
            "/v1/completions", json={"model": MODEL, "prompt": "1 2 3", "n": 2}
        )
        works_after = await client.post(
            "/v1/chat/completions",
            json={"model": MODEL, "messages": messages, "max_tokens": 2, "temperature": 0.0},
        )
    for resp in (wrong_model_chat, wrong_model_completion, multi_chat, multi_completion):
        assert resp.status_code == 400
        error = resp.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["message"]
    assert works_after.status_code == 200
    assert works_after.json()["usage"]["completion_tokens"] == 2


async def test_prompt_over_admission_budget_returns_400(tiny):
    async with serve(tiny, max_num_batched_tokens=8) as client:
        too_long = await client.post(
            "/v1/completions",
            json={"model": MODEL, "prompt": list(range(1, 13)), "max_tokens": 2},
        )
        fits = await client.post(
            "/v1/completions",
            json={"model": MODEL, "prompt": list(range(1, 7)), "max_tokens": 2},
        )
    assert too_long.status_code == 400
    assert "max_num_batched_tokens" in too_long.json()["error"]["message"]
    assert fits.status_code == 200


async def test_ignore_eos_generates_exactly_max_tokens(tiny):
    async with serve(tiny) as client:
        resp = await client.post(
            "/v1/completions",
            json={
                "model": MODEL,
                "prompt": list(range(1, 7)),
                "max_tokens": 4,
                "ignore_eos": True,
                "temperature": 0.0,
            },
        )
    assert resp.status_code == 200
    assert resp.json()["usage"]["completion_tokens"] == 4
