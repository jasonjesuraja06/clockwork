"""Async engine gates: concurrent streaming, mid-stream abort, and sync equivalence."""

from __future__ import annotations

import asyncio

import pytest

from clockwork.config import EngineConfig
from clockwork.engine.async_engine import AsyncLLMEngine
from clockwork.engine.llm_engine import LLMEngine
from clockwork.engine.loader import build_tiny_qwen2
from clockwork.engine.sequence import SamplingParams
from tests.test_llm_engine import BLOCK_SIZE, IntTokenizer, make_prompts

TIMEOUT = 120.0


@pytest.fixture(scope="module")
def tiny():
    model, config = build_tiny_qwen2(seed=0)
    return model, config


def make_config(**overrides) -> EngineConfig:
    overrides.setdefault("block_size", BLOCK_SIZE)
    overrides.setdefault("num_blocks", 128)
    overrides.setdefault("attention_backend", "torch")
    return EngineConfig.defaults("tiny-qwen2", **overrides)


def make_async_engine(tiny, **overrides) -> AsyncLLMEngine:
    model, config = tiny
    return AsyncLLMEngine(
        make_config(**overrides), model=model, hf_config=config, tokenizer=IntTokenizer()
    )


def sync_final_tokens(tiny, prompts, params) -> list[list[int]]:
    model, config = tiny
    engine = LLMEngine(make_config(), model=model, hf_config=config, tokenizer=IntTokenizer())
    return [out.token_ids for out in engine.generate(prompts, params)]


async def collect(engine: AsyncLLMEngine, request_id, prompt, params):
    outputs = []
    async for out in engine.generate(request_id, prompt, params):
        outputs.append(out)
    return outputs


async def test_concurrent_streams_grow_and_match_sync(tiny):
    _, config = tiny
    prompts = make_prompts(config.vocab_size, [6, 10, 13, 8], seed=21)
    params = SamplingParams(max_tokens=8)
    expected = sync_final_tokens(tiny, prompts, params)

    engine = make_async_engine(tiny)
    engine.start()
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(collect(engine, f"a{i}", prompts[i], params) for i in range(len(prompts)))
            ),
            timeout=TIMEOUT,
        )
    finally:
        await engine.shutdown()

    for i, streamed in enumerate(results):
        assert [len(out.token_ids) for out in streamed] == list(range(1, 9))
        text = ""
        for out in streamed:
            text += out.delta_text
            assert out.text == text
            assert out.request_id == f"a{i}"
        assert not any(out.finished for out in streamed[:-1])
        final = streamed[-1]
        assert final.finished
        assert final.finish_reason == "length"
        assert final.token_ids == expected[i]
    assert not engine.engine.has_unfinished_requests()


async def test_early_consumer_exit_aborts_and_others_finish(tiny):
    _, config = tiny
    prompts = make_prompts(config.vocab_size, [7, 11, 9], seed=33)
    params = SamplingParams(max_tokens=10)
    expected = sync_final_tokens(tiny, prompts, params)

    engine = make_async_engine(tiny)
    try:

        async def quitter():
            stream = engine.generate("victim", prompts[0], params)
            received = []
            async for out in stream:
                received.append(out)
                if len(received) == 2:
                    break
            await stream.aclose()
            return received

        results = await asyncio.wait_for(
            asyncio.gather(
                quitter(),
                collect(engine, "b1", prompts[1], params),
                collect(engine, "b2", prompts[2], params),
            ),
            timeout=TIMEOUT,
        )
    finally:
        await engine.shutdown()

    received = results[0]
    assert len(received) == 2
    assert not any(out.finished for out in received)
    for i in (1, 2):
        final = results[i][-1]
        assert final.finished
        assert final.token_ids == expected[i], f"b{i} diverged after the abort"
    assert not engine.engine.has_unfinished_requests()


async def test_explicit_abort_ends_stream_and_engine_serves_next(tiny):
    _, config = tiny
    prompts = make_prompts(config.vocab_size, [8, 12], seed=44)
    slow = SamplingParams(max_tokens=64)
    fast = SamplingParams(max_tokens=6)
    expected = sync_final_tokens(tiny, [prompts[1]], fast)

    engine = make_async_engine(tiny)
    try:
        first_out = asyncio.Event()
        received = []

        async def consumer():
            async for out in engine.generate("long", prompts[0], slow):
                received.append(out)
                first_out.set()
            return received

        task = asyncio.create_task(consumer())
        await asyncio.wait_for(first_out.wait(), timeout=TIMEOUT)
        await engine.abort("long")
        await asyncio.wait_for(task, timeout=TIMEOUT)
        assert received
        assert not any(out.finished for out in received)
        assert len(received) < 64

        streamed = await asyncio.wait_for(
            collect(engine, "after", prompts[1], fast), timeout=TIMEOUT
        )
        assert streamed[-1].finished
        assert streamed[-1].token_ids == expected[0]
    finally:
        await engine.shutdown()
    assert not engine.engine.has_unfinished_requests()
