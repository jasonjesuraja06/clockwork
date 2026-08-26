"""Engine gates: paged continuous batching with the radix cache must match HF greedy exactly."""

from __future__ import annotations

import pytest
import torch

from clockwork.config import EngineConfig
from clockwork.engine.llm_engine import LLMEngine
from clockwork.engine.loader import build_tiny_qwen2, tiny_qwen2_hf
from clockwork.engine.sequence import RequestOutput, SamplingParams

BLOCK_SIZE = 4
MAX_STEPS = 500


class IntTokenizer:
    """Trivial reversible tokenizer for tiny-model tests: token id <-> decimal string."""

    def __init__(self, eos_token_id: int | None = None):
        self.eos_token_id = eos_token_id

    def encode(self, text: str) -> list[int]:
        return [int(part) for part in text.split()]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(i) for i in ids)


def make_prompts(vocab_size: int, lengths: list[int], seed: int = 1234) -> list[list[int]]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab_size, (n,), generator=gen).tolist() for n in lengths]


def make_engine(tiny, tokenizer=None, **overrides) -> LLMEngine:
    model, _, config = tiny
    overrides.setdefault("block_size", BLOCK_SIZE)
    overrides.setdefault("num_blocks", 128)
    overrides.setdefault("attention_backend", "torch")
    cfg = EngineConfig.defaults("tiny-qwen2", **overrides)
    return LLMEngine(cfg, model=model, hf_config=config, tokenizer=tokenizer or IntTokenizer())


def hf_greedy(hf_model, prompt: list[int], num_new: int) -> list[int]:
    ids = list(prompt)
    with torch.inference_mode():
        for _ in range(num_new):
            logits = hf_model(
                input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False
            ).logits
            ids.append(int(logits[0, -1].argmax()))
    return ids[len(prompt) :]


def drain(engine: LLMEngine, collect_all: bool = False):
    finals: dict[str, RequestOutput] = {}
    history: list[RequestOutput] = []
    for _ in range(MAX_STEPS):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            history.append(out)
            if out.finished:
                finals[out.request_id] = out
    assert not engine.has_unfinished_requests(), "engine did not drain within MAX_STEPS"
    return (finals, history) if collect_all else finals


@pytest.fixture(scope="module")
def tiny():
    model, config = build_tiny_qwen2(seed=0)
    hf_model = tiny_qwen2_hf(seed=0)
    return model, hf_model, config


def test_single_request_matches_hf(tiny):
    _, hf_model, config = tiny
    engine = make_engine(tiny)
    prompt = make_prompts(config.vocab_size, [11])[0]
    engine.add_request("r0", prompt_token_ids=prompt, sampling_params=SamplingParams(max_tokens=12))
    finals, history = drain(engine, collect_all=True)
    final = finals["r0"]
    assert final.finished
    assert final.finish_reason == "length"
    assert final.token_ids == hf_greedy(hf_model, prompt, 12)
    assert final.num_prompt_tokens == len(prompt)
    assert final.num_generated_tokens == 12
    assert final.prompt_token_ids == prompt
    assert final.text == " ".join(str(t) for t in final.token_ids)
    assert "".join(out.delta_text for out in history) == final.text
    assert [len(out.token_ids) for out in history] == list(range(1, 13))


def test_eight_concurrent_requests_match_hf_alone(tiny):
    # The continuous-batching gate: requests of different lengths and budgets,
    # submitted together and staggered across steps, each exactly equal to the
    # same prompt run alone through HF greedy.
    _, hf_model, config = tiny
    engine = make_engine(tiny, max_num_batched_tokens=32)
    lengths = [5, 9, 12, 16, 7, 11, 3, 14]
    budgets = [4, 9, 6, 11, 5, 8, 7, 10]
    prompts = make_prompts(config.vocab_size, lengths)
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            assert prompts[i] != prompts[j]

    finals: dict[str, RequestOutput] = {}
    for i in range(4):
        engine.add_request(
            f"r{i}",
            prompt_token_ids=prompts[i],
            sampling_params=SamplingParams(max_tokens=budgets[i]),
        )
    for _ in range(2):
        for out in engine.step():
            if out.finished:
                finals[out.request_id] = out
    for i in range(4, 8):
        engine.add_request(
            f"r{i}",
            prompt_token_ids=prompts[i],
            sampling_params=SamplingParams(max_tokens=budgets[i]),
        )
    finals.update(drain(engine))

    for i in range(8):
        expected = hf_greedy(hf_model, prompts[i], budgets[i])
        final = finals[f"r{i}"]
        assert final.token_ids == expected, f"r{i} diverged from HF alone"
        assert final.num_generated_tokens == budgets[i]
        assert final.finish_reason == "length"


def test_radix_second_pass_identical_and_matches_ablation(tiny):
    _, hf_model, config = tiny
    engine = make_engine(tiny)
    prompts = make_prompts(config.vocab_size, [12, 10, 15, 8], seed=77)
    params = SamplingParams(max_tokens=8)

    out1 = engine.generate(prompts, params)
    assert engine.stats()["prefix_cache_hit_tokens"] == 0
    out2 = engine.generate(prompts, params)
    stats = engine.stats()
    assert stats["prefix_cache_hit_tokens"] > 0

    # The tree holds every full prompt block after pass one, and a match never
    # covers the whole prompt, so pass two hits (len - 1) // bs * bs tokens
    # capped at the stored full blocks.
    expected_cached = [
        min((len(p) - 1) // BLOCK_SIZE * BLOCK_SIZE, len(p) // BLOCK_SIZE * BLOCK_SIZE)
        for p in prompts
    ]
    assert expected_cached == [8, 8, 12, 4]
    assert [o.num_cached_tokens for o in out2] == expected_cached
    assert stats["prefix_cache_hit_tokens"] == sum(expected_cached)

    assert [o.token_ids for o in out2] == [o.token_ids for o in out1]
    assert [o.text for o in out2] == [o.text for o in out1]

    cold = make_engine(tiny, enable_prefix_cache=False)
    out3 = cold.generate(prompts, params)
    assert [o.token_ids for o in out3] == [o.token_ids for o in out1]
    assert all(o.num_cached_tokens == 0 for o in out3)
    for prompt, out in zip(prompts, out1, strict=True):
        assert out.token_ids == hf_greedy(hf_model, prompt, 8)


def test_shared_prefix_hit_accounting(tiny):
    _, hf_model, config = tiny
    engine = make_engine(tiny)
    common = make_prompts(config.vocab_size, [16], seed=9)[0]
    tails = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    prompts = [common + tail for tail in tails]
    params = SamplingParams(max_tokens=6)

    # p0 alone first: its prefill inserts the 16-token common prefix, so the
    # three later admissions each hit exactly 4 blocks.
    engine.add_request("p0", prompt_token_ids=prompts[0], sampling_params=params)
    first = engine.step()
    assert [out.request_id for out in first] == ["p0"]
    for i in (1, 2, 3):
        engine.add_request(f"p{i}", prompt_token_ids=prompts[i], sampling_params=params)
    finals = drain(engine)

    stats = engine.stats()
    assert stats["prefix_cache_queries"] == 4
    assert stats["prefix_cache_prompt_tokens"] == 4 * 19
    assert stats["prefix_cache_hit_tokens"] == 3 * 16
    assert stats["prefix_cache_hit_rate"] == 48 / 76
    assert stats["total_cached_tokens"] == 48
    assert finals["p0"].num_cached_tokens == 0
    for i in (1, 2, 3):
        assert finals[f"p{i}"].num_cached_tokens == 16

    for i in range(4):
        assert finals[f"p{i}"].token_ids == hf_greedy(hf_model, prompts[i], 6)

    off = make_engine(tiny, enable_prefix_cache=False)
    out_off = off.generate(prompts, params)
    assert [o.token_ids for o in out_off] == [finals[f"p{i}"].token_ids for i in range(4)]


def test_memory_pressure_preempts_and_matches_hf(tiny):
    # 8 blocks cannot hold three sequences growing to 16 tokens each, so the
    # scheduler must preempt by recompute; outputs must still be exact.
    _, hf_model, config = tiny
    engine = make_engine(tiny, num_blocks=8, watermark=0.0)
    prompts = make_prompts(config.vocab_size, [8, 8, 8], seed=5)
    params = SamplingParams(max_tokens=8)
    for i, prompt in enumerate(prompts):
        engine.add_request(f"m{i}", prompt_token_ids=prompt, sampling_params=params)
    finals = drain(engine)

    assert engine.stats()["num_preempted_total"] > 0
    for i, prompt in enumerate(prompts):
        final = finals[f"m{i}"]
        assert final.token_ids == hf_greedy(hf_model, prompt, 8), f"m{i} diverged after preemption"
        assert final.finish_reason == "length"
        assert final.num_generated_tokens == 8


def test_sampling_seeded_reproducible_across_engines(tiny):
    _, _, config = tiny
    prompts = make_prompts(config.vocab_size, [10], seed=3)
    params = SamplingParams(max_tokens=10, temperature=0.8, top_k=50, top_p=0.9, seed=1234)
    out1 = make_engine(tiny).generate(prompts, params)
    out2 = make_engine(tiny).generate(prompts, params)
    assert out1[0].token_ids == out2[0].token_ids
    assert len(out1[0].token_ids) == 10
    assert all(0 <= t < config.vocab_size for t in out1[0].token_ids)

    hot_a = SamplingParams(max_tokens=10, temperature=1.5, seed=1)
    hot_b = SamplingParams(max_tokens=10, temperature=1.5, seed=2)
    diff_a = make_engine(tiny).generate(prompts, hot_a)
    diff_b = make_engine(tiny).generate(prompts, hot_b)
    assert diff_a[0].token_ids != diff_b[0].token_ids


def test_eos_stop_token_ids_and_ignore_eos(tiny):
    _, hf_model, config = tiny
    prompt = make_prompts(config.vocab_size, [10], seed=1)[0]
    greedy = hf_greedy(hf_model, prompt, 8)
    k = next(i for i in range(1, len(greedy)) if greedy[i] not in greedy[:i])
    eos = greedy[k]

    engine = make_engine(tiny, tokenizer=IntTokenizer(eos_token_id=eos))
    out = engine.generate([prompt], SamplingParams(max_tokens=8))[0]
    assert out.token_ids == greedy[: k + 1]
    assert out.finish_reason == "stop"
    assert out.text == " ".join(str(t) for t in greedy[:k])

    ignore = make_engine(tiny, tokenizer=IntTokenizer(eos_token_id=eos))
    out_ignore = ignore.generate([prompt], SamplingParams(max_tokens=8, ignore_eos=True))[0]
    assert out_ignore.token_ids == greedy
    assert out_ignore.finish_reason == "length"

    by_id = make_engine(tiny)
    out_id = by_id.generate([prompt], SamplingParams(max_tokens=8, stop_token_ids=[eos]))[0]
    assert out_id.token_ids == greedy[: k + 1]
    assert out_id.finish_reason == "stop"


def test_stop_strings_max_tokens_and_max_model_len(tiny):
    _, hf_model, config = tiny
    prompt = make_prompts(config.vocab_size, [10], seed=13)[0]
    greedy = hf_greedy(hf_model, prompt, 10)

    stop_str = f"{greedy[2]} {greedy[3]}"
    expected_text = None
    expected_count = None
    text = ""
    for i, token in enumerate(greedy):
        text = str(token) if i == 0 else f"{text} {token}"
        pos = text.find(stop_str)
        if pos >= 0:
            expected_text = text[:pos]
            expected_count = i + 1
            break
    assert expected_text is not None, "stop string never appears in the greedy text"

    engine = make_engine(tiny)
    out = engine.generate([prompt], SamplingParams(max_tokens=10, stop=[stop_str]))[0]
    assert out.finish_reason == "stop"
    assert out.text == expected_text
    assert stop_str not in out.text
    assert out.token_ids == greedy[:expected_count]

    exact = make_engine(tiny)
    out_exact = exact.generate([prompt], SamplingParams(max_tokens=5))[0]
    assert out_exact.token_ids == greedy[:5]
    assert out_exact.finish_reason == "length"

    capped = make_engine(tiny, max_model_len=16)
    out_cap = capped.generate([prompt], SamplingParams(max_tokens=100))[0]
    assert out_cap.token_ids == greedy[:6]
    assert out_cap.finish_reason == "length"
