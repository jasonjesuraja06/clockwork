"""Correctness gate: greedy decoding must match the Hugging Face reference exactly."""

from __future__ import annotations

import gc
import os

import pytest
import torch

from clockwork.engine.attn_metadata import AttentionMetadata
from clockwork.engine.loader import build_tiny_qwen2, tiny_qwen2_hf

NUM_NEW_TOKENS = 24
RTOL = 1e-5
ATOL = 1e-5
PROMPT_LENGTHS = (1, 3, 9, 17)
REAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _make_prompts(vocab_size: int) -> list[list[int]]:
    gen = torch.Generator().manual_seed(1234)
    return [torch.randint(0, vocab_size, (n,), generator=gen).tolist() for n in PROMPT_LENGTHS]


def _all_position_logits(model, token_ids: list[int]) -> torch.Tensor:
    num_tokens = len(token_ids)
    metadata = AttentionMetadata(
        is_prefill=True,
        slot_mapping=torch.empty(0, dtype=torch.int64),
        block_tables=None,
        ctx_lens=None,
        query_lens=[num_tokens],
        seq_block_tables=[[]],
        logits_indices=torch.arange(num_tokens),
    )
    with torch.inference_mode():
        return model(
            torch.tensor(token_ids, dtype=torch.long),
            torch.arange(num_tokens),
            None,
            metadata,
        )


def _hf_all_position_logits(hf_model, token_ids: list[int]) -> torch.Tensor:
    with torch.inference_mode():
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        return hf_model(input_ids=input_ids, use_cache=False).logits[0]


def _assert_logits_match(ours: torch.Tensor, theirs: torch.Tensor) -> None:
    assert ours.shape == theirs.shape, f"logit shape mismatch: {ours.shape} vs {theirs.shape}"
    max_diff = (ours - theirs).abs().max().item()
    assert torch.allclose(ours, theirs, rtol=RTOL, atol=ATOL), f"max abs logit diff {max_diff}"


def _greedy(model, prompt: list[int], num_new: int) -> list[int]:
    ids = list(prompt)
    with torch.inference_mode():
        for _ in range(num_new):
            logits = model(
                torch.tensor(ids, dtype=torch.long),
                torch.arange(len(ids)),
                None,
                None,
            )
            ids.append(int(logits[-1].argmax()))
    return ids[len(prompt) :]


def _hf_greedy(hf_model, prompt: list[int], num_new: int) -> list[int]:
    ids = list(prompt)
    with torch.inference_mode():
        for _ in range(num_new):
            logits = hf_model(
                input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False
            ).logits
            ids.append(int(logits[0, -1].argmax()))
    return ids[len(prompt) :]


@pytest.fixture(scope="module")
def tiny_pair():
    model, config = build_tiny_qwen2(seed=0)
    hf_model = tiny_qwen2_hf(seed=0)
    return model, hf_model, config


@pytest.fixture(scope="module")
def greedy_continuations(tiny_pair):
    model, hf_model, config = tiny_pair
    prompts = _make_prompts(config.vocab_size)
    ours = [_greedy(model, prompt, NUM_NEW_TOKENS) for prompt in prompts]
    theirs = [_hf_greedy(hf_model, prompt, NUM_NEW_TOKENS) for prompt in prompts]
    return prompts, ours, theirs


def test_tiny_logit_equivalence(tiny_pair):
    model, hf_model, config = tiny_pair
    prompt = _make_prompts(config.vocab_size)[2]
    ours = _all_position_logits(model, prompt)
    theirs = _hf_all_position_logits(hf_model, prompt)
    _assert_logits_match(ours, theirs)
    assert torch.equal(ours.argmax(dim=-1), theirs.argmax(dim=-1))


def test_tiny_multi_step_greedy_exact(greedy_continuations):
    prompts, ours, theirs = greedy_continuations
    for prompt, our_tokens, hf_tokens in zip(prompts, ours, theirs, strict=True):
        assert our_tokens == hf_tokens, (
            f"greedy divergence on prompt {prompt}: ours={our_tokens} hf={hf_tokens}"
        )


def test_hand_rolled_hf_greedy_matches_generate(tiny_pair, greedy_continuations):
    _, hf_model, _ = tiny_pair
    prompts, _, theirs = greedy_continuations
    prompt = prompts[-1]
    with torch.inference_mode():
        generated = hf_model.generate(
            torch.tensor([prompt], dtype=torch.long),
            max_new_tokens=NUM_NEW_TOKENS,
            do_sample=False,
        )
    assert generated[0, len(prompt) :].tolist() == theirs[-1]


def test_gate_is_not_vacuous(greedy_continuations):
    prompts, ours, theirs = greedy_continuations
    for our_tokens, hf_tokens in zip(ours, theirs, strict=True):
        assert len(our_tokens) == NUM_NEW_TOKENS
        assert len(hf_tokens) == NUM_NEW_TOKENS
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            assert prompts[i] != prompts[j]
    distinct = {tuple(tokens) for tokens in ours}
    assert len(distinct) >= 2, "every prompt produced the same continuation"


def test_negative_control_perturbed_weight_fails(tiny_pair):
    _, hf_model, config = tiny_pair
    model, _ = build_tiny_qwen2(seed=0)
    prompt = _make_prompts(config.vocab_size)[2]
    _assert_logits_match(
        _all_position_logits(model, prompt), _hf_all_position_logits(hf_model, prompt)
    )
    with torch.no_grad():
        model.model.layers[1].self_attn.q_proj.weight.add_(1e-3)
    with pytest.raises(AssertionError):
        _assert_logits_match(
            _all_position_logits(model, prompt), _hf_all_position_logits(hf_model, prompt)
        )


def test_half_precision_cast_keeps_inv_freq_float32():
    model, _ = build_tiny_qwen2(seed=0)
    reference = model.model.inv_freq.clone()
    assert reference.dtype == torch.float32
    for dtype in (torch.float16, torch.bfloat16):
        cast_model, _ = build_tiny_qwen2(seed=0)
        cast_model.to(dtype=dtype)
        inv_freq = cast_model.model.inv_freq
        assert inv_freq.dtype == torch.float32
        assert torch.equal(inv_freq, reference)
        positions = torch.tensor([0, 1, 4000])
        cos, sin = cast_model.model.rope_cos_sin(positions, dtype)
        ref_cos, ref_sin = model.model.rope_cos_sin(positions, torch.float32)
        # Only output rounding remains; the degraded-inv_freq error was 0.11.
        assert torch.allclose(cos.float(), ref_cos, atol=8e-3)
        assert torch.allclose(sin.float(), ref_sin, atol=8e-3)


@pytest.mark.slow
def test_real_model_greedy_exact_match():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from clockwork.config import ModelConfig
    from clockwork.engine.loader import load_model

    try:
        tokenizer = AutoTokenizer.from_pretrained(REAL_MODEL)
        hf_model = AutoModelForCausalLM.from_pretrained(REAL_MODEL, dtype=torch.float32)
    except Exception as exc:
        if os.environ.get("HF_HUB_OFFLINE", "0") not in ("", "0"):
            pytest.skip(f"{REAL_MODEL} not cached locally and HF_HUB_OFFLINE is set: {exc}")
        raise
    hf_model.eval()

    chat_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Name the largest planet in the solar system."}],
        add_generation_prompt=True,
        tokenize=True,
    )
    if not isinstance(chat_ids, list):
        chat_ids = chat_ids["input_ids"]
    prompts = [
        chat_ids,
        tokenizer.encode("The capital of France is"),
        tokenizer.encode("def fibonacci(n):"),
    ]
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            assert prompts[i] != prompts[j]
    # Reference outputs first, then free the HF copy: two fp32 copies of the 1.5B
    # model exceed the 12.7 GB RAM of a free Colab host and draw the OOM killer.
    references = [_hf_greedy(hf_model, prompt, 16) for prompt in prompts]
    del hf_model
    gc.collect()

    cfg = ModelConfig(model=REAL_MODEL, dtype="float32", device="cpu")
    model, _, _ = load_model(cfg)
    for prompt, theirs in zip(prompts, references, strict=True):
        ours = _greedy(model, prompt, 16)
        assert len(ours) == 16
        assert ours == theirs, (
            f"greedy divergence on {tokenizer.decode(prompt)!r}: ours={ours} hf={theirs}"
        )
