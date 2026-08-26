import torch

from clockwork.engine.attn_metadata import AttentionMetadata
from clockwork.engine.sequence import (
    RequestOutput,
    SamplingParams,
    Sequence,
    SequenceStatus,
)


def test_sampling_params_greedy():
    assert SamplingParams().greedy is True
    assert SamplingParams(temperature=0.7).greedy is False


def test_sampling_params_default_lists_are_independent():
    a = SamplingParams()
    b = SamplingParams()
    a.stop.append("</s>")
    a.stop_token_ids.append(2)
    assert b.stop == []
    assert b.stop_token_ids == []


def test_new_sequence_state():
    seq = Sequence("s0", [1, 2, 3], arrival_time=0.0)
    assert seq.status is SequenceStatus.WAITING
    assert seq.token_ids() == [1, 2, 3]
    assert len(seq) == 3
    assert seq.block_table == []
    assert seq.num_computed_tokens == 0
    assert seq.num_cached_tokens == 0
    assert seq.num_uncomputed_tokens() == 3
    assert seq.arrival_time == 0.0
    assert seq.sampling_params.greedy is True


def test_sequence_copies_prompt_token_ids():
    prompt = [1, 2, 3]
    seq = Sequence("s0", prompt, arrival_time=0.0)
    prompt.append(4)
    assert seq.prompt_token_ids == [1, 2, 3]


def test_append_token_and_uncomputed_accounting():
    seq = Sequence("s0", [1, 2, 3], arrival_time=0.0)
    seq.num_computed_tokens = 3
    seq.append_token(7)
    assert seq.token_ids() == [1, 2, 3, 7]
    assert seq.output_token_ids == [7]
    assert len(seq) == 4
    assert seq.num_uncomputed_tokens() == 1


def test_is_finished_per_status():
    seq = Sequence("s0", [1], arrival_time=0.0)
    finished = {
        SequenceStatus.FINISHED_STOPPED,
        SequenceStatus.FINISHED_LENGTH,
        SequenceStatus.FINISHED_ABORTED,
    }
    for status in SequenceStatus:
        seq.status = status
        assert seq.is_finished() is (status in finished)


def test_reset_for_recompute_keeps_tokens_and_drops_progress():
    seq = Sequence("s0", [1, 2, 3], arrival_time=0.0)
    seq.num_computed_tokens = 3
    seq.append_token(7)
    seq.num_computed_tokens = 4
    seq.num_cached_tokens = 2
    seq.block_table = [0, 1]
    seq.reset_for_recompute()
    assert seq.num_computed_tokens == 0
    assert seq.prompt_token_ids == [1, 2, 3]
    assert seq.output_token_ids == [7]
    assert seq.num_uncomputed_tokens() == 4
    # block_table is freed and cleared by the block manager, not by the sequence.
    assert seq.block_table == [0, 1]


def test_request_output_delta_text_default():
    out = RequestOutput(
        request_id="r0",
        prompt_token_ids=[1, 2],
        token_ids=[3],
        text="a",
        finished=True,
        finish_reason="stop",
        num_cached_tokens=0,
        num_prompt_tokens=2,
        num_generated_tokens=1,
    )
    assert out.delta_text == ""
    assert out.finish_reason == "stop"


def test_attention_metadata_construction():
    meta = AttentionMetadata(
        is_prefill=True,
        slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int64),
        block_tables=None,
        ctx_lens=None,
        query_lens=[3],
        seq_block_tables=[[0]],
        logits_indices=torch.tensor([2], dtype=torch.int64),
    )
    assert meta.is_prefill is True
    assert meta.query_lens == [3]
    assert meta.logits_indices.tolist() == [2]
