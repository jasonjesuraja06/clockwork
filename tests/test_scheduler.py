from clockwork.config import SchedulerConfig
from clockwork.engine.sequence import SamplingParams, Sequence, SequenceStatus
from clockwork.kvcache import BlockAllocator
from clockwork.kvcache.block_manager import BlockManager
from clockwork.radix import RadixPrefixCache
from clockwork.scheduler import Scheduler

BLOCK_SIZE = 4


def make_env(
    num_blocks: int = 64,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 64,
    watermark: float = 0.0,
    with_prefix_cache: bool = False,
) -> tuple[Scheduler, BlockAllocator, RadixPrefixCache | None]:
    alloc = BlockAllocator(num_blocks=num_blocks, block_size=BLOCK_SIZE)
    mgr = BlockManager(alloc, watermark=watermark)
    cache = RadixPrefixCache(alloc, BLOCK_SIZE) if with_prefix_cache else None
    cfg = SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens)
    return Scheduler(cfg, mgr, prefix_cache=cache), alloc, cache


def make_seq(seq_id: str, prompt: list[int], arrival: float, max_tokens: int = 64) -> Sequence:
    return Sequence(seq_id, prompt, SamplingParams(max_tokens=max_tokens), arrival_time=arrival)


def engine_step(sched: Scheduler):
    # Minimal engine stand-in: mark scheduled tokens computed, append the next
    # sampled token, finish at max_tokens, then free finished sequences.
    out = sched.schedule()
    for seq in out.prefill_seqs:
        seq.num_computed_tokens = len(seq)
        _append_or_finish(seq)
    for seq in out.decode_seqs:
        seq.num_computed_tokens += 1
        _append_or_finish(seq)
    sched.free_finished()
    return out


def _append_or_finish(seq: Sequence) -> None:
    seq.append_token(7)
    if len(seq.output_token_ids) >= seq.sampling_params.max_tokens:
        seq.status = SequenceStatus.FINISHED_LENGTH


def test_token_budget_splits_a_batch():
    sched, _, _ = make_env(max_num_batched_tokens=10)
    a = make_seq("a", list(range(6)), arrival=0.0)
    b = make_seq("b", list(range(6)), arrival=1.0)
    sched.add_seq(a)
    sched.add_seq(b)

    out1 = sched.schedule()
    assert out1.prefill_seqs == [a]
    assert out1.decode_seqs == []
    assert out1.num_batched_tokens == 6
    assert b.status is SequenceStatus.WAITING

    a.num_computed_tokens = len(a)
    a.append_token(7)
    out2 = sched.schedule()
    assert out2.decode_seqs == [a]
    assert out2.prefill_seqs == [b]
    assert out2.num_batched_tokens == 7


def test_max_num_seqs_cap():
    sched, _, _ = make_env(max_num_seqs=2)
    a = make_seq("a", list(range(4)), arrival=0.0, max_tokens=1)
    b = make_seq("b", list(range(4)), arrival=1.0, max_tokens=8)
    c = make_seq("c", list(range(4)), arrival=2.0, max_tokens=8)
    for seq in (a, b, c):
        sched.add_seq(seq)

    out1 = engine_step(sched)
    assert out1.prefill_seqs == [a, b]
    assert c.status is SequenceStatus.WAITING
    assert a.is_finished()

    out2 = engine_step(sched)
    assert out2.decode_seqs == [b]
    assert out2.prefill_seqs == [c]


def pressure_env(max_num_seqs: int = 2):
    # 6 blocks total: two 2-block prompts fit, but decode growth exhausts the
    # pool and forces a preemption.
    sched, alloc, _ = make_env(num_blocks=6, max_num_seqs=max_num_seqs)
    a = make_seq("a", list(range(8)), arrival=0.0, max_tokens=6)
    b = make_seq("b", list(range(100, 108)), arrival=1.0, max_tokens=6)
    sched.add_seq(a)
    sched.add_seq(b)
    return sched, alloc, a, b


def run_until_preemption(sched: Scheduler, limit: int = 20):
    for _ in range(limit):
        out = engine_step(sched)
        if out.preempted:
            return out
    raise AssertionError("no preemption occurred")


def test_preemption_recompute_and_readmission():
    sched, alloc, a, b = pressure_env()
    out = run_until_preemption(sched)

    # Lowest priority (latest arrival) is the victim; recompute drops its KV.
    assert out.preempted == [b]
    assert b.status is SequenceStatus.PREEMPTED
    assert b.block_table == []
    assert b.num_computed_tokens == 0
    assert sched.waiting[0] is b
    assert a in out.decode_seqs
    assert sched.stats.num_preempted_total == 1

    # The preempted sequence is never lost: it is readmitted and finishes.
    for _ in range(20):
        if b.is_finished():
            break
        engine_step(sched)
    assert b.status is SequenceStatus.FINISHED_LENGTH
    assert len(b.output_token_ids) == b.sampling_params.max_tokens
    assert a.status is SequenceStatus.FINISHED_LENGTH
    assert not sched.has_unfinished()
    assert alloc.num_free_blocks == alloc.num_blocks


def test_preempted_first_readmission_order():
    sched, _, a, b = pressure_env(max_num_seqs=2)
    c = make_seq("c", list(range(200, 204)), arrival=2.0, max_tokens=2)
    sched.add_seq(c)

    run_until_preemption(sched)
    # b rejoined the queue at the front, ahead of c which was waiting longer.
    assert [seq.seq_id for seq in sched.waiting] == ["b", "c"]

    for _ in range(20):
        out = sched.schedule()
        if out.prefill_seqs:
            break
        engine_step_apply(sched, out)
    assert out.prefill_seqs[0] is b
    assert b.status is SequenceStatus.RUNNING


def engine_step_apply(sched: Scheduler, out) -> None:
    for seq in out.prefill_seqs:
        seq.num_computed_tokens = len(seq)
        _append_or_finish(seq)
    for seq in out.decode_seqs:
        seq.num_computed_tokens += 1
        _append_or_finish(seq)
    sched.free_finished()


def cached_env(max_num_batched_tokens: int):
    sched, alloc, cache = make_env(
        num_blocks=16,
        max_num_batched_tokens=max_num_batched_tokens,
        with_prefix_cache=True,
    )
    tokens = list(range(100, 112))
    blocks = alloc.allocate_many(2)
    cache.insert(tokens[:8], blocks)
    alloc.free_many(blocks)
    return sched, alloc, cache, tokens, blocks


def test_radix_hit_reduces_admission_cost():
    # Budget 6 rejects a cold 12 token prompt but admits the same prompt when
    # 8 tokens are served from the radix cache.
    sched, _, _, tokens, blocks = cached_env(max_num_batched_tokens=6)
    seq = make_seq("s", tokens, arrival=0.0)
    sched.add_seq(seq)
    out = sched.schedule()
    assert out.prefill_seqs == [seq]
    assert out.num_batched_tokens == 4
    assert seq.num_cached_tokens == 8
    assert seq.num_computed_tokens == 8
    assert seq.block_table[:2] == blocks
    assert len(seq.block_table) == 3
    assert sched.stats.total_cached_tokens == 8
    assert sched.stats.total_prompt_tokens == 12

    cold_sched, _, _, _, _ = cached_env(max_num_batched_tokens=6)
    cold = make_seq("cold", list(range(200, 212)), arrival=0.0)
    cold_sched.add_seq(cold)
    out = cold_sched.schedule()
    assert out.is_empty()
    assert cold.status is SequenceStatus.WAITING


def test_failed_admission_rolls_back_radix_match():
    # Even with 8 cached tokens the remaining 4 exceed a budget of 3; the
    # speculative match must be fully undone.
    sched, alloc, cache, tokens, blocks = cached_env(max_num_batched_tokens=3)
    seq = make_seq("s", tokens, arrival=0.0)
    sched.add_seq(seq)
    out = sched.schedule()
    assert out.is_empty()
    assert seq.block_table == []
    assert seq.num_cached_tokens == 0
    assert seq.num_computed_tokens == 0
    assert cache.tree.evictable_blocks() == 2
    assert all(alloc.refcount(block_id) == 1 for block_id in blocks)


def test_abort_waiting():
    sched, alloc, _ = make_env()
    a = make_seq("a", list(range(4)), arrival=0.0)
    sched.add_seq(a)
    sched.abort("a")
    assert a.status is SequenceStatus.FINISHED_ABORTED
    assert sched.waiting == []
    assert not sched.has_unfinished()
    assert alloc.num_free_blocks == alloc.num_blocks


def test_abort_running():
    sched, alloc, _ = make_env()
    a = make_seq("a", list(range(6)), arrival=0.0)
    sched.add_seq(a)
    sched.schedule()
    assert a.status is SequenceStatus.RUNNING
    sched.abort("a")
    assert a.status is SequenceStatus.FINISHED_ABORTED
    assert sched.running == []
    assert not sched.has_unfinished()
    assert alloc.num_free_blocks == alloc.num_blocks


def test_abort_running_releases_radix_locks():
    sched, alloc, cache, tokens, blocks = cached_env(max_num_batched_tokens=64)
    seq = make_seq("s", tokens, arrival=0.0)
    sched.add_seq(seq)
    sched.schedule()
    assert seq.num_cached_tokens == 8
    sched.abort("s")
    assert seq.status is SequenceStatus.FINISHED_ABORTED
    assert alloc.num_free_blocks == alloc.num_blocks - 2
    assert cache.tree.evictable_blocks() == 2
    assert all(alloc.refcount(block_id) == 1 for block_id in blocks)


def test_abort_preempted():
    sched, alloc, a, b = pressure_env()
    run_until_preemption(sched)
    assert b.status is SequenceStatus.PREEMPTED
    sched.abort("b")
    assert b.status is SequenceStatus.FINISHED_ABORTED
    assert b not in sched.waiting
    for _ in range(20):
        if not sched.has_unfinished():
            break
        engine_step(sched)
    assert a.is_finished()
    assert alloc.num_free_blocks == alloc.num_blocks


def test_abort_unknown_id_is_a_noop():
    sched, _, _ = make_env()
    sched.abort("missing")
    assert not sched.has_unfinished()


def run_scripted_session() -> list[tuple]:
    # 7 blocks and a 16 token budget: the session goes through admission
    # splits, a self preemption, and a victim preemption before draining.
    sched, _, _ = make_env(num_blocks=7, max_num_seqs=3, max_num_batched_tokens=16)
    seqs = [
        make_seq("a", list(range(8)), arrival=0.0, max_tokens=6),
        make_seq("b", list(range(100, 108)), arrival=1.0, max_tokens=6),
        make_seq("c", list(range(200, 204)), arrival=2.0, max_tokens=4),
        make_seq("d", list(range(300, 312)), arrival=3.0, max_tokens=3),
    ]
    for seq in seqs:
        sched.add_seq(seq)
    history: list[tuple] = []
    for _ in range(30):
        out = engine_step(sched)
        history.append(
            (
                tuple(seq.seq_id for seq in out.prefill_seqs),
                tuple(seq.seq_id for seq in out.decode_seqs),
                tuple(seq.seq_id for seq in out.preempted),
                out.num_batched_tokens,
            )
        )
        if not sched.has_unfinished():
            break
    assert not sched.has_unfinished()
    history.append(tuple(len(seq.output_token_ids) for seq in seqs))
    return history


def test_schedule_is_deterministic():
    assert run_scripted_session() == run_scripted_session()


def test_stats_counters():
    sched, _, _ = make_env(max_num_batched_tokens=10)
    a = make_seq("a", list(range(6)), arrival=0.0, max_tokens=2)
    b = make_seq("b", list(range(6)), arrival=1.0, max_tokens=2)
    sched.add_seq(a)
    sched.add_seq(b)
    assert sched.stats.num_waiting == 2
    engine_step(sched)
    assert sched.stats.num_running == 1
    assert sched.stats.num_waiting == 1
    assert sched.stats.total_prompt_tokens == 6
    assert sched.stats.num_scheduled_batches == 1
    while sched.has_unfinished():
        engine_step(sched)
    assert sched.stats.total_prompt_tokens == 12
    assert sched.stats.num_running == 0
    assert sched.stats.num_waiting == 0


def test_empty_schedule_output():
    sched, _, _ = make_env()
    out = sched.schedule()
    assert out.is_empty()
    assert out.num_batched_tokens == 0
    assert sched.stats.num_scheduled_batches == 0
