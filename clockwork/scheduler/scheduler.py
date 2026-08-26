"""Continuous batching scheduler: decode first, FCFS admission, preemption by recompute."""

from __future__ import annotations

from dataclasses import dataclass, field

from clockwork.config import SchedulerConfig
from clockwork.engine.sequence import Sequence, SequenceStatus
from clockwork.kvcache.block_manager import BlockManager
from clockwork.radix.cache import RadixPrefixCache


@dataclass
class SchedulerOutput:
    prefill_seqs: list[Sequence] = field(default_factory=list)
    decode_seqs: list[Sequence] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    num_batched_tokens: int = 0

    def is_empty(self) -> bool:
        return not (self.prefill_seqs or self.decode_seqs or self.preempted)


@dataclass
class SchedulerStats:
    num_waiting: int = 0
    num_running: int = 0
    num_preempted_total: int = 0
    num_scheduled_batches: int = 0
    total_prompt_tokens: int = 0
    total_cached_tokens: int = 0


class Scheduler:
    """Continuous batching: admit by token and block budget, preempt by recompute."""

    def __init__(
        self,
        cfg: SchedulerConfig,
        block_manager: BlockManager,
        prefix_cache: RadixPrefixCache | None = None,
    ) -> None:
        self.cfg = cfg
        self.block_manager = block_manager
        self.prefix_cache = prefix_cache
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.stats = SchedulerStats()
        # Radix matched prefix blocks per sequence. These are owned through the
        # reference the prefix cache took at match time, not through the block
        # manager, and must be released exactly once when the sequence lets go.
        self._radix_blocks: dict[str, list[int]] = {}
        # Deterministic tiebreak for equal arrival times: add_seq order.
        self._arrival_order: dict[str, int] = {}
        self._arrival_counter = 0

    def _priority(self, seq: Sequence) -> tuple[float, int]:
        return (seq.arrival_time, self._arrival_order.get(seq.seq_id, 0))

    def add_seq(self, seq: Sequence) -> None:
        self._arrival_order[seq.seq_id] = self._arrival_counter
        self._arrival_counter += 1
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)
        self._refresh_counts()

    def _free_resources(self, seq: Sequence) -> None:
        # The radix prefix occupies the leading block_table entries and is never
        # rewritten in place (copy on write can only hit the append position,
        # which lies past the matched span). Strip it so the block manager frees
        # only what it allocated, then drop the cache reference and lock.
        radix_blocks = self._radix_blocks.pop(seq.seq_id, [])
        if radix_blocks:
            del seq.block_table[: len(radix_blocks)]
        self.block_manager.free(seq)
        if radix_blocks and self.prefix_cache is not None:
            self.prefix_cache.release(radix_blocks)

    def _preempt(self, seq: Sequence) -> None:
        self._free_resources(seq)
        seq.reset_for_recompute()
        seq.num_cached_tokens = 0
        seq.status = SequenceStatus.PREEMPTED
        # Preempted first, FCFS within: preempted sequences sit at the front of
        # the waiting queue ahead of every never-admitted sequence, ordered by
        # arrival among themselves.
        key = self._priority(seq)
        i = 0
        while (
            i < len(self.waiting)
            and self.waiting[i].status is SequenceStatus.PREEMPTED
            and self._priority(self.waiting[i]) < key
        ):
            i += 1
        self.waiting.insert(i, seq)
        self.stats.num_preempted_total += 1

    def _schedule_decodes(self, preempted: list[Sequence]) -> list[Sequence]:
        ordered = sorted(self.running, key=self._priority)
        decode_seqs: list[Sequence] = []
        while ordered:
            seq = ordered.pop(0)
            while not self.block_manager.can_append(seq):
                # Victim rule: preempt the lowest priority (latest arrival)
                # running sequence not yet granted a slot this step; recompute
                # mode frees its blocks so earlier arrivals keep decoding.
                victim = ordered.pop() if ordered else seq
                self._preempt(victim)
                preempted.append(victim)
                if victim is seq:
                    break
            if seq.status is SequenceStatus.PREEMPTED:
                continue
            # Reserving the slot here makes can_append plus the slot grant
            # atomic across the batch: a later sequence sees the true pool.
            self.block_manager.append_slots(seq)
            decode_seqs.append(seq)
        return decode_seqs

    def _try_admit(self, seq: Sequence, num_batched_tokens: int) -> bool:
        matched_blocks: list[int] = []
        if self.prefix_cache is not None:
            match = self.prefix_cache.match(seq.token_ids())
            seq.num_cached_tokens = match.num_tokens
            seq.num_computed_tokens = match.num_tokens
            seq.block_table = list(match.block_ids)
            matched_blocks = match.block_ids
        fits = (
            num_batched_tokens + seq.num_uncomputed_tokens() <= self.cfg.max_num_batched_tokens
            and self.block_manager.can_allocate(seq)
        )
        if not fits:
            # Undo the speculative match so the admission attempt leaves no
            # reference or lock behind; the sequence stays at the queue head.
            if matched_blocks and self.prefix_cache is not None:
                self.prefix_cache.release(matched_blocks)
            seq.block_table = []
            seq.num_cached_tokens = 0
            seq.num_computed_tokens = 0
            return False
        self.block_manager.allocate(seq)
        if matched_blocks:
            self._radix_blocks[seq.seq_id] = list(matched_blocks)
        seq.status = SequenceStatus.RUNNING
        self.stats.total_prompt_tokens += len(seq)
        self.stats.total_cached_tokens += seq.num_cached_tokens
        return True

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        # Decode first: every already-running sequence gets its slot before any
        # new prefill can claim tokens or blocks.
        out.decode_seqs = self._schedule_decodes(out.preempted)
        self.running = list(out.decode_seqs)
        out.num_batched_tokens = len(out.decode_seqs)
        # Admission is strict FCFS over the waiting queue with head blocking: a
        # head that does not fit admits nothing behind it. Preempted sequences
        # sit at the front, so none can be bypassed by a newer request.
        while self.waiting and len(self.running) < self.cfg.max_num_seqs:
            seq = self.waiting[0]
            if not self._try_admit(seq, out.num_batched_tokens):
                break
            self.waiting.pop(0)
            self.running.append(seq)
            out.prefill_seqs.append(seq)
            out.num_batched_tokens += seq.num_uncomputed_tokens()
        if not out.is_empty():
            self.stats.num_scheduled_batches += 1
        self._refresh_counts()
        return out

    def free_finished(self) -> None:
        finished = [seq for seq in self.running if seq.is_finished()]
        for seq in finished:
            self._free_resources(seq)
            self._arrival_order.pop(seq.seq_id, None)
        self.running = [seq for seq in self.running if not seq.is_finished()]
        self._refresh_counts()

    def abort(self, seq_id: str) -> None:
        for i, seq in enumerate(self.waiting):
            if seq.seq_id == seq_id:
                # Waiting and preempted sequences hold no blocks and no radix
                # locks; preemption already released both.
                self.waiting.pop(i)
                seq.status = SequenceStatus.FINISHED_ABORTED
                self._arrival_order.pop(seq_id, None)
                self._refresh_counts()
                return
        for i, seq in enumerate(self.running):
            if seq.seq_id == seq_id:
                self._free_resources(seq)
                self.running.pop(i)
                seq.status = SequenceStatus.FINISHED_ABORTED
                self._arrival_order.pop(seq_id, None)
                self._refresh_counts()
                return

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def _refresh_counts(self) -> None:
        self.stats.num_waiting = len(self.waiting)
        self.stats.num_running = len(self.running)
