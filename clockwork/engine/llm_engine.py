"""Synchronous engine loop: scheduling, execution, detokenization, and finish detection."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from clockwork.config import EngineConfig
from clockwork.engine.model_runner import ModelRunner
from clockwork.engine.sequence import RequestOutput, SamplingParams, Sequence, SequenceStatus
from clockwork.kvcache.block import BlockAllocator
from clockwork.kvcache.block_manager import BlockManager
from clockwork.radix.cache import RadixPrefixCache
from clockwork.scheduler.scheduler import Scheduler


@dataclass
class _DetokState:
    prefix_offset: int = 0
    read_offset: int = 0
    text: str = ""


def _resolve_eos_ids(tokenizer, hf_config) -> frozenset[int]:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        eos = getattr(hf_config, "eos_token_id", None)
    if eos is None:
        return frozenset()
    if isinstance(eos, int):
        return frozenset({eos})
    return frozenset(eos)


class LLMEngine:
    """Continuous-batching engine over the scheduler, block manager, and model runner."""

    def __init__(self, cfg: EngineConfig, model=None, hf_config=None, tokenizer=None) -> None:
        self.cfg = cfg
        self.runner = ModelRunner(cfg, model=model, hf_config=hf_config, tokenizer=tokenizer)
        self.tokenizer = self.runner.tokenizer
        if self.tokenizer is None:
            raise ValueError("a tokenizer is required to detokenize outputs")
        self.allocator = BlockAllocator(cfg.cache.num_blocks, cfg.cache.block_size)
        self.block_manager = BlockManager(
            self.allocator, kv_cache=self.runner.kv_cache, watermark=cfg.cache.watermark
        )
        self.prefix_cache = RadixPrefixCache(
            self.allocator, cfg.cache.block_size, enabled=cfg.cache.enable_prefix_cache
        )
        self.scheduler = Scheduler(
            cfg.scheduler, self.block_manager, prefix_cache=self.prefix_cache
        )
        self._eos_ids = _resolve_eos_ids(self.tokenizer, self.runner.hf_config)
        self._seqs: dict[str, Sequence] = {}
        self._detok: dict[str, _DetokState] = {}
        # Arrival times are a counter, not a clock, so scheduling order never
        # depends on wall time.
        self._arrival_counter = itertools.count()
        self._generate_counter = itertools.count()

    @classmethod
    def from_config(cls, cfg: EngineConfig) -> LLMEngine:
        """Build an engine from a config, loading the model and tokenizer."""
        return cls(cfg)

    def add_request(
        self,
        request_id: str,
        prompt: str | None = None,
        sampling_params: SamplingParams | None = None,
        prompt_token_ids: list[int] | None = None,
    ) -> None:
        """Queue one request; exactly one of prompt or prompt_token_ids must be given."""
        if (prompt is None) == (prompt_token_ids is None):
            raise ValueError("provide exactly one of prompt or prompt_token_ids")
        if request_id in self._seqs:
            raise ValueError(f"request id {request_id!r} is already active")
        if prompt_token_ids is None:
            prompt_token_ids = self.tokenizer.encode(prompt)
        prompt_token_ids = list(prompt_token_ids)
        if not prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        if len(prompt_token_ids) >= self.cfg.model.max_model_len:
            raise ValueError(
                f"prompt of {len(prompt_token_ids)} tokens leaves no room to generate "
                f"within max_model_len {self.cfg.model.max_model_len}"
            )
        seq = Sequence(
            request_id,
            prompt_token_ids,
            sampling_params if sampling_params is not None else SamplingParams(),
            arrival_time=float(next(self._arrival_counter)),
        )
        self._seqs[request_id] = seq
        self._detok[request_id] = _DetokState()
        self.scheduler.add_seq(seq)

    def step(self) -> list[RequestOutput]:
        """Run one schedule-execute iteration and return an output per scheduled sequence."""
        out = self.scheduler.schedule()
        if not out.prefill_seqs and not out.decode_seqs:
            if out.preempted or not self.scheduler.has_unfinished():
                return []
            # Admission stalled with nothing running: blocks held only by the
            # radix tree are invisible to the scheduler, so reclaim the idle
            # ones and retry once before declaring the request unschedulable.
            if self.prefix_cache.evict(self.allocator.num_blocks) > 0:
                out = self.scheduler.schedule()
            if not out.prefill_seqs and not out.decode_seqs:
                raise RuntimeError(
                    "waiting request cannot be admitted even with an empty radix cache"
                )
        self.runner.execute(out.prefill_seqs, out.decode_seqs)
        for seq in out.prefill_seqs:
            # Ownership rule of RadixPrefixCache.insert: the tree increfs only
            # blocks it newly adopts, so inserting while the sequence still
            # holds its blocks is safe on both hit and miss paths.
            self.prefix_cache.insert(seq.prompt_token_ids, seq.block_table)
        outputs = [self._process(seq) for seq in [*out.prefill_seqs, *out.decode_seqs]]
        self.scheduler.free_finished()
        for output in outputs:
            if output.finished:
                self._forget(output.request_id)
        return outputs

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished()

    def abort_request(self, request_id: str) -> None:
        """Abort a queued or running request and release its resources."""
        self.scheduler.abort(request_id)
        self._forget(request_id)

    def generate(
        self, prompts: list[str] | list[list[int]], sampling_params: SamplingParams | None = None
    ) -> list[RequestOutput]:
        """Run prompts to completion synchronously; returns final outputs in prompt order."""
        request_ids: list[str] = []
        for prompt in prompts:
            request_id = f"generate-{next(self._generate_counter)}"
            request_ids.append(request_id)
            if isinstance(prompt, str):
                self.add_request(request_id, prompt=prompt, sampling_params=sampling_params)
            else:
                self.add_request(
                    request_id, prompt_token_ids=prompt, sampling_params=sampling_params
                )
        finals: dict[str, RequestOutput] = {}
        pending = set(request_ids)
        while pending:
            outputs = self.step()
            for output in outputs:
                if output.finished and output.request_id in pending:
                    finals[output.request_id] = output
                    pending.discard(output.request_id)
            if not outputs and not self.scheduler.has_unfinished():
                raise RuntimeError(f"engine drained with unfinished requests: {sorted(pending)}")
        return [finals[request_id] for request_id in request_ids]

    def stats(self) -> dict:
        """Snapshot scheduler, prefix cache, and block pool counters."""
        sched = self.scheduler.stats
        cache = self.prefix_cache.stats
        return {
            "num_waiting": sched.num_waiting,
            "num_running": sched.num_running,
            "num_preempted_total": sched.num_preempted_total,
            "num_scheduled_batches": sched.num_scheduled_batches,
            "total_prompt_tokens": sched.total_prompt_tokens,
            "total_cached_tokens": sched.total_cached_tokens,
            "prefix_cache_queries": cache.queries,
            "prefix_cache_hit_tokens": cache.hit_tokens,
            "prefix_cache_prompt_tokens": cache.prompt_tokens,
            "prefix_cache_hit_rate": cache.hit_rate,
            "num_free_blocks": self.allocator.num_free_blocks,
        }

    def _forget(self, request_id: str) -> None:
        self._seqs.pop(request_id, None)
        self._detok.pop(request_id, None)
        self.runner.drop(request_id)

    def _decode_tokens(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _detok_step(self, seq: Sequence, state: _DetokState) -> str:
        ids = seq.output_token_ids
        new_text = self._decode_tokens(ids[state.prefix_offset :])
        if new_text.endswith("\ufffd"):
            # Incomplete multi-token unicode: hold the text back until the
            # byte sequence completes, then emit it in one delta.
            return ""
        prefix_text = self._decode_tokens(ids[state.prefix_offset : state.read_offset])
        delta = new_text[len(prefix_text) :]
        state.prefix_offset = state.read_offset
        state.read_offset = len(ids)
        return delta

    def _process(self, seq: Sequence) -> RequestOutput:
        params = seq.sampling_params
        state = self._detok[seq.seq_id]
        token = seq.output_token_ids[-1]
        finish_reason = None
        delta = ""
        stop_on_token = (not params.ignore_eos and token in self._eos_ids) or (
            token in params.stop_token_ids
        )
        if stop_on_token:
            # The stopping token stays in token_ids but contributes no text.
            finish_reason = "stop"
        else:
            delta = self._detok_step(seq, state)
            candidate = state.text + delta
            if params.stop:
                hits = [pos for pos in (candidate.find(s) for s in params.stop) if pos >= 0]
                if hits:
                    candidate = candidate[: min(hits)]
                    delta = candidate[len(state.text) :] if len(candidate) > len(state.text) else ""
                    finish_reason = "stop"
            state.text = candidate
        if finish_reason is None and (
            len(seq.output_token_ids) >= params.max_tokens
            or len(seq) >= self.cfg.model.max_model_len
        ):
            finish_reason = "length"
        if finish_reason == "stop":
            seq.status = SequenceStatus.FINISHED_STOPPED
        elif finish_reason == "length":
            seq.status = SequenceStatus.FINISHED_LENGTH
        return RequestOutput(
            request_id=seq.seq_id,
            prompt_token_ids=list(seq.prompt_token_ids),
            token_ids=list(seq.output_token_ids),
            text=state.text,
            finished=seq.is_finished(),
            finish_reason=finish_reason,
            num_cached_tokens=seq.num_cached_tokens,
            num_prompt_tokens=len(seq.prompt_token_ids),
            num_generated_tokens=len(seq.output_token_ids),
            delta_text=delta,
        )
