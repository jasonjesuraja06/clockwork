"""Per-request sequence state shared by the scheduler, block manager, and engine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED_STOPPED = "finished_stopped"
    FINISHED_LENGTH = "finished_length"
    FINISHED_ABORTED = "finished_aborted"


_FINISHED_STATUSES = frozenset(
    {
        SequenceStatus.FINISHED_STOPPED,
        SequenceStatus.FINISHED_LENGTH,
        SequenceStatus.FINISHED_ABORTED,
    }
)


@dataclass
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


class Sequence:
    """Token state for one request: prompt, generated tokens, and KV cache progress."""

    def __init__(
        self,
        seq_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        arrival_time: float | None = None,
    ) -> None:
        self.seq_id = seq_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids: list[int] = []
        self.sampling_params = sampling_params if sampling_params is not None else SamplingParams()
        self.status = SequenceStatus.WAITING
        self.block_table: list[int] = []
        self.num_computed_tokens = 0
        self.num_cached_tokens = 0
        self.arrival_time = time.monotonic() if arrival_time is None else arrival_time

    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    def __len__(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def is_finished(self) -> bool:
        return self.status in _FINISHED_STATUSES

    def num_uncomputed_tokens(self) -> int:
        return len(self) - self.num_computed_tokens

    def reset_for_recompute(self) -> None:
        # Preemption by recompute: KV progress is dropped, tokens are kept, and the
        # caller frees and clears block_table through the block manager.
        self.num_computed_tokens = 0


@dataclass
class RequestOutput:
    request_id: str
    prompt_token_ids: list[int]
    token_ids: list[int]
    text: str
    finished: bool
    finish_reason: str | None
    num_cached_tokens: int
    num_prompt_tokens: int
    num_generated_tokens: int
    delta_text: str = ""
