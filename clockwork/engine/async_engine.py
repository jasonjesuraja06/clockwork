"""Async wrapper: background step loop with per-request streaming queues."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator

from clockwork.config import EngineConfig
from clockwork.engine.llm_engine import LLMEngine
from clockwork.engine.sequence import RequestOutput, SamplingParams


class AsyncLLMEngine:
    """Streams RequestOutputs per request while one background task drives LLMEngine.step."""

    def __init__(self, cfg: EngineConfig, model=None, hf_config=None, tokenizer=None) -> None:
        self.engine = LLMEngine(cfg, model=model, hf_config=hf_config, tokenizer=tokenizer)
        # Serializes engine access between the event loop thread and the
        # executor thread running step(); every engine call goes through it.
        self._lock = threading.Lock()
        self._queues: dict[str, asyncio.Queue[RequestOutput | None]] = {}
        self._task: asyncio.Task | None = None
        self._wakeup: asyncio.Event | None = None
        self._running = False

    @classmethod
    def from_config(cls, cfg: EngineConfig) -> AsyncLLMEngine:
        """Build an async engine from a config, loading the model and tokenizer."""
        return cls(cfg)

    def start(self) -> None:
        """Start the background stepping task on the running event loop; idempotent."""
        if self._task is not None and not self._task.done():
            return
        loop = asyncio.get_running_loop()
        if self._wakeup is None:
            self._wakeup = asyncio.Event()
        self._running = True
        self._task = loop.create_task(self._run_loop())

    async def shutdown(self) -> None:
        """Stop the background loop and release every waiting consumer."""
        self._running = False
        if self._wakeup is not None:
            self._wakeup.set()
        task = self._task
        self._task = None
        if task is not None:
            await task

    async def generate(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
    ) -> AsyncIterator[RequestOutput]:
        """Stream outputs for one request; a consumer that stops early aborts it."""
        self.start()
        if request_id in self._queues:
            raise ValueError(f"request {request_id!r} is already streaming")
        queue: asyncio.Queue[RequestOutput | None] = asyncio.Queue()
        self._queues[request_id] = queue
        loop = asyncio.get_running_loop()
        finished = False
        try:
            await loop.run_in_executor(
                None, self._locked_add, request_id, list(prompt_token_ids), sampling_params
            )
            self._wakeup.set()
            while True:
                item = await queue.get()
                if item is None:
                    break
                finished = item.finished
                yield item
                if finished:
                    break
        finally:
            self._queues.pop(request_id, None)
            if not finished:
                with self._lock:
                    self.engine.abort_request(request_id)

    async def abort(self, request_id: str) -> None:
        """Abort a streaming request and end its consumer's stream."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._locked_abort, request_id)
        queue = self._queues.get(request_id)
        if queue is not None:
            queue.put_nowait(None)

    async def _run_loop(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while self._running:
                # Clear before the check: an add that lands after the clear
                # sets the event, one before it is seen by the check itself.
                self._wakeup.clear()
                if not self._has_unfinished():
                    await self._wakeup.wait()
                    continue
                outputs = await loop.run_in_executor(None, self._locked_step)
                for output in outputs:
                    queue = self._queues.get(output.request_id)
                    if queue is None:
                        continue
                    queue.put_nowait(output)
                    if output.finished:
                        queue.put_nowait(None)
        finally:
            self._running = False
            for queue in self._queues.values():
                queue.put_nowait(None)

    def _locked_step(self) -> list[RequestOutput]:
        with self._lock:
            return self.engine.step()

    def _locked_add(
        self, request_id: str, prompt_token_ids: list[int], sampling_params: SamplingParams | None
    ) -> None:
        with self._lock:
            self.engine.add_request(
                request_id, prompt_token_ids=prompt_token_ids, sampling_params=sampling_params
            )

    def _locked_abort(self, request_id: str) -> None:
        with self._lock:
            self.engine.abort_request(request_id)

    def _has_unfinished(self) -> bool:
        with self._lock:
            return self.engine.has_unfinished_requests()
