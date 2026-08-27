"""FastAPI server exposing the async engine through the OpenAI compatible API."""

from __future__ import annotations

import argparse
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from clockwork.config import EngineConfig
from clockwork.engine.async_engine import AsyncLLMEngine
from clockwork.engine.sequence import RequestOutput, SamplingParams
from clockwork.server import protocol
from clockwork.server.metrics import ServerMetrics

_DEFAULT_MAX_TOKENS = 16


def _error(status: int, message: str) -> JSONResponse:
    body = protocol.ErrorResponse(error=protocol.ErrorDetail(message=message, code=status))
    return JSONResponse(status_code=status, content=body.model_dump())


def _sampling_params(
    req: protocol.ChatCompletionRequest | protocol.CompletionRequest,
) -> SamplingParams:
    stop = req.stop
    if stop is None:
        stop = []
    elif isinstance(stop, str):
        stop = [stop]
    return SamplingParams(
        max_tokens=req.max_tokens if req.max_tokens is not None else _DEFAULT_MAX_TOKENS,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=list(stop),
        seed=req.seed,
        ignore_eos=req.ignore_eos,
    )


def _usage(out: RequestOutput) -> protocol.UsageInfo:
    return protocol.UsageInfo(
        prompt_tokens=out.num_prompt_tokens,
        completion_tokens=out.num_generated_tokens,
        total_tokens=out.num_prompt_tokens + out.num_generated_tokens,
        prompt_tokens_details=protocol.PromptTokensDetails(cached_tokens=out.num_cached_tokens),
    )


def _chat_prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    # transformers may return a BatchEncoding or a nested list depending on version.
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(t) for t in ids]


def build_app(cfg: EngineConfig, engine: AsyncLLMEngine | None = None) -> FastAPI:
    """Build the FastAPI app; an injected engine skips model loading, for tests."""
    injected = engine

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        running = injected if injected is not None else AsyncLLMEngine.from_config(cfg)
        running.start()
        app.state.engine = running
        yield
        await running.shutdown()

    app = FastAPI(title="clockwork", lifespan=lifespan)
    served_model = cfg.model.model
    metrics = ServerMetrics()

    @app.middleware("http")
    async def count_requests(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/v1/"):
            metrics.observe_request(request.url.path, response.status_code)
        return response

    async def instrumented(request_id, prompt_ids, params, t0):
        # Wraps engine output to record TTFT at the first generated token and
        # duration plus token counters when the request finishes.
        running: AsyncLLMEngine = app.state.engine
        seen_first = False
        async for out in running.generate(request_id, prompt_ids, params):
            if not seen_first and out.num_generated_tokens > 0:
                seen_first = True
                metrics.ttft.observe(time.perf_counter() - t0)
            yield out
            if out.finished:
                metrics.observe_output(out, time.perf_counter() - t0)

    def bad_request(
        req: protocol.ChatCompletionRequest | protocol.CompletionRequest,
    ) -> JSONResponse | None:
        if req.model != served_model:
            return _error(400, f"model {req.model!r} does not exist; serving {served_model!r}")
        if req.n != 1:
            return _error(400, "only n=1 is supported")
        return None

    def bad_prompt(prompt_ids: list[int]) -> JSONResponse | None:
        if not prompt_ids:
            return _error(400, "prompt must contain at least one token")
        if len(prompt_ids) >= cfg.model.max_model_len:
            return _error(
                400,
                f"prompt of {len(prompt_ids)} tokens leaves no room to generate "
                f"within max_model_len {cfg.model.max_model_len}",
            )
        # Prefill is per sequence without chunking, so a prompt above the batched
        # token budget could never be admitted and would otherwise hang or return
        # an empty completion.
        if len(prompt_ids) > cfg.scheduler.max_num_batched_tokens:
            return _error(
                400,
                f"prompt of {len(prompt_ids)} tokens exceeds the prefill admission "
                f"budget max_num_batched_tokens {cfg.scheduler.max_num_batched_tokens}",
            )
        return None

    async def aggregate(
        raw: Request, request_id: str, prompt_ids: list[int], params: SamplingParams, t0: float
    ) -> RequestOutput | None:
        final = None
        async for out in instrumented(request_id, prompt_ids, params, t0):
            final = out
            # Leaving the stream before it finishes aborts the request in the engine.
            if not out.finished and await raw.is_disconnected():
                return None
        return final if final is not None and final.finished else None

    @app.post("/v1/chat/completions")
    async def chat_completions(req: protocol.ChatCompletionRequest, raw: Request):
        t0 = time.perf_counter()
        err = bad_request(req)
        if err is not None:
            return err
        running: AsyncLLMEngine = app.state.engine
        prompt_ids = _chat_prompt_ids(
            running.engine.tokenizer, [m.model_dump() for m in req.messages]
        )
        err = bad_prompt(prompt_ids)
        if err is not None:
            return err
        params = _sampling_params(req)
        request_id = protocol.random_id("chatcmpl")
        if req.stream:
            return StreamingResponse(
                chat_stream(request_id, prompt_ids, params, req.model, t0),
                media_type="text/event-stream",
            )
        try:
            final = await aggregate(raw, request_id, prompt_ids, params, t0)
        except ValueError as exc:
            return _error(400, str(exc))
        if final is None:
            return _error(400, "client disconnected before completion")
        return protocol.ChatCompletionResponse(
            id=request_id,
            model=req.model,
            choices=[
                protocol.ChatCompletionChoice(
                    message=protocol.ChatMessage(role="assistant", content=final.text),
                    finish_reason=final.finish_reason,
                )
            ],
            usage=_usage(final),
        )

    async def chat_stream(
        request_id: str, prompt_ids: list[int], params: SamplingParams, model: str, t0: float
    ) -> AsyncIterator[str]:
        created = protocol.now()

        def send(delta: protocol.DeltaMessage, finish_reason=None, usage=None) -> str:
            chunk = protocol.ChatCompletionChunk(
                id=request_id,
                created=created,
                model=model,
                choices=[
                    protocol.ChatCompletionChunkChoice(delta=delta, finish_reason=finish_reason)
                ],
                usage=usage,
            )
            return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

        yield send(protocol.DeltaMessage(role="assistant", content=""))
        final = None
        async for out in instrumented(request_id, prompt_ids, params, t0):
            final = out
            if out.delta_text:
                yield send(protocol.DeltaMessage(content=out.delta_text))
        if final is not None and final.finished:
            yield send(
                protocol.DeltaMessage(), finish_reason=final.finish_reason, usage=_usage(final)
            )
        yield "data: [DONE]\n\n"

    @app.post("/v1/completions")
    async def completions(req: protocol.CompletionRequest, raw: Request):
        t0 = time.perf_counter()
        err = bad_request(req)
        if err is not None:
            return err
        running: AsyncLLMEngine = app.state.engine
        if isinstance(req.prompt, str):
            prompt_ids = [int(t) for t in running.engine.tokenizer.encode(req.prompt)]
        else:
            prompt_ids = [int(t) for t in req.prompt]
        err = bad_prompt(prompt_ids)
        if err is not None:
            return err
        params = _sampling_params(req)
        request_id = protocol.random_id("cmpl")
        if req.stream:
            return StreamingResponse(
                completion_stream(request_id, prompt_ids, params, req.model, t0),
                media_type="text/event-stream",
            )
        try:
            final = await aggregate(raw, request_id, prompt_ids, params, t0)
        except ValueError as exc:
            return _error(400, str(exc))
        if final is None:
            return _error(400, "client disconnected before completion")
        return protocol.CompletionResponse(
            id=request_id,
            model=req.model,
            choices=[protocol.CompletionChoice(text=final.text, finish_reason=final.finish_reason)],
            usage=_usage(final),
        )

    async def completion_stream(
        request_id: str, prompt_ids: list[int], params: SamplingParams, model: str, t0: float
    ) -> AsyncIterator[str]:
        created = protocol.now()

        def send(text: str, finish_reason=None, usage=None) -> str:
            chunk = protocol.CompletionChunk(
                id=request_id,
                created=created,
                model=model,
                choices=[protocol.CompletionChoice(text=text, finish_reason=finish_reason)],
                usage=usage,
            )
            return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

        final = None
        async for out in instrumented(request_id, prompt_ids, params, t0):
            final = out
            if out.delta_text:
                yield send(out.delta_text)
        if final is not None and final.finished:
            yield send("", finish_reason=final.finish_reason, usage=_usage(final))
        yield "data: [DONE]\n\n"

    @app.get("/v1/models")
    async def models():
        return protocol.ModelList(data=[protocol.ModelCard(id=served_model)])

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/stats")
    async def stats():
        running: AsyncLLMEngine = app.state.engine
        return {
            "model": served_model,
            "attention_backend": running.engine.runner.attention_backend,
            **running.engine.stats(),
        }

    @app.get("/metrics")
    async def prometheus_metrics():
        running: AsyncLLMEngine = app.state.engine
        gauges = {
            key: value
            for key, value in running.engine.stats().items()
            if isinstance(value, (int, float))
        }
        info = {
            "model": served_model,
            "attention_backend": running.engine.runner.attention_backend,
        }
        return PlainTextResponse(
            metrics.render(gauges, info), media_type="text/plain; version=0.0.4"
        )

    return app


def main() -> None:
    """Serve an engine built from a yaml config over HTTP with uvicorn."""
    parser = argparse.ArgumentParser(prog="clockwork-serve")
    parser.add_argument("--config", required=True, help="path to an engine yaml config")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(build_app(EngineConfig.from_yaml(args.config)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
