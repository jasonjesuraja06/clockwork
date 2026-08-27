"""Async benchmark driver: paced arrivals, SSE streaming, per-request CSV records."""

from __future__ import annotations

import asyncio
import contextlib
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from clockwork.bench.configs import WorkloadConfig
from clockwork.bench.metrics import summarize
from clockwork.bench.workloads import BenchRequest, HashWordTokenizer, generate

CSV_FIELDS = [
    "workload",
    "kind",
    "request_rate",
    "radix_enabled",
    "seed",
    "request_id",
    "session_id",
    "turn",
    "arrival_s",
    "start_s",
    "end_s",
    "ttft_ms",
    "itl_ms",
    "prompt_tokens",
    "output_tokens",
    "cached_tokens",
    "error",
]

SUMMARY_FIELDS = [
    "workload",
    "kind",
    "request_rate",
    "radix_enabled",
    "seed",
    "num_requests",
    "num_errors",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "itl_p50_ms",
    "itl_p99_ms",
    "output_tok_s",
    "hit_rate",
    "gpu_util_mean",
    "gpu_util_max",
]


@dataclass
class _Record:
    request: BenchRequest
    start_s: float = 0.0
    end_s: float = 0.0
    ttft_ms: float | None = None
    itl_ms: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    error: str = ""


class _GpuSampler:
    """Samples GPU utilization via pynvml when present; a no-op on machines without it."""

    def __init__(self) -> None:
        self._samples: list[float] = []
        self._task: asyncio.Task | None = None
        self._nvml = None
        self._handle = None

    def start(self) -> None:
        try:
            import pynvml
        except ImportError:
            return
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return
        self._nvml = pynvml
        self._task = asyncio.get_running_loop().create_task(self._loop())

    async def _loop(self) -> None:
        while True:
            self._samples.append(float(self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu))
            await asyncio.sleep(0.25)

    async def stop(self) -> tuple[str, str]:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        if not self._samples:
            return "", ""
        mean = sum(self._samples) / len(self._samples)
        return f"{mean:.1f}", f"{max(self._samples):.1f}"


def _chunk_text(obj: dict) -> str:
    choices = obj.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    text = choice.get("text")
    if text:
        return text
    delta = choice.get("delta") or {}
    return delta.get("content") or ""


def _cached_tokens(usage: dict) -> int:
    details = usage.get("prompt_tokens_details") or {}
    value = details.get("cached_tokens", usage.get("cached_tokens"))
    return int(value) if value is not None else 0


async def _fetch_model_id(client: httpx.AsyncClient) -> str:
    try:
        resp = await client.get("/v1/models")
        resp.raise_for_status()
        return resp.json()["data"][0]["id"]
    except Exception:
        return "clockwork"


async def _drive(
    client: httpx.AsyncClient,
    request: BenchRequest,
    model: str,
    t0: float,
    record: _Record,
    ignore_eos: bool = False,
) -> None:
    delay = request.arrival_time - (time.perf_counter() - t0)
    if delay > 0:
        await asyncio.sleep(delay)
    record.start_s = time.perf_counter() - t0
    record.prompt_tokens = request.prompt_tokens
    payload: dict = {
        "model": model,
        "max_tokens": request.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    if request.messages is not None:
        path = "/v1/chat/completions"
        payload["messages"] = request.messages
    else:
        path = "/v1/completions"
        payload["prompt"] = request.prompt
    chunk_times: list[float] = []
    usage: dict | None = None
    try:
        async with client.stream("POST", path, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"http {resp.status_code}: {body[:200]!r}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                if _chunk_text(obj):
                    chunk_times.append(time.perf_counter() - t0)
    except Exception as exc:
        record.error = f"{type(exc).__name__}: {exc}"
    record.end_s = time.perf_counter() - t0
    if chunk_times:
        record.ttft_ms = (chunk_times[0] - record.start_s) * 1000.0
        # ITL: gaps between successive output tokens after the first token.
        record.itl_ms = [
            (b - a) * 1000.0 for a, b in zip(chunk_times, chunk_times[1:], strict=False)
        ]
    record.output_tokens = len(chunk_times)
    if usage is not None:
        record.output_tokens = int(usage.get("completion_tokens", record.output_tokens))
        record.prompt_tokens = int(usage.get("prompt_tokens", record.prompt_tokens))
        record.cached_tokens = _cached_tokens(usage)


def _write_csv(path: Path, cfg: WorkloadConfig, records: list[_Record]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "workload": cfg.name,
                    "kind": cfg.kind,
                    "request_rate": cfg.request_rate,
                    "radix_enabled": cfg.radix_enabled,
                    "seed": cfg.seed,
                    "request_id": record.request.request_id,
                    "session_id": record.request.session_id,
                    "turn": record.request.turn,
                    "arrival_s": f"{record.request.arrival_time:.4f}",
                    "start_s": f"{record.start_s:.4f}",
                    "end_s": f"{record.end_s:.4f}",
                    "ttft_ms": "" if record.ttft_ms is None else f"{record.ttft_ms:.3f}",
                    "itl_ms": ";".join(f"{gap:.3f}" for gap in record.itl_ms),
                    "prompt_tokens": record.prompt_tokens,
                    "output_tokens": record.output_tokens,
                    "cached_tokens": record.cached_tokens,
                    "error": record.error,
                }
            )


def _append_summary(summary_path: Path, row: dict) -> None:
    exists = summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


async def run(
    cfg: WorkloadConfig,
    base_url: str,
    out_dir: str | Path,
    tokenizer=None,
    client: httpx.AsyncClient | None = None,
    ignore_eos: bool = False,
) -> Path:
    """Run one workload against an OpenAI-compatible server and write its CSVs."""
    if tokenizer is None:
        tokenizer = HashWordTokenizer()
    requests = generate(cfg, tokenizer)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(300.0, connect=10.0))
    sampler = _GpuSampler()
    try:
        model = await _fetch_model_id(client)
        records = [_Record(request=request) for request in requests]
        sampler.start()
        t0 = time.perf_counter()
        await asyncio.gather(
            *(
                _drive(client, record.request, model, t0, record, ignore_eos=ignore_eos)
                for record in records
            )
        )
        gpu_mean, gpu_max = await sampler.stop()
    finally:
        if own_client:
            await client.aclose()
    csv_path = out_dir / f"{cfg.name}.csv"
    _write_csv(csv_path, cfg, records)
    summary = summarize(csv_path)
    summary["gpu_util_mean"] = gpu_mean
    summary["gpu_util_max"] = gpu_max
    _append_summary(out_dir / "summary.csv", summary)
    return csv_path
