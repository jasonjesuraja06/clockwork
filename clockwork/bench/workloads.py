"""Seeded workload generation: sharegpt single turn, agent traces, radix ablation."""

# The sharegpt kind reads data/sharegpt.json when the file exists and samples
# real conversations with the workload seed. When the file is absent (the
# default checkout; notebooks/bench_t4.ipynb downloads it for GPU runs) the
# generator synthesizes sharegpt-like prompt and output length distributions
# from the seeded lognormal sampler below and fills the text from VOCABULARY.

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from random import Random

from clockwork.bench.configs import WorkloadConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]

VOCABULARY = (
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "for",
    "on",
    "with",
    "by",
    "from",
    "at",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "can",
    "could",
    "may",
    "might",
    "must",
    "shall",
    "not",
    "no",
    "yes",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "when",
    "while",
    "before",
    "after",
    "over",
    "under",
    "between",
    "among",
    "through",
    "during",
    "without",
    "within",
    "into",
    "onto",
    "out",
    "up",
    "down",
    "left",
    "right",
    "first",
    "second",
    "third",
    "last",
    "next",
    "new",
    "old",
    "small",
    "large",
    "long",
    "short",
    "high",
    "low",
    "system",
    "user",
    "agent",
    "tool",
    "call",
    "result",
    "value",
    "key",
    "name",
    "type",
    "list",
    "table",
    "row",
    "column",
    "index",
    "range",
    "count",
    "total",
    "sum",
    "mean",
    "rate",
    "limit",
    "block",
    "cache",
    "token",
    "prompt",
    "prefix",
    "suffix",
    "batch",
    "queue",
    "stream",
    "request",
    "response",
    "server",
    "client",
    "engine",
    "model",
    "layer",
    "head",
    "weight",
    "bias",
    "input",
    "output",
    "state",
    "memory",
    "buffer",
    "page",
    "slot",
    "map",
    "tree",
    "node",
    "leaf",
    "root",
    "branch",
    "path",
    "file",
    "line",
    "word",
    "text",
    "data",
    "test",
    "check",
    "run",
    "start",
    "stop",
    "wait",
    "read",
    "write",
    "load",
    "store",
    "send",
    "receive",
    "open",
    "close",
    "find",
    "search",
    "sort",
    "merge",
    "split",
    "copy",
    "move",
    "add",
    "remove",
    "insert",
    "delete",
    "update",
    "return",
    "raise",
    "catch",
    "retry",
    "skip",
    "parse",
    "encode",
    "decode",
    "match",
    "miss",
    "hit",
    "free",
    "used",
    "ready",
    "done",
    "error",
    "time",
    "step",
    "turn",
    "round",
    "order",
    "plan",
    "task",
    "goal",
    "action",
    "answer",
    "question",
    "detail",
    "summary",
    "report",
    "note",
    "field",
    "record",
    "entry",
    "item",
    "unit",
    "case",
    "point",
    "part",
    "piece",
    "section",
    "region",
    "area",
    "zone",
    "level",
    "stage",
    "phase",
    "mode",
    "form",
    "kind",
)


@dataclass(frozen=True)
class BenchRequest:
    request_id: str
    prompt: str | None
    messages: list[dict[str, str]] | None
    max_tokens: int
    arrival_time: float
    prompt_tokens: int
    session_id: str = ""
    turn: int = 0


@dataclass
class HashWordTokenizer:
    """Deterministic whitespace tokenizer used when no model tokenizer is supplied."""

    vocab_size: int = 512
    eos_token_id: int | None = field(default=None)

    def encode(self, text: str) -> list[int]:
        # Reserve id 0 so generated ids stay valid for the tiny test model.
        return [1 + zlib.crc32(word.encode()) % (self.vocab_size - 1) for word in text.split()]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"t{i}" for i in ids)


def _trace_name(cfg: WorkloadConfig) -> str:
    # Ablation pairs share a trace_name so the on and off runs replay a
    # byte-identical request list, request ids included.
    return cfg.trace_name or cfg.name


def _sample_text(rng: Random, tokenizer, num_tokens: int) -> tuple[str, int]:
    words: list[str] = []
    count = 0
    while count < num_tokens:
        # Grow by roughly the remaining deficit; real tokenizers do not map one
        # word to one token, so re-count against the full text each round.
        need = max(8, num_tokens - count)
        words.extend(rng.choice(VOCABULARY) for _ in range(need))
        count = len(tokenizer.encode(" ".join(words)))
    return " ".join(words), count


def _arrival_gap(rng: Random, cfg: WorkloadConfig, rate: float) -> float:
    if cfg.arrival_process == "pareto":
        # paretovariate(alpha) has mean alpha / (alpha - 1); rescale to a mean gap of 1 / rate.
        scale = (cfg.pareto_alpha - 1.0) / (cfg.pareto_alpha * rate)
        return rng.paretovariate(cfg.pareto_alpha) * scale
    return rng.expovariate(rate)


def _clamped_lognormal(rng: Random, mean: float, sigma: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(rng.lognormvariate(mean, sigma))))


def _sharegpt_file(cfg: WorkloadConfig) -> Path:
    if cfg.sharegpt_path:
        return Path(cfg.sharegpt_path)
    return _REPO_ROOT / "data" / "sharegpt.json"


def _sharegpt_pairs(path: Path, tokenizer, cfg: WorkloadConfig) -> list[tuple[str, int, int]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    pairs: list[tuple[str, int, int]] = []
    for entry in entries:
        conversations = entry.get("conversations") or []
        prompt = reply = None
        for turn, follow in zip(conversations, conversations[1:], strict=False):
            if turn.get("from") in ("human", "user") and follow.get("from") in ("gpt", "assistant"):
                prompt = turn.get("value") or ""
                reply = follow.get("value") or ""
                break
        if not prompt or reply is None:
            continue
        prompt_tokens = len(tokenizer.encode(prompt))
        if not (cfg.prompt_len_min <= prompt_tokens <= cfg.prompt_len_max):
            continue
        out_len = max(cfg.output_len_min, min(cfg.max_tokens, len(tokenizer.encode(reply))))
        pairs.append((prompt, prompt_tokens, out_len))
    return pairs


def _gen_sharegpt(cfg: WorkloadConfig, tokenizer, rng: Random) -> list[BenchRequest]:
    trace = _trace_name(cfg)
    dataset = _sharegpt_file(cfg)
    pairs = _sharegpt_pairs(dataset, tokenizer, cfg) if dataset.is_file() else []
    requests: list[BenchRequest] = []
    now = 0.0
    for i in range(cfg.num_requests):
        now += _arrival_gap(rng, cfg, cfg.request_rate)
        if pairs:
            text, prompt_tokens, out_len = rng.choice(pairs)
        else:
            target = _clamped_lognormal(
                rng,
                cfg.prompt_len_log_mean,
                cfg.prompt_len_log_sigma,
                cfg.prompt_len_min,
                cfg.prompt_len_max,
            )
            text, prompt_tokens = _sample_text(rng, tokenizer, target)
            out_len = _clamped_lognormal(
                rng,
                cfg.output_len_log_mean,
                cfg.output_len_log_sigma,
                cfg.output_len_min,
                cfg.max_tokens,
            )
        requests.append(
            BenchRequest(
                request_id=f"{trace}-{i}",
                prompt=None,
                messages=[{"role": "user", "content": text}],
                max_tokens=out_len,
                arrival_time=now,
                prompt_tokens=prompt_tokens,
                session_id=f"{trace}-{i}",
                turn=0,
            )
        )
    return requests


def _gen_agent(cfg: WorkloadConfig, tokenizer, rng: Random) -> list[BenchRequest]:
    # One shared system-plus-tool-schema prefix for the whole trace; every
    # turn's prompt is that prefix plus the accumulated conversation, sent as a
    # raw completion so prefix sharing is exact at the token level.
    trace = _trace_name(cfg)
    half = max(8, cfg.shared_prefix_tokens // 2)
    system_text, _ = _sample_text(rng, tokenizer, half)
    tools_text, _ = _sample_text(rng, tokenizer, max(8, cfg.shared_prefix_tokens - half))
    prefix_text = f"system: {system_text}\ntools: {tools_text}"
    mean_turns = (cfg.turns_min + cfg.turns_max) / 2.0
    session_rate = cfg.request_rate / mean_turns
    requests: list[BenchRequest] = []
    session_start = 0.0
    session = 0
    while len(requests) < cfg.num_requests:
        session_start += _arrival_gap(rng, cfg, session_rate)
        num_turns = rng.randint(cfg.turns_min, cfg.turns_max)
        convo = prefix_text
        turn_time = session_start
        for turn in range(num_turns):
            if len(requests) >= cfg.num_requests:
                break
            suffix_target = max(8, int(rng.expovariate(1.0 / cfg.suffix_tokens_mean)))
            suffix_text, _ = _sample_text(rng, tokenizer, suffix_target)
            prompt = f"{convo}\nuser: {suffix_text}\nassistant:"
            prompt_tokens = len(tokenizer.encode(prompt))
            if prompt_tokens > cfg.prompt_len_max:
                break
            if turn > 0:
                # Open-loop approximation of agent think time between turns; the
                # previous turn's generation latency is not simulated.
                turn_time += rng.expovariate(1.0 / cfg.think_time_mean_s)
            requests.append(
                BenchRequest(
                    request_id=f"{trace}-s{session}-t{turn}",
                    prompt=prompt,
                    messages=None,
                    max_tokens=cfg.max_tokens,
                    arrival_time=turn_time,
                    prompt_tokens=prompt_tokens,
                    session_id=f"s{session}",
                    turn=turn,
                )
            )
            reply_target = max(4, int(rng.expovariate(1.0 / cfg.reply_tokens_mean)))
            reply_text, _ = _sample_text(rng, tokenizer, reply_target)
            convo = f"{prompt} {reply_text}"
        session += 1
    requests.sort(key=lambda r: (r.arrival_time, r.request_id))
    return requests


def generate(cfg: WorkloadConfig, tokenizer) -> list[BenchRequest]:
    """Deterministically generate the request trace for one workload configuration."""
    rng = Random(cfg.seed)
    if cfg.kind == "sharegpt":
        return _gen_sharegpt(cfg, tokenizer, rng)
    if cfg.kind in ("agent", "ablation"):
        return _gen_agent(cfg, tokenizer, rng)
    raise ValueError(f"unknown workload kind {cfg.kind!r}")
