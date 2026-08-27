"""Nearest-rank percentile math and per-workload summaries over runner CSVs."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile: the ceil(pct / 100 * n)-th smallest value."""
    # Nearest-rank method, no interpolation between order statistics: the
    # result is always an observed value, unlike numpy's default linear
    # interpolation.
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _parse_itl(cell: str) -> list[float]:
    return [float(part) for part in cell.split(";") if part]


def summarize(csv_path: str | Path) -> dict:
    """Summarize one per-request CSV into latency percentiles, throughput, and hit rate."""
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ok = [row for row in rows if not row.get("error")]
    ttfts = [float(row["ttft_ms"]) for row in ok if row.get("ttft_ms")]
    # ITL is the list of gaps between successive output tokens after the first
    # token; the first token's latency is TTFT and is excluded.
    itls = [gap for row in ok for gap in _parse_itl(row.get("itl_ms", ""))]
    output_tokens = sum(int(row["output_tokens"]) for row in ok if row.get("output_tokens"))
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in ok if row.get("prompt_tokens"))
    cached_tokens = sum(int(row["cached_tokens"]) for row in ok if row.get("cached_tokens"))
    starts = [float(row["start_s"]) for row in ok if row.get("start_s")]
    ends = [float(row["end_s"]) for row in ok if row.get("end_s")]
    duration = max(ends) - min(starts) if starts and ends else 0.0
    first = rows[0] if rows else {}
    return {
        "workload": first.get("workload", ""),
        "kind": first.get("kind", ""),
        "request_rate": float(first.get("request_rate", 0.0) or 0.0),
        "radix_enabled": first.get("radix_enabled", ""),
        "seed": first.get("seed", ""),
        "num_requests": len(rows),
        "num_errors": len(rows) - len(ok),
        "ttft_p50_ms": round(percentile(ttfts, 50), 3),
        "ttft_p99_ms": round(percentile(ttfts, 99), 3),
        "itl_p50_ms": round(percentile(itls, 50), 3),
        "itl_p99_ms": round(percentile(itls, 99), 3),
        "output_tok_s": round(output_tokens / duration, 3) if duration > 0 else 0.0,
        "hit_rate": round(cached_tokens / prompt_tokens, 4) if prompt_tokens > 0 else 0.0,
    }
