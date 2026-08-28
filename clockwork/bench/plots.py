"""Figures from bench summary CSVs; plots measured values only."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def _read_summary(summary_csv: str | Path) -> list[dict]:
    with Path(summary_csv).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["request_rate"] = float(row["request_rate"])
        for key in ("ttft_p50_ms", "ttft_p99_ms", "itl_p50_ms", "itl_p99_ms", "output_tok_s"):
            row[key] = float(row[key]) if row.get(key) not in (None, "", "nan") else float("nan")
        row["hit_rate"] = float(row["hit_rate"]) if row.get("hit_rate") else 0.0
    return rows


def _by_rate_mean(rows: list[dict], key: str) -> tuple[list[float], list[float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["request_rate"]].append(row[key])
    rates = sorted(grouped)
    return rates, [sum(grouped[rate]) / len(grouped[rate]) for rate in rates]


def _latency_figure(rows: list[dict], p50_key: str, p99_key: str, ylabel: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind in ("singleturn", "agent"):
        kind_rows = [row for row in rows if row["kind"] == kind]
        if not kind_rows:
            continue
        for key, style in ((p50_key, "-o"), (p99_key, "--s")):
            rates, means = _by_rate_mean(kind_rows, key)
            label = f"{kind} {key.split('_')[-2]}"
            ax.plot(rates, means, style, label=label)
    ax.set_xlabel("request rate (req/s)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _throughput_figure(rows: list[dict], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind in ("singleturn", "agent"):
        kind_rows = [row for row in rows if row["kind"] == kind]
        if not kind_rows:
            continue
        rates, means = _by_rate_mean(kind_rows, "output_tok_s")
        ax.plot(rates, means, "-o", label=kind)
    ax.set_xlabel("request rate (req/s)")
    ax.set_ylabel("output tokens per second")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _ablation_figure(rows: list[dict], path: Path) -> Path:
    rates = sorted({row["request_rate"] for row in rows})
    fig, (ax_ttft, ax_hit) = plt.subplots(1, 2, figsize=(9, 4.5))
    width = 0.35
    variants = ((-width / 2, "True", "radix on"), (width / 2, "False", "radix off"))
    for offset, enabled, label in variants:
        subset = {row["request_rate"]: row for row in rows if row["radix_enabled"] == enabled}
        xs = [i + offset for i, rate in enumerate(rates) if rate in subset]
        ttfts = [subset[rate]["ttft_p50_ms"] for rate in rates if rate in subset]
        hits = [subset[rate]["hit_rate"] for rate in rates if rate in subset]
        ax_ttft.bar(xs, ttfts, width=width, label=label)
        ax_hit.bar(xs, hits, width=width, label=label)
    for ax, ylabel in ((ax_ttft, "ttft p50 (ms)"), (ax_hit, "prefix cache hit rate")):
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{rate:g}" for rate in rates])
        ax.set_xlabel("request rate (req/s)")
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_all(summary_csv: str | Path, out_dir: str | Path = "docs/figures") -> list[Path]:
    """Render every figure that the summary CSV has data for; returns written paths."""
    rows = _read_summary(summary_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    rate_rows = [row for row in rows if row["kind"] in ("singleturn", "agent")]
    if rate_rows:
        written.append(
            _latency_figure(
                rate_rows, "ttft_p50_ms", "ttft_p99_ms", "ttft (ms)", out_dir / "ttft_vs_rate.png"
            )
        )
        written.append(
            _latency_figure(
                rate_rows,
                "itl_p50_ms",
                "itl_p99_ms",
                "inter-token latency (ms)",
                out_dir / "itl_vs_rate.png",
            )
        )
        written.append(_throughput_figure(rate_rows, out_dir / "throughput_vs_rate.png"))
    ablation_rows = [row for row in rows if row["kind"] == "ablation"]
    if ablation_rows:
        written.append(_ablation_figure(ablation_rows, out_dir / "radix_ablation.png"))
    return written
