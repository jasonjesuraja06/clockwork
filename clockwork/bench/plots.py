"""Figures from bench summary CSVs; plots measured values only."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

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


def _median_by_rate(rows: list[dict]) -> dict[float, dict]:
    """Collapse repeated runs of the same rate to the median, as docs/results.md reports them."""
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["request_rate"]].append(row)
    return {
        rate: {
            key: median([row[key] for row in group])
            for key in ("ttft_p50_ms", "ttft_p99_ms", "itl_p50_ms", "itl_p99_ms", "output_tok_s")
        }
        for rate, group in grouped.items()
    }


def _ablation_figure(rows: list[dict], path: Path) -> Path:
    # ttft p50 spans two orders of magnitude between the on and off runs, so a
    # linear axis draws the on bars as slivers; log y keeps both readable. The
    # second panel plots output tok/s rather than prefix hit rate, which is
    # zero by construction whenever the cache is off and carries no signal.
    rates = sorted({row["request_rate"] for row in rows})
    fig, (ax_ttft, ax_tok) = plt.subplots(1, 2, figsize=(9, 4.6))
    width = 0.35
    variants = (
        (-width / 2, "True", "radix on", "#0072B2"),
        (width / 2, "False", "radix off", "#E69F00"),
    )
    by_variant: dict[str, dict[float, dict]] = {}
    for offset, enabled, label, color in variants:
        subset = _median_by_rate([row for row in rows if row["radix_enabled"] == enabled])
        by_variant[enabled] = subset
        present = [rate for rate in rates if rate in subset]
        xs = [rates.index(rate) + offset for rate in present]
        ttfts = [subset[rate]["ttft_p50_ms"] for rate in present]
        toks = [subset[rate]["output_tok_s"] for rate in present]
        ax_ttft.bar(xs, ttfts, width=width, label=label, color=color, zorder=3)
        ax_tok.bar(xs, toks, width=width, label=label, color=color, zorder=3)
        for ax, values in ((ax_ttft, ttfts), (ax_tok, toks)):
            for x, value in zip(xs, values, strict=True):
                ax.text(x, value, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
    on, off = by_variant.get("True", {}), by_variant.get("False", {})
    ratios = [
        on[rate]["output_tok_s"] / off[rate]["output_tok_s"]
        for rate in rates
        if rate in on and rate in off and off[rate]["output_tok_s"]
    ]
    for ax, ylabel in ((ax_ttft, "ttft p50 (ms), log scale"), (ax_tok, "output tokens per second")):
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{rate:g}" for rate in rates])
        ax.set_xlabel("request rate (req/s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
    ax_ttft.set_yscale("log")
    ax_ttft.set_ylim(
        bottom=min(row["ttft_p50_ms"] for row in rows) / 2.5,
        top=max(row["ttft_p50_ms"] for row in rows) * 4.0,
    )
    ax_tok.set_ylim(top=max(row["output_tok_s"] for row in rows) * 1.22)
    if ratios:
        # The label follows the workload's `radix_enabled` flag, the same column
        # docs/results.md prints, so the title claims a measured ratio and not
        # an engine behaviour the CSV cannot vouch for.
        fig.suptitle(
            f"Prefix cache flag on versus off, identical traces: output tok/s"
            f" {min(ratios):.2f}x to {max(ratios):.2f}x",
            fontsize=12,
            fontweight="bold",
        )
        for rate, ratio in zip(
            [rate for rate in rates if rate in on and rate in off], ratios, strict=True
        ):
            ax_tok.text(
                rates.index(rate),
                max(on[rate]["output_tok_s"], off[rate]["output_tok_s"]) * 1.09,
                f"{ratio:.2f}x",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
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
