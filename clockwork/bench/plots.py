"""Figures from bench summary CSVs; plots measured values only."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito, safe for deuteranopia and protanopia; no red/green pair.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GRAY = "#595959"

# One hue per workload family, one line style per percentile, so four series
# need only two colors and the palette stays colorblind-safe.
KIND_COLOR = {"singleturn": BLUE, "agent": ORANGE}
KIND_TEXT = {"singleturn": BLUE, "agent": "#8A6100"}


def _title(fig, ax, headline: str, subtitle: str) -> None:
    """One bold takeaway line above a smaller line of scope."""
    fig.suptitle(headline, fontsize=12.5, fontweight="bold")
    ax.set_title(subtitle, fontsize=9, color=GRAY)


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


def _latency_figure(
    rows: list[dict], p50_key: str, p99_key: str, ylabel: str, label: str, path: Path
) -> Path:
    # Log y: p99 runs an order of magnitude above p50 at the loaded rates, so a
    # linear axis flattens every p50 series onto the baseline.
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    rate_span: list[float] = []
    worst: tuple[float, str, float, float, float] | None = None
    for kind in ("singleturn", "agent"):
        kind_rows = [row for row in rows if row["kind"] == kind]
        if not kind_rows:
            continue
        color = KIND_COLOR[kind]
        series = {}
        for key, style, marker in ((p50_key, "-", "o"), (p99_key, "--", "s")):
            rates, means = _by_rate_mean(kind_rows, key)
            series[key] = (rates, means)
            rate_span += rates
            ax.plot(
                rates,
                means,
                style,
                marker=marker,
                color=color,
                linewidth=2.0,
                markersize=5.5,
                label=f"{kind} {key.split('_')[-2]}",
                zorder=3,
            )
            # The endpoint carries its own value, so the reader never has to
            # trace a marker back to the axis.
            ax.annotate(
                f"{means[-1]:.0f}",
                xy=(rates[-1], means[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
                color=KIND_TEXT[kind],
                zorder=5,
            )
        # Where the tail is furthest from the median: the point of the figure.
        p50_rates, p50_means = series[p50_key]
        p99_by_rate = dict(zip(*series[p99_key], strict=True))
        for rate, p50 in zip(p50_rates, p50_means, strict=True):
            p99 = p99_by_rate.get(rate)
            if not p99 or not p50:
                continue
            if worst is None or p99 / p50 > worst[0]:
                worst = (p99 / p50, kind, rate, p50, p99)

    ax.set_xlabel("request rate (req/s, log scale)")
    ax.set_ylabel(f"{ylabel}, log scale")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    if rate_span:
        ax.set_xlim(min(rate_span) * 0.82, max(rate_span) * 2.15)
    ax.grid(True, alpha=0.3, which="both", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if worst:
        ratio, kind, rate, p50, p99 = worst
        _title(
            fig,
            ax,
            f"{label}: the p99 tail reaches {ratio:.0f}x the p50 median,"
            f" so the median hides the tail",
            f"Widest gap at {kind} {rate:g} req/s: p99 {p99:.0f} ms against p50 {p50:.0f} ms."
            " Solid is p50, dashed is p99; endpoints carry their value in ms.",
        )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _throughput_figure(rows: list[dict], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    peaks = []
    for kind in ("singleturn", "agent"):
        kind_rows = [row for row in rows if row["kind"] == kind]
        if not kind_rows:
            continue
        rates, means = _by_rate_mean(kind_rows, "output_tok_s")
        ax.plot(
            rates,
            means,
            "-o",
            color=KIND_COLOR[kind],
            linewidth=2.2,
            markersize=6,
            label=kind,
            zorder=3,
        )
        for rate, value in zip(rates, means, strict=True):
            ax.annotate(
                f"{value:.0f}",
                xy=(rate, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=KIND_TEXT[kind],
                zorder=5,
            )
        best = max(range(len(means)), key=lambda i: means[i])
        peaks.append(f"{kind} peaks at {means[best]:.0f} tok/s at {rates[best]:g} req/s")

    ax.set_xlabel("request rate (req/s, log scale)")
    ax.set_ylabel("output tokens per second, higher is better")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if peaks:
        _title(
            fig,
            ax,
            "Output throughput: " + "; ".join(peaks),
            "Mean of the runs at each rate. Every point carries its value in output tok/s.",
        )
    fig.savefig(path, dpi=150, bbox_inches="tight")
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
            for key in (
                "ttft_p50_ms",
                "ttft_p99_ms",
                "itl_p50_ms",
                "itl_p99_ms",
                "output_tok_s",
                "hit_rate",
            )
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
    # The legend names the workload's `radix_enabled` flag, not an engine
    # state: whether the engine honored the flag is a separate measurement,
    # and the footnote below reports it from the recorded hit rate.
    variants = (
        (-width / 2, "True", "prefix cache flag on", BLUE),
        (width / 2, "False", "flag off", ORANGE),
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
    # A run where the flag-off rows still report a high prefix hit rate is not
    # an ablation of prefix caching: the engine kept its own cache on. That is
    # true of the vLLM sweep, so the figure has to say it rather than leave the
    # legend implying an engine setting the CSV contradicts.
    hit_on = median([subset["hit_rate"] for subset in on.values()]) if on else 0.0
    hit_off = median([subset["hit_rate"] for subset in off.values()]) if off else 0.0
    note = (
        f"Bars follow the workload's radix_enabled flag. Measured prefix hit rate:"
        f" flag on {hit_on:.2f}, flag off {hit_off:.2f}."
    )
    if hit_off > 0.5:
        note += (
            "\nThe engine kept its own prefix cache on through the flag-off runs,"
            " so these bars are not an ablation of prefix caching."
        )
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=8.5, color=GRAY)
    if ratios:
        # The title claims a measured ratio between the two flag settings and
        # not an engine behaviour the CSV cannot vouch for.
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
                rate_rows,
                "ttft_p50_ms",
                "ttft_p99_ms",
                "time to first token (ms)",
                "Time to first token",
                out_dir / "ttft_vs_rate.png",
            )
        )
        written.append(
            _latency_figure(
                rate_rows,
                "itl_p50_ms",
                "itl_p99_ms",
                "inter-token latency (ms)",
                "Inter-token latency",
                out_dir / "itl_vs_rate.png",
            )
        )
        written.append(_throughput_figure(rate_rows, out_dir / "throughput_vs_rate.png"))
    ablation_rows = [row for row in rows if row["kind"] == "ablation"]
    if ablation_rows:
        written.append(_ablation_figure(ablation_rows, out_dir / "radix_ablation.png"))
    return written
