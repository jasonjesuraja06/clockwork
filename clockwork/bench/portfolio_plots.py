"""Headline figures for the README; every plotted value is read from a results CSV.

Sibling of `plots.py`. `plots.py` renders the per-rate sweep figures from one
summary CSV; this module renders the three cross-file comparison figures that
need more than one input (clockwork against vLLM, the decode microbenchmark,
and the cold versus warm TTFT replay).

Run `python scripts/make_figures.py` to regenerate everything under
`docs/figures/`.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from clockwork.bench.metrics import percentile  # noqa: E402

# Okabe-Ito, safe for deuteranopia and protanopia; no red/green pair.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GRAY = "#595959"
LIGHT_GRAY = "#BBBBBB"

TITLE_SIZE = 13.5
SUBTITLE_SIZE = 9.5
NOTE_SIZE = 8.5


def _read_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _title(fig, ax, headline: str, subtitle: str) -> None:
    """One bold takeaway line above a smaller line of scope."""
    fig.suptitle(headline, fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_title(subtitle, fontsize=SUBTITLE_SIZE, color=GRAY)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _workload_label(row: dict) -> str:
    """Turn `agent_p1536_t6to12_pois_r8` into `p1536 t6to12 pois, 8 req/s`."""
    parts = row["workload"].split("_")
    if parts and parts[0] == "agent":
        parts = parts[1:]
    if parts and parts[-1].startswith("r") and parts[-1][1:].isdigit():
        parts = parts[:-1]
    rate = float(row["request_rate"])
    return f"{' '.join(parts)}, {rate:g} req/s"


def _agent_throughput(rows: list[dict]) -> tuple[dict[str, float], dict[str, dict]]:
    """Median output tok/s per agent workload, plus one representative row each."""
    values: dict[str, list[float]] = {}
    sample: dict[str, dict] = {}
    for row in rows:
        if row.get("kind") != "agent":
            continue
        values.setdefault(row["workload"], []).append(float(row["output_tok_s"]))
        sample.setdefault(row["workload"], row)
    return {name: statistics.median(v) for name, v in values.items()}, sample


def agent_throughput_figure(
    clockwork_summary: str | Path, vllm_summary: str | Path, path: Path
) -> Path:
    """Grouped horizontal bars, clockwork against vLLM on every shared agent workload."""
    clockwork, sample = _agent_throughput(_read_csv(clockwork_summary))
    vllm, _ = _agent_throughput(_read_csv(vllm_summary))
    shared = sorted(set(clockwork) & set(vllm), key=lambda name: clockwork[name] / vllm[name])
    if not shared:
        raise ValueError("no agent workload appears in both summary CSVs")
    ratios = [clockwork[name] / vllm[name] for name in shared]
    wins = sum(1 for ratio in ratios if ratio > 1.0)

    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    height = 0.38
    for index, name in enumerate(shared):
        ax.barh(index + height / 2, clockwork[name], height=height, color=BLUE, zorder=3)
        ax.barh(index - height / 2, vllm[name], height=height, color=ORANGE, zorder=3)
        ax.text(
            clockwork[name] + 4,
            index + height / 2,
            f"{clockwork[name]:.0f}",
            va="center",
            fontsize=8.5,
            color=BLUE,
        )
        ax.text(
            vllm[name] + 4,
            index - height / 2,
            f"{vllm[name]:.0f}",
            va="center",
            fontsize=8.5,
            color="#8A6100",
        )

    top = max(max(clockwork[name] for name in shared), max(vllm[name] for name in shared))
    ratio_x = top * 1.20
    for index, ratio in enumerate(ratios):
        ax.text(
            ratio_x,
            index,
            f"{ratio:.2f}x",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#222222",
        )
    ax.text(
        ratio_x,
        len(shared) - 0.45,
        "ratio",
        va="center",
        ha="left",
        fontsize=8.5,
        color=GRAY,
    )

    ax.set_yticks(range(len(shared)))
    ax.set_yticklabels([_workload_label(sample[name]) for name in shared], fontsize=9)
    ax.set_xlim(0, top * 1.36)
    ax.set_xlabel("output tokens per second, higher is better")
    ax.grid(True, axis="x", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # Series are named on the top pair instead of in a legend, so the reader
    # never has to map a color box back onto a bar.
    top_index = len(shared) - 1
    for value, y_offset, color, label in (
        (clockwork[shared[-1]], height / 2, BLUE, "clockwork"),
        (vllm[shared[-1]], -height / 2, "#8A6100", "vLLM"),
    ):
        ax.text(
            value + top * 0.09,
            top_index + y_offset,
            label,
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    _title(
        fig,
        ax,
        f"clockwork output throughput beats vLLM on {wins} of {len(shared)} agent workloads,"
        f" {min(ratios):.2f}x to {max(ratios):.2f}x",
        "Tesla T4, Qwen2.5-1.5B-Instruct float16, same traces through the same harness.\n"
        "clockwork cells are the median of 2 runs, vLLM ran once.",
    )
    fig.text(
        0.5,
        0.012,
        f"Scope: these {len(shared)} agent traces are self-designed by this project. Their long"
        " shared prefixes are the structure clockwork exists to exploit,\nso the comparison favors"
        " clockwork by construction. On the 5 ShareGPT single-turn workloads the order reverses:"
        " vLLM leads 575.0 to 464.5 tok/s.",
        ha="center",
        va="top",
        fontsize=NOTE_SIZE,
        color=GRAY,
    )
    return _save(fig, path)


def decode_speedup_figure(microbench_csv: str | Path, path: Path) -> Path:
    """Per-shape Triton speedup over the torch reference, with the one loss called out."""
    rows = _read_csv(microbench_csv)
    shapes = sorted(
        (
            {
                "batch": int(row["batch"]),
                "ctx_len": int(row["ctx_len"]),
                "torch_ms": float(row["torch_ms"]),
                "triton_ms": float(row["triton_ms"]),
                "speedup": float(row["speedup"]),
            }
            for row in rows
        ),
        key=lambda shape: shape["speedup"],
    )
    wins = [shape for shape in shapes if shape["speedup"] > 1.0]
    losses = [shape for shape in shapes if shape["speedup"] <= 1.0]
    peak = shapes[-1]

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    colors = [BLUE if shape["speedup"] > 1.0 else ORANGE for shape in shapes]
    positions = list(range(len(shapes)))
    ax.barh(positions, [shape["speedup"] for shape in shapes], color=colors, zorder=3)
    ax.axvline(1.0, color=GRAY, linestyle="--", linewidth=1.4, zorder=4)
    ax.text(1.12, len(shapes) - 0.42, "1.0x parity with torch", fontsize=9, color=GRAY, va="center")

    # A fixed right-hand column carries the absolute times, so no reader has to
    # trust a ratio without seeing the two numbers behind it.
    times_x = peak["speedup"] * 1.20
    ax.text(
        times_x,
        len(shapes) - 0.42,
        "torch / Triton (ms)",
        fontsize=8.5,
        color=GRAY,
        va="center",
    )
    for index, shape in enumerate(shapes):
        wins_here = shape["speedup"] > 1.0
        suffix = ""
        if shape is peak:
            suffix = "  peak"
        elif not wins_here:
            suffix = "  the only shape torch wins"
        ax.text(
            shape["speedup"] + 0.14,
            index,
            f"{shape['speedup']:.2f}x{suffix}",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=BLUE if wins_here else "#8A6100",
        )
        ax.text(
            times_x,
            index,
            f"{shape['torch_ms']:.3f} / {shape['triton_ms']:.3f}",
            va="center",
            fontsize=8.5,
            color="#333333",
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"batch {shape['batch']}, ctx {shape['ctx_len']}" for shape in shapes], fontsize=9
    )
    ax.set_ylim(-0.75, len(shapes) - 0.15)
    ax.set_xlim(0, peak["speedup"] * 1.42)
    ax.set_xlabel("Triton speedup over the torch paged-decode reference, higher is better")
    ax.grid(True, axis="x", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    loss_note = (
        " The lead shrinks as context grows and reverses at "
        + ", ".join(f"batch {loss['batch']}, ctx {loss['ctx_len']}" for loss in losses)
        + "."
        if losses
        else " Every measured shape favors Triton."
    )
    _title(
        fig,
        ax,
        f"Triton paged decode beats the torch fallback on {len(wins)} of {len(shapes)} shapes,"
        f" up to {peak['speedup']:.2f}x",
        "Tesla T4 float16, one decode step per shape, wall time in ms." + loss_note,
    )
    return _save(fig, path)


def _ttfts(request_csv: str | Path) -> list[float]:
    rows = _read_csv(request_csv)
    return [float(row["ttft_ms"]) for row in rows if not row.get("error") and row.get("ttft_ms")]


def _cdf(values: list[float]) -> tuple[list[float], list[float]]:
    ordered = sorted(values)
    n = len(ordered)
    return ordered, [(index + 1) / n for index in range(n)]


def _check_derived(derived_csv: Path, computed: dict[str, float]) -> None:
    """Fail loudly if the plotted percentiles drift from the recorded derived metrics."""
    if not derived_csv.is_file():
        return
    recorded = {row["metric"]: float(row["value"]) for row in _read_csv(derived_csv)}
    mismatches = [
        f"{metric}: plotted {round(value, 1)} vs {derived_csv.name} {recorded[metric]}"
        for metric, value in computed.items()
        if metric in recorded and abs(round(value, 1) - recorded[metric]) > 0.05
    ]
    if mismatches:
        raise ValueError(
            "derived metrics disagree with the per-request CSVs: " + "; ".join(mismatches)
        )


def ttft_cdf_figure(
    cold_csv: str | Path, warm_csv: str | Path, path: Path, derived_csv: str | Path | None = None
) -> Path:
    """TTFT distribution of a cold trace and its identical warm replay, as a log-x CDF."""
    cold, warm = _ttfts(cold_csv), _ttfts(warm_csv)
    stats = {
        "ttft_p50_cold_ms": percentile(cold, 50),
        "ttft_p99_cold_ms": percentile(cold, 99),
        "ttft_p50_warm_ms": percentile(warm, 50),
        "ttft_p99_warm_ms": percentile(warm, 99),
    }
    cold_50, warm_50 = stats["ttft_p50_cold_ms"], stats["ttft_p50_warm_ms"]
    cold_99, warm_99 = stats["ttft_p99_cold_ms"], stats["ttft_p99_warm_ms"]
    cut_50 = 100.0 * (cold_50 - warm_50) / cold_50
    cut_99 = 100.0 * (cold_99 - warm_99) / cold_99
    if derived_csv is not None:
        _check_derived(
            Path(derived_csv), stats | {"ttft_p50_cut_pct": cut_50, "ttft_p99_cut_pct": cut_99}
        )

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    cold_x, cold_y = _cdf(cold)
    warm_x, warm_y = _cdf(warm)
    ax.step(cold_x, cold_y, where="post", color=BLUE, linewidth=2.4, zorder=3)
    ax.step(warm_x, warm_y, where="post", color=ORANGE, linewidth=2.4, zorder=3)

    # Curves are named where they run instead of in a legend, and each
    # percentile marker carries its own value, so nothing has to be read off
    # the axis. Cold labels sit right of their marker, warm labels left.
    ax.annotate(
        "cold pass, empty prefix cache",
        xy=(cold_x[int(0.62 * len(cold_x))], 0.62),
        xytext=(cold_x[int(0.62 * len(cold_x))] * 1.7, 0.42),
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=BLUE,
        arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 0.9},
    )
    ax.annotate(
        "warm replay, same trace",
        xy=(warm_x[int(0.55 * len(warm_x))], 0.55),
        xytext=(min(warm) * 1.05, 0.78),
        ha="left",
        fontsize=10,
        fontweight="bold",
        color="#8A6100",
        arrowprops={"arrowstyle": "-", "color": "#8A6100", "linewidth": 0.9},
    )

    marks = (
        (BLUE, "left", 1.06, cold_50, 0.435, cold_99),
        ("#8A6100", "right", 0.90, warm_50, 0.565, warm_99),
    )
    for color, align, shift, p50, p50_y, p99 in marks:
        for value, y, text_y in ((p50, 0.5, p50_y), (p99, 1.0, 1.07)):
            ax.plot([value], [y], "o", color=color, zorder=6, ms=7)
            ax.text(
                value * shift,
                text_y,
                f"{value:.0f} ms",
                ha=align,
                va="center",
                fontsize=9.5,
                color=color,
                fontweight="bold",
                zorder=6,
            )
    for value, y, label in (
        ((warm_50 * cold_50) ** 0.5, 0.5, "p50"),
        ((warm_99 * cold_99) ** 0.5, 1.0, "p99"),
    ):
        ax.text(
            value,
            y + 0.035,
            label,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#222222",
            fontweight="bold",
            zorder=6,
        )
    for warm_value, cold_value, y in ((warm_50, cold_50, 0.5), (warm_99, cold_99, 1.0)):
        ax.annotate(
            "",
            xy=(warm_value, y),
            xytext=(cold_value, y),
            zorder=5,
            arrowprops={"arrowstyle": "->", "color": "#222222", "linewidth": 1.5},
        )

    ax.text(
        0.985,
        0.04,
        f"p50  {cold_50:.0f} ms cold to {warm_50:.0f} ms warm,  down {cut_50:.1f}%\n"
        f"p99  {cold_99:.0f} ms cold to {warm_99:.0f} ms warm,  down {cut_99:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#222222",
        zorder=7,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": LIGHT_GRAY},
    )

    ax.set_xscale("log")
    ax.set_xlim(min(min(cold), min(warm)) * 0.72, max(max(cold), max(warm)) * 2.6)
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("time to first token (ms, log scale)")
    ax.set_ylabel("fraction of requests at or below")
    ax.grid(True, alpha=0.3, which="both", zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(1.0, color=LIGHT_GRAY, linewidth=0.8, zorder=1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    _title(
        fig,
        ax,
        f"Warm replay cuts time to first token p99 by {cut_99:.1f} percent"
        f" and p50 by {cut_50:.1f} percent",
        f"Tesla T4, clockwork with the radix prefix cache on, {len(cold)} requests per pass on the"
        " same seeded agent trace at 2 req/s.",
    )
    fig.text(
        0.5,
        -0.03,
        f"Both passes use the identical trace and config; only the cache state differs."
        f" Percentiles are nearest rank, so p99 of {len(cold)} requests is the\nlargest observed"
        " value and sits at the top of each curve. The gap at the median is smaller than the gap"
        " in the tail.",
        ha="center",
        va="top",
        fontsize=NOTE_SIZE,
        color=GRAY,
    )
    return _save(fig, path)


def plot_portfolio(results_dir: str | Path, out_dir: str | Path = "docs/figures") -> list[Path]:
    """Render the three headline figures; skips any whose input CSVs are absent."""
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    written: list[Path] = []

    clockwork_summary = results_dir / "clockwork" / "summary.csv"
    vllm_summary = results_dir / "vllm" / "summary.csv"
    if clockwork_summary.is_file() and vllm_summary.is_file():
        written.append(
            agent_throughput_figure(
                clockwork_summary, vllm_summary, out_dir / "agent_throughput_vs_vllm.png"
            )
        )

    microbench = results_dir / "microbench_decode.csv"
    if microbench.is_file():
        written.append(decode_speedup_figure(microbench, out_dir / "decode_kernel_speedup.png"))

    cold = results_dir / "ttft_cold" / "agent_p1536_t6to12_pois_r2.csv"
    warm = results_dir / "ttft_warm" / "agent_p1536_t6to12_pois_r2.csv"
    if cold.is_file() and warm.is_file():
        written.append(
            ttft_cdf_figure(
                cold, warm, out_dir / "ttft_cold_vs_warm.png", results_dir / "derived_metrics.csv"
            )
        )
    return written


def main() -> None:
    """Render the headline figures from a results directory."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--results", default="results", help="directory of benchmark result CSVs")
    parser.add_argument("--figures", default="docs/figures", help="directory for written figures")
    args = parser.parse_args()
    for figure in plot_portfolio(args.results, args.figures):
        print(f"figure: {figure}")


if __name__ == "__main__":
    main()
