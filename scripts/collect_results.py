"""Collect bench result CSVs into markdown tables, results.md, and figures."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from clockwork.bench.metrics import summarize
from clockwork.bench.plots import plot_all
from clockwork.bench.runner import CSV_FIELDS, SUMMARY_FIELDS

_RUNNER_HEADER = ",".join(CSV_FIELDS)


def _fmt(value: str) -> str:
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number))
    return f"{number:.4g}"


def _markdown_table(header: list[str], body: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(_fmt(cell) for cell in row) + " |" for row in body]
    return "\n".join(lines)


def _is_runner_csv(path: Path) -> bool:
    # Per-request CSVs written by clockwork.bench.runner are identified by
    # their exact header, which keeps unrelated CSVs (microbenchmarks, notes)
    # out of the summaries.
    try:
        with path.open(encoding="utf-8") as f:
            return f.readline().strip() == _RUNNER_HEADER
    except OSError:
        return False


def _result_dirs(bench_dir: Path) -> list[Path]:
    candidates = [bench_dir] + sorted(p for p in bench_dir.iterdir() if p.is_dir())
    return [
        candidate
        for candidate in candidates
        if (candidate / "summary.csv").is_file()
        or any(_is_runner_csv(p) for p in sorted(candidate.glob("*.csv")))
    ]


def _ensure_summary(result_dir: Path) -> Path:
    # Rebuild summary.csv from the per-request CSVs when the runner's own
    # summary is missing, so partial runs still collect.
    summary_path = result_dir / "summary.csv"
    if summary_path.is_file():
        return summary_path
    workload_csvs = [p for p in sorted(result_dir.glob("*.csv")) if _is_runner_csv(p)]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for path in workload_csvs:
            row = summarize(path)
            row.setdefault("gpu_util_mean", "")
            row.setdefault("gpu_util_max", "")
            writer.writerow(row)
    return summary_path


def main() -> None:
    """Print markdown tables for a directory of bench CSVs, write results.md and figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        required=True,
        help="directory of run_bench.py CSVs, or a directory of such directories",
    )
    parser.add_argument("--figures", default="docs/figures", help="directory for plotted figures")
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    args = parser.parse_args()

    bench_dir = Path(args.out)
    if not bench_dir.is_dir():
        sys.exit(f"result dir {bench_dir} does not exist; run scripts/run_bench.py first")
    result_dirs = _result_dirs(bench_dir)
    if not result_dirs:
        sys.exit(f"no bench CSVs under {bench_dir}; run scripts/run_bench.py first")

    sections = ["# Benchmark results"]
    for result_dir in result_dirs:
        summary_path = _ensure_summary(result_dir)
        with summary_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            print(f"{result_dir.name}: summary.csv has no data rows")
            continue
        table = _markdown_table(rows[0], rows[1:])
        sections.append(f"## {result_dir.name}\n\n{table}")
        print(f"## {result_dir.name}\n")
        print(table)
        print()

    markdown_path = bench_dir / "results.md"
    markdown_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"wrote {markdown_path}")

    if not args.no_figures:
        figures_root = Path(args.figures)
        for result_dir in result_dirs:
            summary_path = result_dir / "summary.csv"
            fig_dir = figures_root if result_dir == bench_dir else figures_root / result_dir.name
            for figure in plot_all(summary_path, fig_dir):
                print(f"figure: {figure}")


if __name__ == "__main__":
    main()
