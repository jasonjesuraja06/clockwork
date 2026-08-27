"""Collect bench result directories into markdown tables and a results zip."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path

from clockwork.bench.metrics import summarize
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


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "results"


def _bundle(
    out_dir: Path,
    suffix: str,
    bench_dir: Path,
    result_dirs: list[Path],
    markdown_path: Path,
    figures_dir: Path,
    env_path: Path | None,
) -> Path:
    zip_path = out_dir / f"clockwork_results_{suffix}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(markdown_path, markdown_path.name)
        if env_path is not None:
            bundle.write(env_path, env_path.name)
        for result_dir in result_dirs:
            rel = result_dir.relative_to(bench_dir) if result_dir != bench_dir else Path()
            for path in sorted(result_dir.glob("*.csv")):
                bundle.write(path, Path(bench_dir.name) / rel / path.name)
        if figures_dir.is_dir():
            for path in sorted(figures_dir.rglob("*.png")):
                bundle.write(path, Path("figures") / path.relative_to(figures_dir))
    return zip_path


def main() -> None:
    """Print markdown tables for every result directory and bundle a results zip."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench-dir",
        default="bench_results",
        help="directory of run_bench.py CSVs, or a directory of such directories",
    )
    parser.add_argument("--figures", default="docs/figures", help="figures directory to bundle")
    parser.add_argument("--env", default=None, help="JSON file describing the machine")
    parser.add_argument("--out-dir", default=None, help="where results.md and the zip land")
    args = parser.parse_args()

    bench_dir = Path(args.bench_dir)
    if not bench_dir.is_dir():
        sys.exit(f"bench dir {bench_dir} does not exist; run scripts/run_bench.py first")
    result_dirs = _result_dirs(bench_dir)
    if not result_dirs:
        sys.exit(f"no bench CSVs under {bench_dir}; run scripts/run_bench.py first")
    out_dir = Path(args.out_dir) if args.out_dir else bench_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    env_path = None
    env: dict = {}
    if args.env is not None:
        env_path = Path(args.env)
        if not env_path.is_file():
            sys.exit(f"env file {env_path} does not exist")
        env = json.loads(env_path.read_text(encoding="utf-8"))

    sections = ["# Benchmark results"]
    if env:
        sections.append("\n".join(f"- {key}: {value}" for key, value in env.items()))
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

    markdown_path = out_dir / "results.md"
    markdown_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"wrote {markdown_path}")
    suffix = _slug(str(env.get("gpu_name", "")) or bench_dir.name)
    zip_path = _bundle(
        out_dir, suffix, bench_dir, result_dirs, markdown_path, Path(args.figures), env_path
    )
    print(f"wrote {zip_path}")


if __name__ == "__main__":
    main()
