"""Regenerate every committed figure under docs/figures from the result CSVs.

One command, no arguments needed:

    uv run python scripts/make_figures.py

Reads only `results/`; writes only `docs/figures/`. Nothing here invents a
number: every value plotted comes from a CSV in the results directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clockwork.bench.plots import plot_all
from clockwork.bench.portfolio_plots import plot_portfolio

# Which summary CSV feeds which figure directory. The clockwork sweep owns the
# top level because the README embeds from there; vLLM's sweep is kept beside
# it for comparison.
SWEEPS = (
    (Path("clockwork") / "summary.csv", Path(".")),
    (Path("vllm") / "summary.csv", Path("vllm")),
)


def main() -> None:
    """Render the sweep figures and the headline figures into the figures directory."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--results", default="results", help="directory of benchmark result CSVs")
    parser.add_argument("--figures", default="docs/figures", help="directory for written figures")
    args = parser.parse_args()

    results_dir = Path(args.results)
    figures_dir = Path(args.figures)
    if not results_dir.is_dir():
        sys.exit(f"result dir {results_dir} does not exist; run scripts/run_bench.py first")

    written = []
    for summary_rel, figure_rel in SWEEPS:
        summary = results_dir / summary_rel
        if not summary.is_file():
            print(f"skip {summary}: not present")
            continue
        written += plot_all(summary, figures_dir / figure_rel)
    written += plot_portfolio(results_dir, figures_dir)

    if not written:
        sys.exit(f"no result CSVs under {results_dir}; run scripts/run_bench.py first")
    for figure in written:
        print(f"figure: {figure}")


if __name__ == "__main__":
    main()
