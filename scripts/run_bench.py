"""CLI benchmark driver: runs workloads against a server and writes CSVs and figures."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
from pathlib import Path

import httpx

from clockwork.bench import runner
from clockwork.bench.configs import WORKLOADS, WorkloadConfig
from clockwork.bench.workloads import HashWordTokenizer


def _fetch_model_id(base_url: str) -> str | None:
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/v1/models", timeout=10.0)
        resp.raise_for_status()
        return resp.json()["data"][0]["id"]
    except Exception:
        return None


def _resolve_tokenizer(spec: str, base_url: str):
    if spec == "builtin":
        return HashWordTokenizer()
    name = _fetch_model_id(base_url) if spec == "auto" else spec
    if name is not None:
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(name)
        except Exception as exc:
            if spec != "auto":
                raise SystemExit(f"cannot load tokenizer {spec!r}: {exc}") from exc
    print("model tokenizer unavailable, using the builtin vocabulary tokenizer")
    return HashWordTokenizer()


def _select_workloads(spec: str) -> list[WorkloadConfig]:
    if spec == "all":
        return list(WORKLOADS)
    by_name = {cfg.name: cfg for cfg in WORKLOADS}
    selected = []
    for name in spec.split(","):
        name = name.strip()
        if name not in by_name:
            raise SystemExit(f"unknown workload {name!r}; known: {', '.join(sorted(by_name))}")
        selected.append(by_name[name])
    return selected


def main(argv: list[str] | None = None) -> None:
    """Run the selected benchmark workloads and write per-workload plus summary CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default="all", help="all, or comma-separated workload names")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", default="bench_results", help="output directory for CSVs")
    parser.add_argument(
        "--tokenizer",
        default="auto",
        help="auto (served model), builtin (vocabulary sampler), or a Hugging Face name",
    )
    parser.add_argument("--num-requests", type=int, default=None, help="override per workload")
    parser.add_argument("--max-tokens", type=int, default=None, help="override per workload")
    parser.add_argument(
        "--ignore-eos", action="store_true", help="force generation to run to max_tokens"
    )
    parser.add_argument("--plots", default=None, help="also render figures into this directory")
    args = parser.parse_args(argv)

    workloads = _select_workloads(args.configs)
    tokenizer = _resolve_tokenizer(args.tokenizer, args.base_url)
    overrides = {}
    if args.num_requests is not None:
        overrides["num_requests"] = args.num_requests
    if args.max_tokens is not None:
        overrides["max_tokens"] = args.max_tokens
    out_dir = Path(args.out)
    for cfg in workloads:
        if overrides:
            cfg = dataclasses.replace(cfg, **overrides)
        csv_path = asyncio.run(
            runner.run(cfg, args.base_url, out_dir, tokenizer=tokenizer, ignore_eos=args.ignore_eos)
        )
        print(f"{cfg.name}: wrote {csv_path}")
    summary_path = out_dir / "summary.csv"
    print(f"summary: {summary_path}")
    if args.plots is not None:
        from clockwork.bench.plots import plot_all

        for figure in plot_all(summary_path, args.plots):
            print(f"figure: {figure}")


if __name__ == "__main__":
    sys.exit(main())
