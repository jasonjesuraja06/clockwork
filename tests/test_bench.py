"""Bench harness gates: workload determinism, metrics math, runner end to end, plots."""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict, replace

import httpx
import pytest

from clockwork.bench import runner
from clockwork.bench.configs import WORKLOADS, WorkloadConfig
from clockwork.bench.metrics import percentile, summarize
from clockwork.bench.plots import plot_all
from clockwork.bench.workloads import HashWordTokenizer, generate
from clockwork.config import EngineConfig
from clockwork.engine.async_engine import AsyncLLMEngine
from clockwork.engine.loader import build_tiny_qwen2
from clockwork.server.app import build_app

TIMEOUT = 120.0


def first_of_kind(kind: str) -> WorkloadConfig:
    return next(cfg for cfg in WORKLOADS if cfg.kind == kind)


def test_generate_is_deterministic():
    tokenizer = HashWordTokenizer()
    for base in (first_of_kind("singleturn"), first_of_kind("agent")):
        cfg = replace(base, num_requests=12)
        one = generate(cfg, tokenizer)
        two = generate(cfg, tokenizer)
        assert one == two
        assert [r.arrival_time for r in one] == [r.arrival_time for r in two]
        assert len(one) == 12
        arrivals = [r.arrival_time for r in one]
        assert arrivals == sorted(arrivals)
        assert all(r.arrival_time >= 0.0 for r in one)


def test_workload_matrix_shape():
    assert len(WORKLOADS) >= 20
    names = [cfg.name for cfg in WORKLOADS]
    assert len(names) == len(set(names))
    kinds = {cfg.kind for cfg in WORKLOADS}
    assert kinds == {"singleturn", "agent", "ablation"}
    for cfg in WORKLOADS:
        assert isinstance(cfg.seed, int)
        assert cfg.num_requests > 0
        assert cfg.request_rate > 0
    singleturn_rates = {cfg.request_rate for cfg in WORKLOADS if cfg.kind == "singleturn"}
    assert len(singleturn_rates) >= 3


def test_no_workload_claims_a_dataset_the_repo_does_not_ship():
    # The shipped matrix has no real conversation trace on disk, so no workload name
    # or kind may imply one. Renaming a config back to sharegpt means shipping the
    # dataset and rerunning; until then this guard keeps the published labels honest.
    for cfg in WORKLOADS:
        assert "sharegpt" not in cfg.name.lower(), cfg.name
        assert "sharegpt" not in cfg.kind.lower(), cfg.name
    assert {cfg.name for cfg in WORKLOADS if cfg.kind == "singleturn"} == {
        f"synthetic_singleturn_r{rate:g}" for rate in (1, 2, 4, 8, 16)
    }


def test_ablation_pairs_differ_only_in_radix_enabled():
    pairs: dict[int, list[WorkloadConfig]] = {}
    for cfg in WORKLOADS:
        if cfg.kind == "ablation":
            pairs.setdefault(cfg.seed, []).append(cfg)
    assert pairs, "no ablation workloads"
    for seed, pair in pairs.items():
        assert len(pair) == 2, f"seed {seed} is not a pair"
        assert {cfg.radix_enabled for cfg in pair} == {True, False}
        on, off = (asdict(cfg) for cfg in pair)
        for key in ("name", "radix_enabled"):
            on.pop(key)
            off.pop(key)
        assert on == off, f"ablation pair for seed {seed} differs beyond radix_enabled"


def test_ablation_pair_traces_are_identical():
    tokenizer = HashWordTokenizer()
    pair = [cfg for cfg in WORKLOADS if cfg.kind == "ablation" and cfg.seed == 301]
    assert len(pair) == 2
    on = replace(next(c for c in pair if c.radix_enabled), num_requests=8)
    off = replace(next(c for c in pair if not c.radix_enabled), num_requests=8)
    assert generate(on, tokenizer) == generate(off, tokenizer)


def test_singleturn_samples_the_real_trace_when_the_file_is_present(tmp_path, caplog):
    data = [
        {
            "conversations": [
                {"from": "human", "value": "alpha beta gamma delta"},
                {"from": "gpt", "value": "one two three"},
            ]
        },
        {"conversations": [{"from": "gpt", "value": "reply with no prompt"}]},
        {
            "conversations": [
                {"from": "human", "value": "epsilon zeta"},
                {"from": "gpt", "value": "four five six seven eight nine ten"},
            ]
        },
    ]
    trace_path = tmp_path / "sharegpt.json"
    trace_path.write_text(json.dumps(data), encoding="utf-8")
    cfg = replace(
        first_of_kind("singleturn"),
        name="singleturn_local_trace",
        num_requests=6,
        prompt_len_min=1,
        prompt_len_max=64,
        real_trace_path=str(trace_path),
    )
    tokenizer = HashWordTokenizer()
    with caplog.at_level(logging.DEBUG, logger="clockwork.bench.workloads"):
        requests = generate(cfg, tokenizer)
    allowed = {"alpha beta gamma delta", "epsilon zeta"}
    assert len(requests) == 6
    # Every prompt is verbatim trace text, so the synthetic sampler was not used.
    assert {r.messages[0]["content"] for r in requests} <= allowed
    assert generate(cfg, tokenizer) == requests
    assert not [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert any("real trace" in rec.getMessage() for rec in caplog.records)


def test_singleturn_warns_loudly_when_the_real_trace_is_absent(tmp_path, caplog):
    missing = tmp_path / "sharegpt.json"
    cfg = replace(
        first_of_kind("singleturn"),
        name="singleturn_missing_trace",
        num_requests=4,
        real_trace_path=str(missing),
    )
    with caplog.at_level(logging.WARNING, logger="clockwork.bench.workloads"):
        requests = generate(cfg, tokenizer=HashWordTokenizer())
    assert len(requests) == 4
    warnings = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert len(warnings) == 1, "the synthetic fallback must not be silent"
    message = warnings[0].getMessage()
    assert "SYNTHESIZING" in message
    assert "must not be labeled ShareGPT" in message
    assert str(missing) in message


def test_summarize_matches_hand_computed_percentiles(tmp_path):
    csv_path = tmp_path / "unit.csv"
    base = {
        "workload": "unit",
        "kind": "agent",
        "request_rate": "2.0",
        "radix_enabled": "True",
        "seed": "1",
        "session_id": "s0",
        "turn": "0",
        "arrival_s": "0.0",
        "error": "",
    }
    rows = [
        base
        | {
            "request_id": "r0",
            "start_s": "0.0",
            "end_s": "1.0",
            "ttft_ms": "10.0",
            "itl_ms": "1.0;2.0",
            "prompt_tokens": "10",
            "output_tokens": "3",
            "cached_tokens": "0",
        },
        base
        | {
            "request_id": "r1",
            "start_s": "0.5",
            "end_s": "1.5",
            "ttft_ms": "20.0",
            "itl_ms": "3.0",
            "prompt_tokens": "10",
            "output_tokens": "2",
            "cached_tokens": "5",
        },
        base
        | {
            "request_id": "r2",
            "start_s": "0.2",
            "end_s": "2.0",
            "ttft_ms": "30.0",
            "itl_ms": "4.0;5.0",
            "prompt_tokens": "20",
            "output_tokens": "3",
            "cached_tokens": "15",
        },
        base
        | {
            "request_id": "r3",
            "start_s": "0.1",
            "end_s": "1.2",
            "ttft_ms": "40.0",
            "itl_ms": "",
            "prompt_tokens": "10",
            "output_tokens": "1",
            "cached_tokens": "0",
        },
        base
        | {
            "request_id": "r4",
            "start_s": "",
            "end_s": "",
            "ttft_ms": "",
            "itl_ms": "",
            "prompt_tokens": "",
            "output_tokens": "",
            "cached_tokens": "",
            "error": "boom",
        },
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=runner.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(csv_path)
    # Nearest-rank by hand over the four non-error rows:
    # ttft sorted [10, 20, 30, 40]: p50 rank ceil(0.50 * 4) = 2 -> 20,
    #                               p99 rank ceil(0.99 * 4) = 4 -> 40.
    # itl flattened [1, 2, 3, 4, 5]: p50 rank ceil(0.50 * 5) = 3 -> 3,
    #                                p99 rank ceil(0.99 * 5) = 5 -> 5.
    # duration = max(end) - min(start) = 2.0 - 0.0; output_tok_s = 9 / 2.0.
    # hit_rate = (0 + 5 + 15 + 0) / (10 + 10 + 20 + 10) = 20 / 50.
    assert summary["workload"] == "unit"
    assert summary["kind"] == "agent"
    assert summary["request_rate"] == 2.0
    assert summary["num_requests"] == 5
    assert summary["num_errors"] == 1
    assert summary["ttft_p50_ms"] == 20.0
    assert summary["ttft_p99_ms"] == 40.0
    assert summary["itl_p50_ms"] == 3.0
    assert summary["itl_p99_ms"] == 5.0
    assert summary["output_tok_s"] == 4.5
    assert summary["hit_rate"] == 0.4


def test_percentile_single_value():
    assert percentile([7.5], 50) == 7.5
    assert percentile([7.5], 99) == 7.5


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_qwen2(seed=0)


@asynccontextmanager
async def serve_app(tiny):
    model, config = tiny
    cfg = EngineConfig.defaults(
        "tiny-qwen2", block_size=4, num_blocks=128, attention_backend="torch"
    )
    engine = AsyncLLMEngine(cfg, model=model, hf_config=config, tokenizer=HashWordTokenizer())
    app = build_app(cfg, engine=engine)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=TIMEOUT
        ) as client:
            yield client


async def test_runner_end_to_end_smoke(tiny, tmp_path):
    cfg = WorkloadConfig(
        name="smoke_agent",
        kind="agent",
        request_rate=50.0,
        num_requests=4,
        seed=7,
        max_tokens=5,
        shared_prefix_tokens=24,
        turns_min=2,
        turns_max=2,
        suffix_tokens_mean=12,
        reply_tokens_mean=8,
        think_time_mean_s=0.001,
    )
    async with serve_app(tiny) as client:
        csv_path = await runner.run(
            cfg, "http://test", tmp_path, tokenizer=HashWordTokenizer(), client=client
        )
    assert csv_path == tmp_path / "smoke_agent.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == runner.CSV_FIELDS
        rows = list(reader)
    assert len(rows) == 4
    for row in rows:
        assert row["error"] == ""
        assert float(row["ttft_ms"]) > 0.0
        assert int(row["output_tokens"]) == 5
        assert int(row["prompt_tokens"]) > 0
        assert int(row["cached_tokens"]) >= 0
        gaps = [part for part in row["itl_ms"].split(";") if part]
        assert len(gaps) == int(row["output_tokens"]) - 1
    summary_path = tmp_path / "summary.csv"
    assert summary_path.is_file()
    with summary_path.open(newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    assert len(summary_rows) == 1
    assert summary_rows[0]["workload"] == "smoke_agent"
    assert int(summary_rows[0]["num_requests"]) == 4
    assert int(summary_rows[0]["num_errors"]) == 0
    assert float(summary_rows[0]["ttft_p50_ms"]) > 0.0


def test_collect_results_tables(tmp_path, repo_root, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(
        "collect_results", repo_root / "scripts" / "collect_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    bench = tmp_path / "results"
    (bench / "clockwork").mkdir(parents=True)
    (bench / "vllm").mkdir()
    request_row = {
        "workload": "unit",
        "kind": "agent",
        "request_rate": "2.0",
        "radix_enabled": "True",
        "seed": "1",
        "request_id": "r0",
        "session_id": "s0",
        "turn": "0",
        "arrival_s": "0.0",
        "start_s": "0.0",
        "end_s": "1.0",
        "ttft_ms": "10.0",
        "itl_ms": "1.0;2.0",
        "prompt_tokens": "10",
        "output_tokens": "3",
        "cached_tokens": "0",
        "error": "",
    }
    with (bench / "clockwork" / "unit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=runner.CSV_FIELDS)
        writer.writeheader()
        writer.writerow(request_row)
    summary_row = dict.fromkeys(runner.SUMMARY_FIELDS, "")
    summary_row |= {"workload": "unit", "kind": "agent", "ttft_p50_ms": "9.0"}
    with (bench / "vllm" / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=runner.SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(summary_row)
    # A CSV that the runner did not write must stay out of summaries and the zip.
    (bench / "microbench_decode.csv").write_text("batch,ctx_len,speedup\n8,512,1.4\n")
    figures = tmp_path / "figures"
    (figures / "vllm").mkdir(parents=True)
    (figures / "ttft_vs_rate.png").write_bytes(b"png")
    (figures / "vllm" / "ttft_vs_rate.png").write_bytes(b"png")

    monkeypatch.setattr(
        sys,
        "argv",
        ["collect_results.py", "--out", str(bench), "--figures", str(figures), "--no-figures"],
    )
    module.main()
    printed = capsys.readouterr().out
    assert "## clockwork" in printed
    assert "## vllm" in printed
    assert (bench / "clockwork" / "summary.csv").is_file()
    assert not (bench / "summary.csv").exists()
    results_md = (bench / "results.md").read_text(encoding="utf-8")
    assert "## clockwork" in results_md
    assert "## vllm" in results_md
    assert "microbench" not in results_md
    assert len(list(figures.rglob("*.png"))) == 2


def test_plots_write_pngs_into_tmp_path(tmp_path, repo_root):
    values = {
        "num_requests": "8",
        "num_errors": "0",
        "gpu_util_mean": "",
        "gpu_util_max": "",
    }
    rows = [
        values
        | {
            "workload": "sg1",
            "kind": "singleturn",
            "request_rate": "1.0",
            "radix_enabled": "True",
            "seed": "101",
            "ttft_p50_ms": "10.0",
            "ttft_p99_ms": "20.0",
            "itl_p50_ms": "1.0",
            "itl_p99_ms": "2.0",
            "output_tok_s": "100.0",
            "hit_rate": "0.1",
        },
        values
        | {
            "workload": "sg2",
            "kind": "singleturn",
            "request_rate": "2.0",
            "radix_enabled": "True",
            "seed": "102",
            "ttft_p50_ms": "12.0",
            "ttft_p99_ms": "24.0",
            "itl_p50_ms": "1.2",
            "itl_p99_ms": "2.4",
            "output_tok_s": "180.0",
            "hit_rate": "0.1",
        },
        values
        | {
            "workload": "ag1",
            "kind": "agent",
            "request_rate": "1.0",
            "radix_enabled": "True",
            "seed": "201",
            "ttft_p50_ms": "8.0",
            "ttft_p99_ms": "16.0",
            "itl_p50_ms": "0.9",
            "itl_p99_ms": "1.8",
            "output_tok_s": "120.0",
            "hit_rate": "0.5",
        },
        values
        | {
            "workload": "ag2",
            "kind": "agent",
            "request_rate": "2.0",
            "radix_enabled": "True",
            "seed": "202",
            "ttft_p50_ms": "9.0",
            "ttft_p99_ms": "18.0",
            "itl_p50_ms": "1.0",
            "itl_p99_ms": "2.0",
            "output_tok_s": "200.0",
            "hit_rate": "0.5",
        },
        values
        | {
            "workload": "ab_on",
            "kind": "ablation",
            "request_rate": "2.0",
            "radix_enabled": "True",
            "seed": "301",
            "ttft_p50_ms": "7.0",
            "ttft_p99_ms": "14.0",
            "itl_p50_ms": "1.0",
            "itl_p99_ms": "2.0",
            "output_tok_s": "210.0",
            "hit_rate": "0.6",
        },
        values
        | {
            "workload": "ab_off",
            "kind": "ablation",
            "request_rate": "2.0",
            "radix_enabled": "False",
            "seed": "301",
            "ttft_p50_ms": "11.0",
            "ttft_p99_ms": "22.0",
            "itl_p50_ms": "1.1",
            "itl_p99_ms": "2.2",
            "output_tok_s": "190.0",
            "hit_rate": "0.0",
        },
    ]
    summary_path = tmp_path / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=runner.SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    figures_dir = repo_root / "docs" / "figures"
    before = set(figures_dir.iterdir()) if figures_dir.is_dir() else set()
    written = plot_all(summary_path, tmp_path / "figs")
    after = set(figures_dir.iterdir()) if figures_dir.is_dir() else set()
    assert after == before, "plot test wrote into docs/figures"
    names = {path.name for path in written}
    assert names == {
        "ttft_vs_rate.png",
        "itl_vs_rate.png",
        "throughput_vs_rate.png",
        "radix_ablation.png",
    }
    for path in written:
        assert tmp_path in path.parents
        assert path.stat().st_size > 0


def test_notebook_calls_match_script_interfaces():
    import ast
    import json
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks" / "bench_t4.ipynb").read_text(encoding="utf-8"))
    scripts = ("run_bench.py", "collect_results.py", "serve.py")
    used: dict[str, set[str]] = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in ast.walk(tree):
            if not isinstance(node, ast.List):
                continue
            texts = [
                el.value
                for el in node.elts
                if isinstance(el, ast.Constant) and isinstance(el.value, str)
            ]
            for script in scripts:
                if any(text.endswith(f"scripts/{script}") for text in texts):
                    flags = {text for text in texts if text.startswith("--")}
                    used.setdefault(script, set()).update(flags)
    assert used, "notebook never invokes the bench scripts as list literals"
    for script, flags in used.items():
        source = (root / "scripts" / script).read_text(encoding="utf-8")
        declared = set(re.findall(r'add_argument\(\s*"(--[a-z-]+)"', source))
        missing = flags - declared
        assert not missing, f"{script} lacks flags the notebook uses: {sorted(missing)}"
