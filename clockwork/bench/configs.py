"""Benchmark workload matrix; names encode kind, key parameters, and request rate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadConfig:
    name: str
    kind: str  # "singleturn" | "agent" | "ablation"
    request_rate: float
    num_requests: int
    seed: int
    radix_enabled: bool = True
    max_tokens: int = 128
    prompt_len_log_mean: float = 5.4
    prompt_len_log_sigma: float = 0.8
    prompt_len_min: int = 16
    prompt_len_max: int = 3072
    output_len_log_mean: float = 4.6
    output_len_log_sigma: float = 0.7
    output_len_min: int = 8
    shared_prefix_tokens: int = 1536
    turns_min: int = 6
    turns_max: int = 12
    suffix_tokens_mean: int = 96
    reply_tokens_mean: int = 64
    think_time_mean_s: float = 0.25
    arrival_process: str = "poisson"  # "poisson" | "pareto"
    pareto_alpha: float = 1.5
    # Ablation pairs share a trace_name (empty means use name) so both runs
    # replay a byte-identical request list.
    trace_name: str = ""
    # Real single-turn conversation trace. Empty means the default data/sharegpt.json
    # at the repo root; when that file is missing the generator synthesizes lengths
    # from the seeded sampler instead and warns that it did so.
    real_trace_path: str = ""


def _synthetic_singleturn(rate: float, seed: int) -> WorkloadConfig:
    # Named synthetic_ because no benchmark run of this repository has had
    # data/sharegpt.json on disk, so these configs have only ever produced sampler
    # output. Drop the prefix only after shipping the real trace and rerunning.
    return WorkloadConfig(
        name=f"synthetic_singleturn_r{rate:g}",
        kind="singleturn",
        request_rate=rate,
        num_requests=128,
        seed=seed,
        max_tokens=256,
    )


def _agent(
    prefix: int, turns_min: int, turns_max: int, suffix: int, process: str, rate: float, seed: int
) -> WorkloadConfig:
    proc = "pois" if process == "poisson" else "pareto"
    return WorkloadConfig(
        name=f"agent_p{prefix}_t{turns_min}to{turns_max}_{proc}_r{rate:g}",
        kind="agent",
        request_rate=rate,
        num_requests=96,
        seed=seed,
        shared_prefix_tokens=prefix,
        turns_min=turns_min,
        turns_max=turns_max,
        suffix_tokens_mean=suffix,
        arrival_process=process,
    )


def _ablation(radix_enabled: bool, rate: float, seed: int) -> WorkloadConfig:
    state = "on" if radix_enabled else "off"
    return WorkloadConfig(
        name=f"ablation_p1536_radix_{state}_r{rate:g}",
        kind="ablation",
        request_rate=rate,
        num_requests=96,
        seed=seed,
        radix_enabled=radix_enabled,
        shared_prefix_tokens=1536,
        turns_min=6,
        turns_max=12,
        suffix_tokens_mean=96,
        trace_name=f"ablation_p1536_r{rate:g}",
    )


WORKLOADS: list[WorkloadConfig] = [
    _synthetic_singleturn(1.0, seed=101),
    _synthetic_singleturn(2.0, seed=102),
    _synthetic_singleturn(4.0, seed=103),
    _synthetic_singleturn(8.0, seed=104),
    _synthetic_singleturn(16.0, seed=105),
    # The agent traces below are self-designed by this project, not a public dataset.
    _agent(1024, 4, 8, 64, "poisson", 2.0, seed=201),
    _agent(1024, 4, 8, 64, "poisson", 4.0, seed=202),
    _agent(1024, 4, 8, 64, "poisson", 8.0, seed=203),
    _agent(1024, 4, 8, 64, "pareto", 4.0, seed=204),
    _agent(1536, 6, 12, 96, "poisson", 2.0, seed=205),
    _agent(1536, 6, 12, 96, "poisson", 4.0, seed=206),
    _agent(1536, 6, 12, 96, "poisson", 8.0, seed=207),
    _agent(1536, 6, 12, 96, "pareto", 4.0, seed=208),
    _agent(1536, 6, 12, 96, "pareto", 8.0, seed=209),
    _agent(2048, 8, 16, 128, "poisson", 4.0, seed=210),
    _agent(2048, 8, 16, 128, "poisson", 8.0, seed=211),
    _agent(2048, 8, 16, 128, "pareto", 8.0, seed=212),
    # Ablation pairs share a seed so the on and off runs replay the identical trace.
    _ablation(True, 2.0, seed=301),
    _ablation(False, 2.0, seed=301),
    _ablation(True, 8.0, seed=302),
    _ablation(False, 8.0, seed=302),
]


def get_workload(name: str) -> WorkloadConfig:
    """Look up one workload configuration by name."""
    for cfg in WORKLOADS:
        if cfg.name == name:
            return cfg
    raise KeyError(f"unknown workload {name!r}")
