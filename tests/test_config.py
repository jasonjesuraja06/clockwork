from pathlib import Path

import pytest
import torch

from clockwork.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_builds_full_config():
    cfg = EngineConfig.defaults("Qwen/Qwen2.5-1.5B-Instruct")
    assert cfg.model.model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.model.dtype == "float32"
    assert cfg.model.device == "cpu"
    assert cfg.model.seed == 0
    assert cfg.cache == CacheConfig()
    assert cfg.scheduler == SchedulerConfig()
    assert cfg.attention_backend == "auto"


def test_defaults_routes_overrides_to_sub_configs():
    cfg = EngineConfig.defaults(
        "m",
        dtype="float16",
        block_size=32,
        max_num_seqs=8,
        preemption_mode="swap",
        attention_backend="torch",
    )
    assert cfg.model.dtype == "float16"
    assert cfg.cache.block_size == 32
    assert cfg.scheduler.max_num_seqs == 8
    assert cfg.scheduler.preemption_mode == "swap"
    assert cfg.attention_backend == "torch"


def test_defaults_max_model_len_sets_model_and_scheduler():
    cfg = EngineConfig.defaults("m", max_model_len=1024)
    assert cfg.model.max_model_len == 1024
    assert cfg.scheduler.max_model_len == 1024


def test_defaults_rejects_unknown_override():
    with pytest.raises(ValueError, match="bogus"):
        EngineConfig.defaults("m", bogus=1)


def test_defaults_rejects_model_as_override():
    # "model" binds to the positional parameter, so a duplicate keyword fails at the call.
    with pytest.raises(TypeError):
        EngineConfig.defaults("m", model="other")


@pytest.mark.parametrize(
    ("name", "model"),
    [
        ("qwen2.5-1.5b-instruct.yaml", "Qwen/Qwen2.5-1.5B-Instruct"),
        ("qwen2.5-3b-instruct.yaml", "Qwen/Qwen2.5-3B-Instruct"),
    ],
)
def test_from_yaml_shipped_configs(name: str, model: str):
    cfg = EngineConfig.from_yaml(CONFIG_DIR / name)
    assert cfg.model.model == model
    assert cfg.model.dtype == "float32"
    assert cfg.model.device == "cpu"
    assert cfg.cache.block_size == 16
    assert cfg.cache.enable_prefix_cache is True
    assert cfg.scheduler.preemption_mode == "recompute"
    assert cfg.attention_backend == "auto"


def test_from_yaml_accepts_str_path():
    cfg = EngineConfig.from_yaml(str(CONFIG_DIR / "qwen2.5-1.5b-instruct.yaml"))
    assert cfg.model.model == "Qwen/Qwen2.5-1.5B-Instruct"


def test_from_yaml_defaults_missing_sections(tmp_path: Path):
    path = tmp_path / "minimal.yaml"
    path.write_text("model:\n  model: tiny\n", encoding="utf-8")
    cfg = EngineConfig.from_yaml(path)
    assert cfg.model.model == "tiny"
    assert cfg.cache == CacheConfig()
    assert cfg.scheduler == SchedulerConfig()
    assert cfg.attention_backend == "auto"


def test_from_yaml_requires_model_section(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("cache:\n  block_size: 16\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model"):
        EngineConfig.from_yaml(path)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("float32", torch.float32),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
    ],
)
def test_torch_dtype_mapping(name: str, expected: torch.dtype):
    assert ModelConfig(model="m", dtype=name).torch_dtype() is expected


def test_torch_dtype_rejects_unknown():
    with pytest.raises(ValueError, match="int8"):
        ModelConfig(model="m", dtype="int8").torch_dtype()
