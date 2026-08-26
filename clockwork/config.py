"""Engine, model, cache, and scheduler configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    import torch

_DTYPES = ("float32", "float16", "bfloat16")


@dataclass
class ModelConfig:
    model: str
    dtype: str = "float32"
    device: str = "cpu"
    max_model_len: int = 4096
    seed: int = 0
    trust_remote_code: bool = False

    def torch_dtype(self) -> torch.dtype:
        import torch

        if self.dtype not in _DTYPES:
            raise ValueError(f"unknown dtype {self.dtype!r}, expected one of {_DTYPES}")
        return getattr(torch, self.dtype)


@dataclass
class CacheConfig:
    block_size: int = 16
    num_blocks: int = 512
    enable_prefix_cache: bool = True
    watermark: float = 0.01


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 2048
    max_model_len: int = 4096
    preemption_mode: str = "recompute"


@dataclass
class EngineConfig:
    model: ModelConfig
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    attention_backend: str = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path) -> EngineConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        model = data.get("model")
        if not isinstance(model, dict) or "model" not in model:
            raise ValueError(f"{path}: expected a top-level 'model' section naming the model")
        return cls(
            model=ModelConfig(**model),
            cache=CacheConfig(**(data.get("cache") or {})),
            scheduler=SchedulerConfig(**(data.get("scheduler") or {})),
            attention_backend=data.get("attention_backend", "auto"),
        )

    @classmethod
    def defaults(cls, model: str, **overrides) -> EngineConfig:
        cfg = cls(model=ModelConfig(model=model))
        for key, value in overrides.items():
            if key == "attention_backend":
                cfg.attention_backend = value
                continue
            # max_model_len exists on both ModelConfig and SchedulerConfig; set every match.
            targets = [
                sub
                for sub in (cfg.model, cfg.cache, cfg.scheduler)
                if key in {f.name for f in fields(sub)}
            ]
            if not targets:
                raise ValueError(f"unknown config override {key!r}")
            for sub in targets:
                setattr(sub, key, value)
        return cfg
