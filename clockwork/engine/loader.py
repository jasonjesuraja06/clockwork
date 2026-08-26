"""Weight loading from the Hugging Face cache and tiny random-weight model builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, Qwen2Config
from transformers import Qwen2ForCausalLM as HFQwen2ForCausalLM

from clockwork.config import ModelConfig
from clockwork.engine.model import Qwen2ForCausalLM

_TINY_DEFAULTS = {
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "intermediate_size": 256,
    "vocab_size": 512,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "use_sliding_window": False,
}


def _resolve_checkpoint_dir(model: str) -> Path:
    path = Path(model)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json", "*.txt"],
        )
    )


def _load_safetensors_state(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {checkpoint_dir}")
    state: dict[str, torch.Tensor] = {}
    for file in files:
        state.update(load_file(file, device="cpu"))
    return state


def _load_state_dict(model: Qwen2ForCausalLM, state: dict[str, torch.Tensor]) -> None:
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Checkpoints of tied models omit lm_head.weight; the tied parameter covers it.
    if tied:
        missing = [key for key in missing if key != "lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing} unexpected={unexpected}")


def load_model(cfg: ModelConfig) -> tuple[Qwen2ForCausalLM, Any, Any]:
    """Load a Qwen2 checkpoint into the engine's model; returns (model, hf_config, tokenizer)."""
    hf_config = AutoConfig.from_pretrained(cfg.model, trust_remote_code=cfg.trust_remote_code)
    if hf_config.model_type != "qwen2":
        raise ValueError(f"expected a qwen2 model, got model_type={hf_config.model_type!r}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=cfg.trust_remote_code)
    model = Qwen2ForCausalLM(hf_config)
    model.to(dtype=cfg.torch_dtype())
    checkpoint_dir = _resolve_checkpoint_dir(cfg.model)
    _load_state_dict(model, _load_safetensors_state(checkpoint_dir))
    model.to(device=cfg.device)
    model.eval()
    return model, hf_config, tokenizer


def _tiny_config(**overrides) -> Qwen2Config:
    params = dict(_TINY_DEFAULTS)
    params.update(overrides)
    return Qwen2Config(**params)


def tiny_qwen2_hf(seed: int = 0, **overrides) -> HFQwen2ForCausalLM:
    """Build a small random-weight Hugging Face Qwen2ForCausalLM with a fixed seed."""
    config = _tiny_config(**overrides)
    torch.manual_seed(seed)
    hf_model = HFQwen2ForCausalLM(config)
    hf_model.eval()
    return hf_model


def build_tiny_qwen2(seed: int = 0, **overrides) -> tuple[Qwen2ForCausalLM, Any]:
    """Build the engine's tiny Qwen2 holding the same weights as tiny_qwen2_hf(seed)."""
    hf_model = tiny_qwen2_hf(seed, **overrides)
    model = Qwen2ForCausalLM(hf_model.config)
    _load_state_dict(model, hf_model.state_dict())
    model.eval()
    return model, hf_model.config
