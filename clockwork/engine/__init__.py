"""Engine core: sequence state, attention metadata, model, and execution loop."""

from clockwork.engine.sequence import RequestOutput, SamplingParams, Sequence, SequenceStatus

__all__ = [
    "AsyncLLMEngine",
    "LLMEngine",
    "ModelRunner",
    "RequestOutput",
    "SamplingParams",
    "Sequence",
    "SequenceStatus",
]

_LAZY = {
    "AsyncLLMEngine": "clockwork.engine.async_engine",
    "LLMEngine": "clockwork.engine.llm_engine",
    "ModelRunner": "clockwork.engine.model_runner",
}


def __getattr__(name: str):
    # The engine classes pull in torch; loading them lazily keeps this package
    # torch-free at import time for sequence-only users.
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)
