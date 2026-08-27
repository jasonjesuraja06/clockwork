"""Launch the OpenAI-compatible server from a config file plus key=value overrides."""

from __future__ import annotations

import argparse
from dataclasses import fields

import yaml

from clockwork.config import EngineConfig


def _apply_overrides(cfg: EngineConfig, pairs: list[str]) -> None:
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"override {pair!r} is not KEY=VALUE")
        key, raw = pair.split("=", 1)
        value = yaml.safe_load(raw)
        if key == "attention_backend":
            cfg.attention_backend = value
            continue
        targets = [
            sub
            for sub in (cfg.model, cfg.cache, cfg.scheduler)
            if key in {f.name for f in fields(sub)}
        ]
        if not targets:
            raise SystemExit(f"unknown config key {key!r}")
        for sub in targets:
            setattr(sub, key, value)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, build the engine config, and serve with uvicorn."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen2.5-1.5b-instruct.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any model, cache, or scheduler field, e.g. --set num_blocks=1024",
    )
    args = parser.parse_args(argv)
    cfg = EngineConfig.from_yaml(args.config)
    _apply_overrides(cfg, args.overrides)

    import uvicorn

    from clockwork.server.app import build_app

    uvicorn.run(build_app(cfg), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
