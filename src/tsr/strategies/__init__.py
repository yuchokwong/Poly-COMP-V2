"""Strategy registry. Modules are imported lazily by key."""

from __future__ import annotations

import importlib

from tsr.strategies.base import Strategy, StrategyContext

STRATEGY_MODULES = {
    "clip": "tsr.strategies.clip_only",
    "clip_nlp": "tsr.strategies.clip_nlp",
    "vlm": "tsr.strategies.vlm",
}

DISPLAY_ORDER = ("clip", "clip_nlp", "vlm")


def available_keys() -> list[str]:
    return list(DISPLAY_ORDER)


def get_strategy(key: str, ctx: StrategyContext) -> Strategy:
    module_name = STRATEGY_MODULES.get(key)
    if module_name is None:
        raise KeyError(f"unknown strategy {key!r}")
    module = importlib.import_module(module_name)
    return module.STRATEGY(ctx)


__all__ = [
    "DISPLAY_ORDER",
    "STRATEGY_MODULES",
    "Strategy",
    "StrategyContext",
    "available_keys",
    "get_strategy",
]
