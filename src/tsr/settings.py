"""Runtime-editable settings, surfaced through the web UI.

Only parameters that are safe to change while the server runs are editable:
fusion weights, fetch multipliers, reranker switches and UI defaults. Build-time
choices (models, index type, dataset) are read-only. Edits are persisted to
``data/settings.json`` and re-applied at startup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tsr.config import AttrDict
from tsr.logging_utils import get_logger

logger = get_logger(__name__)

STRATEGY = "Strategy B — CLIP + NLP"
VLM = "Strategy C — VLM"
BASELINE = "Strategy A — CLIP"
INTERFACE = "Interface"
MODEL = "Models (read-only)"
INDEX = "Index (read-only)"

FIELDS: list[dict[str, Any]] = [
    {
        "key": "app.default_top_k",
        "type": "int",
        "label": "Default results per page",
        "group": INTERFACE,
        "editable": True,
        "min": 1,
        "max": 100,
    },
    {
        "key": "strategies.clip.prompt_ensemble",
        "type": "bool",
        "label": "Prompt ensemble",
        "group": BASELINE,
        "editable": True,
        "help": "Average the CLIP text embedding over the prompt templates.",
    },
    {
        "key": "strategies.clip_nlp.weights.clip",
        "type": "float",
        "label": "CLIP weight",
        "group": STRATEGY,
        "editable": True,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    {
        "key": "strategies.clip_nlp.weights.doc",
        "type": "float",
        "label": "Dense annotation weight",
        "group": STRATEGY,
        "editable": True,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    {
        "key": "strategies.clip_nlp.weights.bm25",
        "type": "float",
        "label": "BM25 weight",
        "group": STRATEGY,
        "editable": True,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "help": "Weights are renormalised after disabled channels are dropped.",
    },
    {
        "key": "strategies.clip_nlp.fetch_multiplier",
        "type": "int",
        "label": "Over-fetch multiplier",
        "group": STRATEGY,
        "editable": True,
        "min": 1,
        "max": 20,
    },
    {
        "key": "strategies.clip_nlp.expand",
        "type": "bool",
        "label": "Lexicon query expansion",
        "group": STRATEGY,
        "editable": True,
        "help": "Expand known meme phrases for the BM25 signal only.",
    },
    {
        "key": "strategies.clip_nlp.expansion_weight",
        "type": "float",
        "label": "Expansion weight",
        "group": STRATEGY,
        "editable": True,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    {
        "key": "strategies.clip_nlp.rerank.enabled",
        "type": "bool",
        "label": "Cross-encoder rerank",
        "group": STRATEGY,
        "editable": True,
    },
    {
        "key": "strategies.clip_nlp.rerank.top_n",
        "type": "int",
        "label": "Rerank candidates",
        "group": STRATEGY,
        "editable": True,
        "min": 1,
        "max": 200,
    },
    {
        "key": "strategies.clip_nlp.rerank.alpha",
        "type": "float",
        "label": "Rerank weight",
        "group": STRATEGY,
        "editable": True,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    {
        "key": "strategies.vlm.clip_weight",
        "type": "float",
        "label": "CLIP fusion weight",
        "group": VLM,
        "editable": True,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "help": "0 = pure VLM text retrieval; 0.2 = VLM + CLIP visual signal.",
    },
    {
        "key": "strategies.vlm.fetch_multiplier",
        "type": "int",
        "label": "Over-fetch multiplier",
        "group": VLM,
        "editable": True,
        "min": 1,
        "max": 20,
    },
    {
        "key": "strategies.vlm.rerank.enabled",
        "type": "bool",
        "label": "Cross-encoder rerank",
        "group": VLM,
        "editable": True,
    },
    {
        "key": "models.clip.name",
        "type": "str",
        "label": "CLIP backbone",
        "group": MODEL,
        "editable": False,
    },
    {
        "key": "models.clip.pretrained",
        "type": "str",
        "label": "CLIP pretrained",
        "group": MODEL,
        "editable": False,
    },
    {
        "key": "models.text.name",
        "type": "str",
        "label": "Text encoder",
        "group": MODEL,
        "editable": False,
    },
    {
        "key": "models.vlm.model",
        "type": "str",
        "label": "VLM",
        "group": MODEL,
        "editable": False,
    },
    {
        "key": "models.vlm.backend",
        "type": "str",
        "label": "VLM backend",
        "group": MODEL,
        "editable": False,
    },
    {
        "key": "index.type",
        "type": "str",
        "label": "Index type",
        "group": INDEX,
        "editable": False,
    },
    {
        "key": "dataset.name",
        "type": "str",
        "label": "Dataset",
        "group": INDEX,
        "editable": False,
    },
]

_SPEC_BY_KEY = {field["key"]: field for field in FIELDS}


def _get(cfg: AttrDict, key: str) -> Any:
    node: Any = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _set(cfg: AttrDict, key: str, value: Any) -> None:
    parts = key.split(".")
    node: Any = cfg
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = AttrDict()
        node = node[part]
    node[parts[-1]] = value


def coerce(field: dict[str, Any], value: Any) -> Any:
    kind = field["type"]
    if kind == "bool":
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "enum":
        value = str(value)
        if value not in field.get("choices", []):
            raise ValueError(f"{field['key']}: must be one of {field['choices']}")
        return value
    if kind == "list":
        if isinstance(value, str):
            return [line.strip() for line in value.splitlines() if line.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        raise ValueError(f"{field['key']}: expected a list")
    return str(value)


def validate(field: dict[str, Any], value: Any) -> Any:
    value = coerce(field, value)
    lo = field.get("min")
    hi = field.get("max")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ValueError(f"{field['key']}: must be between {lo} and {hi}")
    return value


def get_settings(cfg: AttrDict) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for field in FIELDS:
        item = dict(field)
        item["value"] = _get(cfg, field["key"])
        out.append(item)
    return out


def apply_settings(cfg: AttrDict, updates: dict[str, Any]) -> dict[str, Any]:
    """Validate every edit first, then apply them in-place atomically."""
    resolved: dict[str, Any] = {}
    for key, raw in updates.items():
        field = _SPEC_BY_KEY.get(key)
        if field is None or not field.get("editable", False):
            raise ValueError(f"{key}: not an editable setting")
        resolved[key] = validate(field, raw)
    for key, value in resolved.items():
        _set(cfg, key, value)
    return resolved


def settings_path(cfg: AttrDict) -> Path:
    return Path(cfg.paths.data_dir) / "settings.json"


def save_settings(cfg: AttrDict) -> None:
    data = {field["key"]: _get(cfg, field["key"]) for field in FIELDS if field.get("editable")}
    path = settings_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_persisted(cfg: AttrDict) -> None:
    path = settings_path(cfg)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Ignoring bad settings file %s: %s", path, exc)
        return
    for key, value in data.items():
        field = _SPEC_BY_KEY.get(key)
        if field is None or not field.get("editable"):
            continue
        try:
            _set(cfg, key, validate(field, value))
        except (ValueError, TypeError) as exc:
            logger.warning("Ignoring invalid persisted setting %s=%r: %s", key, value, exc)
