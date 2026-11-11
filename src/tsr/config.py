"""Configuration loading and filesystem path helpers.

Layering (lowest to highest precedence):
    1. configs/default.yaml
    2. optional file from $TSR_CONFIG
    3. environment variables prefixed with ``TSR_`` (nested keys use ``__``)

Example: ``TSR_APP__PORT=9000`` sets ``cfg.app.port = 9000``.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class AttrDict(dict):
    """dict with attribute access, recursively applied."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _coerce(value: str) -> Any:
    v = value.strip()
    low = v.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none", ""}:
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if (v.startswith("{") and v.endswith("}")) or (v.startswith("[") and v.endswith("]")):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return value
    return value


def _env_overrides(prefix: str = "TSR_") -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("__")
        node = overrides
        ok = True
        for part in parts[:-1]:
            nxt = node.get(part)
            if nxt is None:
                nxt = {}
                node[part] = nxt
            if not isinstance(nxt, dict):
                ok = False
                break
            node = nxt
        if ok:
            node[parts[-1]] = _coerce(value)
    return overrides


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _to_attr(obj: Any) -> Any:
    if isinstance(obj, dict):
        return AttrDict({k: _to_attr(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attr(v) for v in obj]
    return obj


def load_config(path: str | Path | None = None) -> AttrDict:
    """Load and validate configuration into an attribute-accessible dict."""
    raw: dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.exists():
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}

    if path is None:
        path = os.environ.get("TSR_CONFIG")
    if path is not None:
        override_path = Path(path)
        if override_path.exists() and override_path.resolve() != DEFAULT_CONFIG_PATH.resolve():
            raw = _deep_merge(
                raw, yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
            )

    raw = _deep_merge(raw, _env_overrides())
    cfg = _to_attr(raw)
    cfg["root"] = str(PROJECT_ROOT)

    data_dir = Path(cfg.paths.data_dir)
    if not data_dir.is_absolute():
        data_dir = (PROJECT_ROOT / data_dir).resolve()
    cfg.paths["data_dir"] = str(data_dir)

    cache_dir = Path(cfg.paths.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (data_dir / cache_dir).resolve()
    cfg.paths["cache_dir"] = str(cache_dir)

    return cfg


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #
def data_dir(cfg: AttrDict) -> Path:
    return Path(cfg.paths.data_dir)


def images_dir(cfg: AttrDict) -> Path:
    return data_dir(cfg) / cfg.paths.images_subdir


def cache_dir(cfg: AttrDict) -> Path:
    return Path(cfg.paths.cache_dir)


def web_dir(cfg: AttrDict) -> Path:
    return Path(cfg.root) / cfg.paths.web_dir


def db_path(cfg: AttrDict) -> Path:
    return data_dir(cfg) / cfg.paths.db_name


def manifest_path(cfg: AttrDict) -> Path:
    return data_dir(cfg) / cfg.paths.manifest_name


def raw_dir(cfg: AttrDict) -> Path:
    return data_dir(cfg) / "raw"


def raw_images_dir(cfg: AttrDict) -> Path:
    return raw_dir(cfg) / "memecap" / "images"


def raw_captions_dir(cfg: AttrDict) -> Path:
    return raw_dir(cfg) / "memecap" / "caption"


def archive_path(cfg: AttrDict) -> Path:
    return raw_dir(cfg) / cfg.dataset.archive_name


def index_root(cfg: AttrDict) -> Path:
    return data_dir(cfg) / cfg.paths.index_subdir


def strategy_index_dir(cfg: AttrDict, key: str) -> Path:
    return index_root(cfg) / key


def vlm_cache_path(cfg: AttrDict) -> Path:
    return data_dir(cfg) / cfg.models.vlm.cache_file


_VLM_BACKEND_ALIASES = {"qwen": "qwen2_5_vl", "qwen2.5-vl": "qwen2_5_vl"}
_VLM_BACKEND_DISABLED = {"none", "disabled", "off", ""}


def canonical_vlm_backend(value: object) -> str | None:
    """Canonical backend name, or None when the VLM is disabled.

    ``encoders.vlm`` accepts aliases but persists ``QwenVLBackend.name``; every
    reader of ``vlm_docs`` must compare against that same canonical string.
    """
    name = str("qwen2_5_vl" if value is None else value).strip().lower()
    if name in _VLM_BACKEND_DISABLED:
        return None
    return _VLM_BACKEND_ALIASES.get(name, name)


def vlm_provenance(cfg: AttrDict) -> dict[str, str | bool | None]:
    """Filters that pin VLM rows to the generator the config currently names.

    Stale descriptions from an earlier model or prompt must never be served or
    counted as coverage, so every read of ``vlm_docs`` filters with this.
    ``enabled`` is False when ``models.vlm.backend`` disables captioning; the
    filter values are then unset and callers must refuse to serve VLM output.
    """
    vlm = cfg.models.get("vlm") or {}
    backend = canonical_vlm_backend(vlm.get("backend", "qwen2_5_vl"))
    model = vlm.get("model")
    return {
        "backend": backend,
        "model": str(model) if model else None,
        "prompt_version": str(vlm.get("prompt_version", "v1")),
        "enabled": backend is not None,
    }


def vlm_doc_filters(cfg: AttrDict) -> dict[str, str | None]:
    """Keyword filters for ``store`` VLM readers (no ``enabled`` flag)."""
    provenance = vlm_provenance(cfg)
    return {key: provenance[key] for key in ("backend", "model", "prompt_version")}


def eval_dir(cfg: AttrDict) -> Path:
    return data_dir(cfg) / cfg.paths.eval_dir


def reports_dir(cfg: AttrDict) -> Path:
    return data_dir(cfg) / cfg.paths.reports_dir


def tuning_dir(cfg: AttrDict) -> Path:
    """Separate tree for parameter-sweep artifacts; never the main reports dir."""
    return data_dir(cfg) / "tuning"


def settings_path(cfg: AttrDict) -> Path:
    return data_dir(cfg) / "settings.json"
