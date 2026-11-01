"""Artifact metadata: hashes, validation and atomic ``meta.json`` writes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_hash(manifest_file: str | Path) -> str:
    path = Path(manifest_file)
    return file_sha256(path) if path.exists() else ""


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:  # pragma: no cover - not a git checkout
        return ""


def meta_path(directory: str | Path) -> Path:
    return Path(directory) / "meta.json"


def write_meta(directory: str | Path, meta: dict) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        **meta,
    }
    dest = meta_path(directory)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, dest)
    return dest


def read_meta(directory: str | Path) -> dict:
    path = meta_path(directory)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def validate_meta(
    directory: str | Path,
    *,
    kind: str | None = None,
    model: str | None = None,
    dim: int | None = None,
    count: int | None = None,
    dataset_hash_expected: str | None = None,
) -> tuple[bool, str | None]:
    meta = read_meta(directory)
    if not meta:
        return False, f"missing meta.json in {directory}"
    if meta.get("schema_version") != SCHEMA_VERSION:
        return False, f"artifact schema {meta.get('schema_version')} != {SCHEMA_VERSION}"
    for name, expected in (
        ("kind", kind),
        ("model", model),
        ("dim", dim),
        ("count", count),
        ("dataset_hash", dataset_hash_expected),
    ):
        if expected is None:
            continue
        actual = meta.get(name)
        if actual != expected:
            return False, f"artifact {name}={actual!r} does not match {expected!r}"
    return True, None
