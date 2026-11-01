"""Manifest records: one JSON object per prepared gallery image."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_FIELDS = (
    "image_id",
    "img_fname",
    "split",
    "path",
    "width",
    "height",
    "sha1",
    "phash",
    "title",
    "source_url",
    "post_id",
    "img_captions",
    "meme_captions",
    "metaphors",
    "category",
)

NOT_RELATED = "not related to the meme context"


@dataclass
class ManifestRecord:
    image_id: str
    img_fname: str
    split: str
    path: str
    width: int | None = None
    height: int | None = None
    sha1: str | None = None
    phash: str | None = None
    title: str | None = None
    source_url: str | None = None
    post_id: str | None = None
    img_captions: list[str] = field(default_factory=list)
    meme_captions: list[str] = field(default_factory=list)
    metaphors: list[dict] = field(default_factory=list)
    category: str = "memes"

    @classmethod
    def from_dict(cls, data: dict) -> ManifestRecord:
        return cls(
            image_id=str(data["image_id"]),
            img_fname=str(data["img_fname"]),
            split=str(data.get("split", "trainval")),
            path=str(data["path"]),
            width=_as_int(data.get("width")),
            height=_as_int(data.get("height")),
            sha1=data.get("sha1"),
            phash=data.get("phash"),
            title=_clean_text(data.get("title")),
            source_url=data.get("source_url"),
            post_id=data.get("post_id"),
            img_captions=list(data.get("img_captions") or []),
            meme_captions=list(data.get("meme_captions") or []),
            metaphors=list(data.get("metaphors") or []),
            category=str(data.get("category") or "memes"),
        )

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in MANIFEST_FIELDS}

    def meaningful_metaphors(self) -> list[dict]:
        """Metaphors that carry non-literal meaning (drop 'not related ...')."""
        out = []
        for item in self.metaphors:
            if not isinstance(item, dict):
                continue
            metaphor = _clean_text(item.get("metaphor"))
            meaning = _clean_text(item.get("meaning"))
            if not metaphor or not meaning:
                continue
            if meaning.strip().lower() == NOT_RELATED:
                continue
            out.append({"metaphor": metaphor, "meaning": meaning})
        return out


def _as_int(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _clean_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def write_manifest(records: list[ManifestRecord], path: str | Path) -> Path:
    """Write atomically: a crash mid-write must not truncate the manifest.

    Every index records the manifest hash, so a half-written file would either
    hash-mismatch (rebuild) or, worse, parse into a shorter gallery.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    path = Path(path)
    if not path.exists():
        return []
    records: list[ManifestRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(ManifestRecord.from_dict(json.loads(line)))
    return records


def manifest_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(records) -> str:
    """Hash the fields that strategy documents and image vectors are built from.

    Image ids alone are not enough: correcting an annotation in place leaves
    every id unchanged, so an id-only check silently keeps stale documents.
    ``sha1``/``path``/size are included too, because the CLIP vectors come from
    the file and the VLM cache is keyed on ``(pk, sha1)``: re-issued image bytes
    under the same names must invalidate both. Both ``ManifestRecord`` and
    ``store.ImageRecord`` provide these fields.
    """
    digest = hashlib.sha256()
    for record in records:
        payload = {
            "image_id": record.image_id,
            "split": record.split,
            "path": record.path,
            "sha1": record.sha1,
            "width": record.width,
            "height": record.height,
            "title": record.title,
            "img_captions": list(record.img_captions),
            "meme_captions": list(record.meme_captions),
            "metaphors": [
                {
                    "metaphor": str(item.get("metaphor", "")),
                    "meaning": str(item.get("meaning", "")),
                }
                for item in record.metaphors
                if isinstance(item, dict)
            ],
        }
        digest.update(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()
