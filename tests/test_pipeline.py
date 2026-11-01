"""Gallery build pipeline: id layout, manifest guard and artifact metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tsr import pipeline
from tsr.manifest import write_manifest
from tsr.store import GalleryStore


class FakeClipEncoder:
    dim = 6

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def encode_images(self, paths):
        return np.ones((len(paths), self.dim), dtype=np.float32)


@pytest.fixture
def built_env(cfg, records, monkeypatch, tmp_path: Path):
    pytest.importorskip("faiss")
    root = Path(cfg.paths.data_dir)
    write_manifest(records, root / "manifest.jsonl")
    for record in records:
        image = root / "images" / record.path
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"not-a-real-jpeg")
    monkeypatch.setattr(pipeline, "ClipEncoder", FakeClipEncoder)
    return cfg


def test_build_gallery_writes_index_and_meta(built_env) -> None:
    cfg = built_env
    size = pipeline.build_gallery(cfg)

    assert size == 3
    out_dir = Path(cfg.paths.data_dir) / "index" / "clip"
    assert (out_dir / "image.faiss").exists()
    assert (out_dir / "image_ids.npy").exists()
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["kind"] == "clip_image"
    assert meta["count"] == 3
    assert meta["dim"] == FakeClipEncoder.dim

    store = GalleryStore(Path(cfg.paths.data_dir) / "gallery.db")
    try:
        assert [rec.id for rec in store.iter_all()] == [1, 2, 3]
    finally:
        store.close()


def test_build_gallery_rejects_annotations_edited_in_place(built_env, record_factory) -> None:
    """Same image ids, corrected caption: an id-only check would keep stale text."""
    cfg = built_env
    pipeline.build_gallery(cfg)

    rewritten = [record_factory(i) for i in range(3)]
    rewritten[1].img_captions = ["a corrected caption for the same image"]
    write_manifest(rewritten, Path(cfg.paths.data_dir) / "manifest.jsonl")

    with pytest.raises(RuntimeError, match="annotations differ"):
        pipeline.build_gallery(cfg)


def test_build_gallery_rejects_stale_database(built_env, record_factory) -> None:
    cfg = built_env
    pipeline.build_gallery(cfg)

    # A manifest that gained a record after the DB was populated must fail loudly
    # instead of silently desynchronising FAISS row ids from SQLite pks.
    stale = [record_factory(0), record_factory(1)]
    write_manifest(stale, Path(cfg.paths.data_dir) / "manifest.jsonl")
    with pytest.raises(RuntimeError, match="does not match the manifest"):
        pipeline.build_gallery(cfg)


def test_build_gallery_clears_thumbnails_when_it_assigns_new_pks(built_env) -> None:
    """Thumbnails are keyed by pk; a fresh insert can reuse pks for other images."""
    cfg = built_env
    thumbs = Path(cfg.paths.cache_dir) / "thumbs" / "480"
    thumbs.mkdir(parents=True)
    (thumbs / "1.jpg").write_bytes(b"stale thumbnail")

    pipeline.build_gallery(cfg)

    assert not thumbs.exists()
