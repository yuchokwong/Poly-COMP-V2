"""Shared fixtures: temporary configs, manifest records and a seeded store."""

from __future__ import annotations

from pathlib import Path

import pytest

from tsr.config import load_config
from tsr.manifest import ManifestRecord
from tsr.store import GalleryStore


@pytest.fixture
def cfg(tmp_path):
    config = load_config()
    config.paths["data_dir"] = str(tmp_path)
    config.paths["cache_dir"] = str(tmp_path / ".cache")
    return config


def make_record(
    index: int = 0,
    split: str = "test",
    img_captions: list[str] | None = None,
    meme_captions: list[str] | None = None,
    metaphors: list[dict] | None = None,
) -> ManifestRecord:
    image_id = f"memes_{index:05d}"
    return ManifestRecord(
        image_id=image_id,
        img_fname=f"{image_id}.png",
        split=split,
        path=f"memecap/{image_id}.jpg",
        width=640,
        height=480,
        sha1=f"sha{index}",
        phash=f"ph{index}",
        title=f"Title {index}",
        source_url=f"https://example.com/{image_id}",
        post_id=f"p{index}",
        img_captions=img_captions or [f"a photo of scene number {index}"],
        meme_captions=meme_captions or [f"a meme meaning number {index}"],
        metaphors=metaphors or [{"metaphor": f"object {index}", "meaning": "meme poster"}],
        category="memes",
    )


@pytest.fixture
def record_factory():
    """Expose ``make_record`` so tests need not import the conftest module."""
    return make_record


@pytest.fixture
def records() -> list[ManifestRecord]:
    return [
        make_record(0),
        make_record(
            1,
            img_captions=["a cat sitting at a dinner table"],
            meme_captions=["pretending to be polite at dinner"],
        ),
        make_record(
            2,
            img_captions=["a dog in a burning room"],
            meme_captions=["everything is fine"],
            metaphors=[{"metaphor": "fire", "meaning": "chaos"}],
        ),
    ]


@pytest.fixture
def store(cfg, records):
    store = GalleryStore(Path(cfg.paths.data_dir) / "gallery.db")
    store.insert_images(records)
    yield store
    store.close()
