"""Strategy A: pure CLIP text-to-image retrieval."""

from __future__ import annotations

import time

from tsr.artifacts import read_meta
from tsr.strategies.base import Hit, SearchOutput, Strategy


class ClipOnlyStrategy(Strategy):
    key = "clip"
    title = "CLIP only"
    description = (
        "OpenCLIP text encoder against the shared image index. No captions, "
        "annotations or generated descriptions are used."
    )

    def load(self) -> None:
        ok, error = self.ctx.clip.index_valid()
        if not ok:
            self.ready = False
            self.load_error = (
                f"{error}. Build it with `python -m tsr build --strategy clip`."
            )
            return
        # An index without image rows would answer "ready" and then return
        # nothing: the DB is a separate artifact and can be missing or empty.
        if self.store.count() <= 0:
            self.ready = False
            self.load_error = (
                "Gallery database is empty. Run `python -m tsr prepare` and "
                "`python -m tsr build --strategy gallery`."
            )
            return
        self.ctx.clip.index()
        self.ready = True
        self.load_error = None

    def search(self, query: str, k: int, **options) -> SearchOutput:
        self.ensure_loaded()
        if not self.ready:
            raise RuntimeError(self.load_error or "strategy clip is not ready")
        started = time.perf_counter()
        templates = self.ctx.clip_templates()
        entries = self.ctx.clip.search_text(query, max(k * 2, k), templates)
        entries = [(pk, score) for pk, score in entries if score > 0.0]
        hits = [
            Hit(image_pk=pk, score=float(score), components={"clip": float(score)})
            for pk, score in entries[:k]
        ]
        return SearchOutput(
            query=query,
            strategy=self.key,
            hits=hits,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
            expanded_queries=[query],
        )

    def stats(self) -> dict:
        self.ensure_loaded()
        meta = read_meta(self.index_dir)
        return {
            "strategy": self.key,
            "title": self.title,
            "description": self.description,
            "ready": self.ready,
            "error": self.load_error,
            "num_images": self.store.count(),
            "dim": meta.get("dim"),
            "model": meta.get("model"),
            "built_at": meta.get("built_at"),
            "meta": meta,
        }


STRATEGY = ClipOnlyStrategy
