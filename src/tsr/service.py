"""Strategy manager: one shared gallery, three lazily loaded retrieval pipelines."""

from __future__ import annotations

import random
import threading
from pathlib import Path

from tsr.config import (
    AttrDict,
    cache_dir,
    db_path,
    images_dir,
    index_root,
    manifest_path,
    vlm_doc_filters,
    vlm_provenance,
)
from tsr.encoders.clip import resolve_device
from tsr.logging_utils import get_logger
from tsr.store import GalleryStore, ImageRecord
from tsr.strategies import DISPLAY_ORDER, get_strategy
from tsr.strategies.base import Hit, SearchOutput, StrategyContext

logger = get_logger(__name__)


class UnknownStrategy(KeyError):
    pass


class StrategyUnavailable(RuntimeError):
    pass


class StrategyManager:
    """Owns the SQLite store and the strategy instances for the process."""

    def __init__(
        self,
        cfg: AttrDict,
        store: GalleryStore | None = None,
        skip: set[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self._lock = threading.RLock()
        self._store = store
        self._store_error: str | None = None
        self._skip = skip or set()
        self.strategies = {}
        self._build()

    # ------------------------------------------------------------------ #
    def _open_store(self) -> GalleryStore | None:
        if self._store is not None:
            return self._store
        try:
            self._store = GalleryStore(db_path(self.cfg))
            return self._store
        except Exception as exc:  # noqa: BLE001
            self._store_error = str(exc)
            logger.warning("Gallery database unavailable: %s", exc)
            return None

    @property
    def store(self) -> GalleryStore:
        store = self._open_store()
        if store is None:
            raise StrategyUnavailable(
                self._store_error or "Gallery database is unavailable"
            )
        return store

    @property
    def store_error(self) -> str | None:
        return self._store_error

    def _build(self) -> None:
        store = self._open_store()
        enabled = [str(key) for key in (self.cfg.strategies.get("enabled") or DISPLAY_ORDER)]
        if store is None:
            self._store_error = self._store_error or "Gallery database is unavailable"
            return
        ctx = StrategyContext(self.cfg, store)
        for key in DISPLAY_ORDER:
            if key not in enabled or key in self._skip:
                continue
            try:
                self.strategies[key] = get_strategy(key, ctx)
            except Exception as exc:  # noqa: BLE001 - one bad plugin must not kill all
                logger.exception("Could not construct strategy %s: %s", key, exc)

    # ------------------------------------------------------------------ #
    def keys(self) -> list[str]:
        return list(self.strategies)

    def get(self, key: str):
        strategy = self.strategies.get(key)
        if strategy is None:
            raise UnknownStrategy(key)
        return strategy

    def status(self) -> list[dict]:
        out = []
        for key, strategy in self.strategies.items():
            try:
                stats = strategy.stats()
            except Exception as exc:  # noqa: BLE001
                stats = {
                    "strategy": key,
                    "title": strategy.title,
                    "description": strategy.description,
                    "ready": False,
                    "error": str(exc),
                }
            stats.setdefault("key", key)
            out.append(stats)
        return out

    def health(self) -> dict:
        store = self._open_store()
        statuses = self.status() if store is not None else []
        return {
            "status": "ok" if any(s.get("ready") for s in statuses) else "degraded",
            "ready": any(s.get("ready") for s in statuses),
            "store_error": self._store_error,
            "num_images": store.count() if store is not None else 0,
            "strategies": [
                {
                    "key": item.get("key") or item.get("strategy"),
                    "title": item.get("title"),
                    "ready": bool(item.get("ready")),
                    "error": item.get("error"),
                }
                for item in statuses
            ],
        }

    def stats(self) -> dict:
        store = self._open_store()
        return {
            "dataset": {
                "name": str(self.cfg.dataset.name),
                "num_images": store.count() if store is not None else 0,
                "manifest": str(manifest_path(self.cfg)),
                "manifest_exists": manifest_path(self.cfg).exists(),
                "vlm_docs": (
                    store.vlm_count(**vlm_doc_filters(self.cfg)) if store is not None else 0
                ),
            },
            "models": {
                "clip": f"{self.cfg.models.clip.name}/{self.cfg.models.clip.pretrained}",
                "text": str(self.cfg.models.text.name),
                "vlm": str(self.cfg.models.vlm.get("model")),
                "vlm_backend": str(self.cfg.models.vlm.get("backend")),
                "device": resolve_device(self.cfg.models.clip.get("device", "auto")),
            },
            "index": {
                "type": str(self.cfg.index.get("type", "flat")),
                "root": str(index_root(self.cfg)),
            },
            "strategies": self.status(),
            "store_error": self._store_error,
        }

    # ------------------------------------------------------------------ #
    def search(self, key: str, query: str, k: int) -> SearchOutput:
        strategy = self.get(key)
        strategy.ensure_loaded()
        if not strategy.ready:
            raise StrategyUnavailable(strategy.load_error or f"strategy {key} is not ready")
        if not query.strip():
            raise ValueError("query must not be empty")
        with self._lock:
            return strategy.search(query.strip(), k)

    def compare(self, query: str, k: int) -> dict[str, dict]:
        results: dict[str, dict] = {}
        for key, strategy in self.strategies.items():
            strategy.ensure_loaded()
            if not strategy.ready:
                results[key] = {
                    "strategy": key,
                    "ready": False,
                    "error": strategy.load_error,
                    "results": [],
                    "count": 0,
                    "latency_ms": 0.0,
                }
                continue
            try:
                with self._lock:
                    output = strategy.search(query.strip(), k)
                results[key] = self._payload(output, k)
            except Exception as exc:  # noqa: BLE001 - per-column error
                logger.warning("Strategy %s failed for %r: %s", key, query, exc)
                results[key] = {
                    "strategy": key,
                    "ready": False,
                    "error": str(exc),
                    "results": [],
                    "count": 0,
                    "latency_ms": 0.0,
                }
        return results

    def similar(self, pk: int, key: str, k: int) -> SearchOutput:
        strategy = self.get(key)
        strategy.ensure_loaded()
        if not strategy.ready:
            raise StrategyUnavailable(strategy.load_error or f"strategy {key} is not ready")
        if self.store.get(pk) is None:
            raise LookupError(pk)
        with self._lock:
            return strategy.similar(pk, k)

    # ------------------------------------------------------------------ #
    def _payload(self, output: SearchOutput, k: int) -> dict:
        records = self.store.get_many([hit.image_pk for hit in output.hits])
        self.store.enrich(records)
        results = []
        for rank, hit in enumerate(output.hits[:k], start=1):
            record = records.get(hit.image_pk)
            if record is None:
                continue
            results.append(self._result(rank, hit, record))
        return {
            "query": output.query,
            "strategy": output.strategy,
            "ready": True,
            "latency_ms": output.latency_ms,
            "expanded_queries": output.expanded_queries,
            "debug": output.debug,
            "count": len(results),
            "results": results,
        }

    def _vlm_doc(self, pk: int) -> dict | None:
        if not vlm_provenance(self.cfg)["enabled"]:
            return None
        return self.store.get_vlm_doc(pk, **vlm_doc_filters(self.cfg))

    def _result(self, rank: int, hit: Hit, record: ImageRecord) -> dict:
        vlm = self._vlm_doc(hit.image_pk)
        return {
            "rank": rank,
            "id": hit.image_pk,
            "image_id": record.image_id,
            "score": round(float(hit.score), 6),
            "score_components": {
                name: round(float(value), 6) for name, value in hit.components.items()
            },
            "url": f"/api/image/{hit.image_pk}",
            "thumbnail": f"/api/thumbnail/{hit.image_pk}",
            "title": record.title,
            "img_captions": record.img_captions,
            "meme_captions": record.meme_captions,
            "metaphors": record.metaphors,
            "category": record.category,
            "split": record.split,
            "width": record.width,
            "height": record.height,
            "source_url": record.source_url,
            "vlm": (
                {
                    "scene": vlm.get("scene"),
                    "ocr_text": vlm.get("ocr_text"),
                    "meaning": vlm.get("meaning"),
                    "model": vlm.get("model"),
                }
                if vlm
                else None
            ),
        }

    def search_payload(self, key: str, query: str, k: int) -> dict:
        with self._lock:
            output = self.search(key, query, k)
        return self._payload(output, k)

    def similar_payload(self, pk: int, key: str, k: int) -> dict:
        output = self.similar(pk, key, k)
        payload = self._payload(output, k)
        payload["query"] = f"#{pk}"
        return payload

    # ------------------------------------------------------------------ #
    def get_record(self, pk: int) -> ImageRecord | None:
        try:
            rec = self.store.get(pk)
        except StrategyUnavailable:
            return None
        if rec is None:
            return None
        self.store.enrich({pk: rec})
        return rec

    def detail(self, pk: int) -> dict | None:
        record = self.get_record(pk)
        if record is None:
            return None
        vlm = self._vlm_doc(pk)
        return {
            "id": record.id,
            "image_id": record.image_id,
            "title": record.title,
            "split": record.split,
            "category": record.category,
            "width": record.width,
            "height": record.height,
            "source_url": record.source_url,
            "post_id": record.post_id,
            "url": f"/api/image/{record.id}",
            "thumbnail": f"/api/thumbnail/{record.id}",
            "img_captions": record.img_captions,
            "meme_captions": record.meme_captions,
            "metaphors": record.metaphors,
            "vlm": vlm,
        }

    def image_file(self, pk: int) -> Path | None:
        record = self.get_record(pk)
        if record is None:
            return None
        root = images_dir(self.cfg).resolve()
        path = (root / record.path).resolve()
        if root not in path.parents and path != root:
            return None
        return path if path.exists() else None

    def thumbnail(self, pk: int, size: int) -> Path | None:
        src = self.image_file(pk)
        if src is None:
            return None
        dest = cache_dir(self.cfg) / "thumbs" / str(size) / f"{pk}.jpg"
        if dest.exists():
            return dest
        from PIL import Image

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(src) as img:
                img = img.convert("RGB")
                img.thumbnail((size, size), Image.LANCZOS)
                img.save(dest, "JPEG", quality=82, optimize=True)
        except Exception as exc:  # noqa: BLE001 - corrupted image is a 404
            logger.warning("Thumbnail failed for %s: %s", pk, exc)
            return None
        return dest

    def random_sample(self, n: int) -> list[dict]:
        try:
            records = self.store.iter_all()
        except StrategyUnavailable:
            return []
        if not records:
            return []
        picked = random.sample(records, min(n, len(records)))
        self.store.enrich({rec.id: rec for rec in picked})
        return [
            {
                "id": rec.id,
                "image_id": rec.image_id,
                "url": f"/api/image/{rec.id}",
                "thumbnail": f"/api/thumbnail/{rec.id}",
                "title": rec.title,
                "img_captions": rec.img_captions[:2],
            }
            for rec in picked
        ]

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None


__all__ = [
    "StrategyManager",
    "StrategyUnavailable",
    "UnknownStrategy",
]
