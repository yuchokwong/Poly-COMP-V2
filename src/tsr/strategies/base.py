"""Strategy protocol and the shared CLIP backbone."""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tsr.artifacts import read_meta, validate_meta
from tsr.config import AttrDict, manifest_path, strategy_index_dir
from tsr.encoders.clip import ClipEncoder
from tsr.encoders.text import TextEncoder
from tsr.index import VectorIndex
from tsr.logging_utils import get_logger
from tsr.nlp.rerank import CrossEncoderReranker

logger = get_logger(__name__)


@dataclass
class Hit:
    image_pk: int
    score: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class SearchOutput:
    query: str
    strategy: str
    hits: list[Hit]
    latency_ms: float
    expanded_queries: list[str] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


class ClipBackbone:
    """Shared CLIP image index + text/image encoder.

    Strategies A and B both use the same image vectors; the backbone is created
    once per manager and only loads the model on the first query.
    """

    def __init__(self, cfg: AttrDict) -> None:
        self.cfg = cfg
        self.dir = strategy_index_dir(cfg, "clip")
        self._index: VectorIndex | None = None
        self._encoder: ClipEncoder | None = None

    @property
    def index_path(self) -> Path:
        return self.dir / "image.faiss"

    @property
    def ids_path(self) -> Path:
        return self.dir / "image_ids.npy"

    def index(self) -> VectorIndex:
        if self._index is None:
            self._index = VectorIndex.load(
                self.index_path,
                self.ids_path,
                index_type=str(self.cfg.index.get("type", "flat")),
                nprobe=int(self.cfg.index.get("nprobe", 32)),
                ef_search=int(self.cfg.index.get("ef_search", 64)),
            )
        return self._index

    @property
    def encoder(self) -> ClipEncoder:
        if self._encoder is None:
            self._encoder = ClipEncoder(self.cfg)
        return self._encoder

    def index_valid(self) -> tuple[bool, str | None]:
        ok, error = validate_meta(
            self.dir,
            kind="clip_image",
            model=f"{self.cfg.models.clip.name}/{self.cfg.models.clip.pretrained}",
            dataset_hash_expected=_dataset_hash(self.cfg),
        )
        if not ok:
            return False, error
        if not self.index_path.exists() or not self.ids_path.exists():
            return False, "CLIP index files are missing"
        return True, None

    def search_text(self, query: str, k: int, templates: list[str] | None = None) -> list[tuple[int, float]]:
        encoder = self.encoder
        if templates and len(templates) > 1 and bool(
            self.cfg.strategies.clip.get("prompt_ensemble", False)
        ):
            vector = encoder.encode_text_with_templates(query, templates)
        else:
            vector = encoder.encode_texts([query])
        scores, ids = self.index().search(vector, k)
        return [(int(pk), float(score)) for score, pk in zip(scores, ids, strict=False)
                if float(score) > -2.0]

    def search_vector(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        scores, ids = self.index().search(vector, k)
        return [(int(pk), float(score)) for score, pk in zip(scores, ids, strict=False)]

    def reconstruct(self, pk: int) -> np.ndarray:
        return self.index().reconstruct([pk])


def _dataset_hash(cfg: AttrDict) -> str:
    path = manifest_path(cfg)
    if not path.exists():
        return ""
    from tsr.artifacts import dataset_hash

    return dataset_hash(path)


class StrategyContext:
    """Everything a strategy may need, with lazy shared resources."""

    def __init__(self, cfg: AttrDict, store) -> None:
        self.cfg = cfg
        self.store = store
        self._clip: ClipBackbone | None = None
        self._text: TextEncoder | None = None
        self._rerankers: dict[str, CrossEncoderReranker] = {}

    @property
    def clip(self) -> ClipBackbone:
        if self._clip is None:
            self._clip = ClipBackbone(self.cfg)
        return self._clip

    @property
    def text(self) -> TextEncoder:
        if self._text is None:
            self._text = TextEncoder(self.cfg)
        return self._text

    def reranker(self, model: str | None) -> CrossEncoderReranker:
        name = model or "BAAI/bge-reranker-base"
        if name not in self._rerankers:
            self._rerankers[name] = CrossEncoderReranker(name)
        return self._rerankers[name]

    def clip_templates(self) -> list[str]:
        return list(self.cfg.strategies.clip.get("prompt_templates") or ["{}"])


class Strategy(abc.ABC):
    key: str = ""
    title: str = ""
    description: str = ""

    def __init__(self, ctx: StrategyContext) -> None:
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.store = ctx.store
        self.ready = False
        self.load_error: str | None = None
        self._loaded = False

    # ------------------------------------------------------------------ #
    @property
    def index_dir(self) -> Path:
        return strategy_index_dir(self.cfg, self.key)

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            self.load()
        except Exception as exc:  # noqa: BLE001 - readiness is surfaced, not raised
            self.ready = False
            self.load_error = str(exc)
            logger.warning("Strategy %s not ready: %s", self.key, exc)

    @abc.abstractmethod
    def load(self) -> None:
        """Verify artifacts and load indexes. Must set ``ready``/``load_error``."""

    @abc.abstractmethod
    def search(self, query: str, k: int, **options) -> SearchOutput:
        ...

    def stats(self) -> dict:
        self.ensure_loaded()
        meta = read_meta(self.index_dir)
        return {
            "strategy": self.key,
            "ready": self.ready,
            "error": self.load_error,
            "num_images": self.store.count(),
            "meta": meta,
        }

    def similar(self, pk: int, k: int) -> SearchOutput:
        """Default image-to-image search over the shared CLIP vectors."""
        started = time.perf_counter()
        backbone = self.ctx.clip
        ok, error = backbone.index_valid()
        if not ok:
            raise RuntimeError(error or "CLIP index unavailable")
        vector = backbone.reconstruct(pk)
        if vector.shape[0] == 0:
            raise ValueError("image is not in the CLIP index")
        ranked = [(candidate, score) for candidate, score in backbone.search_vector(vector, k + 1)
                  if candidate != pk][:k]
        hits = [Hit(candidate, score, {"clip": score}) for candidate, score in ranked]
        return SearchOutput(
            query=str(pk),
            strategy=self.key,
            hits=hits,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )

    @staticmethod
    def top_n(scores: dict[int, float], k: int) -> list[tuple[int, float]]:
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[: max(0, k)]


def result_records(store, hits: list[Hit]):
    records = store.get_many([hit.image_pk for hit in hits])
    return store.enrich(records)
