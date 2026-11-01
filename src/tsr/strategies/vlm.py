"""Strategy C: VLM-generated descriptions, matched at query time.

The VLM never runs at query time. Offline captioning stores a structured
document per image; this strategy embeds those documents and optionally fuses
the result with the shared CLIP visual score.
"""

from __future__ import annotations

import time

from tsr.artifacts import dataset_hash, read_meta, validate_meta
from tsr.config import manifest_path, vlm_doc_filters, vlm_provenance
from tsr.index import VectorIndex
from tsr.logging_utils import get_logger
from tsr.nlp.fusion import fuse, normalize
from tsr.strategies.base import Hit, SearchOutput, Strategy

logger = get_logger(__name__)

# Below this fraction of the gallery the index is treated as still being built.
MIN_COVERAGE = 0.95


class VlmStrategy(Strategy):
    key = "vlm"
    title = "VLM descriptions"
    description = (
        "Qwen2.5-VL reads each meme offline and writes scene / OCR / meaning / "
        "metaphor text. Retrieval matches that text and can fuse CLIP visual "
        "similarity. No human annotations are used."
    )

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._doc_index: VectorIndex | None = None
        self._meta: dict = {}

    def load(self) -> None:
        doc_model = str(self.cfg.models.text.name)
        expected_hash = dataset_hash(manifest_path(self.cfg))
        ok, error = validate_meta(
            self.index_dir,
            kind="doc_index",
            model=doc_model,
            dataset_hash_expected=expected_hash or None,
        )
        if not ok:
            self.ready = False
            self.load_error = (
                f"{error}. Run `python -m tsr vlm-caption` and then "
                "`python -m tsr build --strategy vlm`."
            )
            return
        index_file = self.index_dir / "doc.faiss"
        ids_file = self.index_dir / "doc_ids.npy"
        if not index_file.exists() or not ids_file.exists():
            self.ready = False
            self.load_error = (
                "VLM doc index files are missing. Run `python -m tsr build --strategy vlm`."
            )
            return
        self._doc_index = VectorIndex.load(index_file, ids_file)
        meta = read_meta(self.index_dir)
        self._meta = meta
        provenance = vlm_provenance(self.cfg)
        if not provenance["enabled"]:
            self._doc_index = None
            self.ready = False
            self.load_error = (
                "VLM backend is disabled (models.vlm.backend=none); set it to "
                "qwen2_5_vl to enable strategy C."
            )
            return
        mismatch = [
            f"{field}={meta.get(field)!r} (config {provenance[key]!r})"
            for field, key in (
                ("vlm_backend", "backend"),
                ("vlm_model", "model"),
                ("prompt_version", "prompt_version"),
            )
            if provenance[key] and meta.get(field) != provenance[key]
        ]
        if mismatch:
            # The index would embed descriptions from a generator the current
            # config no longer names; serving it silently mixes provenance.
            self._doc_index = None
            self.ready = False
            self.load_error = (
                "VLM index was built from a different generator: "
                + "; ".join(mismatch)
                + ". Rerun `python -m tsr build --strategy vlm --force`."
            )
            return
        count = int(meta.get("count") or 0)
        loaded = int(self._doc_index.size)
        if loaded != count:
            # meta.json is written after the vector files; an interrupted
            # rebuild leaves a smaller index beside the old count.
            self._doc_index = None
            self.ready = False
            self.load_error = (
                f"VLM index has {loaded} vectors but meta.json records {count}. "
                "Rerun `python -m tsr build --strategy vlm --force`."
            )
            return
        total = self.store.count()
        if total <= 0:
            self._doc_index = None
            self.ready = False
            self.load_error = (
                "Gallery database is empty. Run `python -m tsr prepare` and "
                "`python -m tsr build --strategy gallery`."
            )
            return
        if self.store.vlm_count(**vlm_doc_filters(self.cfg)) <= 0:
            # Retrieval would still answer from the index, but rerank and the
            # detail panel read vlm_docs; a wiped table must not serve.
            self._doc_index = None
            self.ready = False
            self.load_error = (
                "No VLM descriptions are stored for the current backend/model/prompt "
                "version. Run `python -m tsr vlm-caption` and then "
                "`python -m tsr build --strategy vlm --force`."
            )
            return
        # A half-captioned gallery would otherwise serve a 2% index as "ready".
        if count < total * MIN_COVERAGE:
            self._doc_index = None
            self.ready = False
            self.load_error = (
                f"VLM index covers only {count} of {total} images. Finish "
                "`python -m tsr vlm-caption`, then rerun "
                "`python -m tsr build --strategy vlm --force`."
            )
            return
        self.ready = True
        self.load_error = None

    def search(self, query: str, k: int, **options) -> SearchOutput:
        self.ensure_loaded()
        if not self.ready:
            raise RuntimeError(self.load_error or "strategy vlm is not ready")
        started = time.perf_counter()
        cfg = self.cfg.strategies.vlm
        fetch = max(int(k) * int(cfg.get("fetch_multiplier", 3)), 100)

        assert self._doc_index is not None
        doc_error = None
        doc_scores: dict[int, float] = {}
        try:
            vector = self.ctx.text.encode([query], is_query=True)
            scores, ids = self._doc_index.search(vector, fetch)
            doc_scores = {int(pk): float(score) for score, pk in zip(scores, ids, strict=False)}
        except Exception as exc:  # noqa: BLE001 - report as a load failure
            doc_error = str(exc)
            logger.warning("VLM doc encoding failed: %s", exc)

        clip_weight = float(cfg.get("clip_weight", 0.2))
        clip_scores: dict[int, float] = {}
        if clip_weight > 0:
            try:
                if self.ctx.clip.index_valid()[0]:
                    clip_scores = dict(
                        self.ctx.clip.search_text(query, fetch, self.ctx.clip_templates())
                    )
            except Exception as exc:  # noqa: BLE001 - optional signal
                logger.warning("CLIP fusion unavailable for strategy C: %s", exc)

        weights = {"doc": 1.0 - clip_weight if clip_scores else 1.0,
                   "clip": clip_weight if clip_scores else 0.0}
        fused = fuse(
            {"doc": doc_scores, "clip": clip_scores},
            weights,
            method=str(cfg.get("fusion", "minmax")),
        )
        if not fused:
            return SearchOutput(
                query=query,
                strategy=self.key,
                hits=[],
                latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
                expanded_queries=[query],
                debug={"doc_error": doc_error},
            )

        candidates = sorted(fused)
        components = {
            "doc": normalize(doc_scores, "minmax", candidates=candidates),
            "clip": normalize(clip_scores, "minmax", candidates=candidates) if clip_scores else {},
        }

        rerank_cfg = cfg.get("rerank") or {}
        rerank_scores: dict[int, float] = {}
        rerank_error: str | None = None
        if bool(rerank_cfg.get("enabled", False)):
            top_n = int(rerank_cfg.get("top_n", 60))
            rerank_ids = [pk for pk, _ in self.top_n(fused, top_n)]
            docs = self.store.strategy_docs_for(self.key, rerank_ids)
            if docs:
                try:
                    values = self.ctx.reranker(
                        str(rerank_cfg.get("model", "BAAI/bge-reranker-base"))
                    ).score(query, [docs.get(pk, "") for pk in rerank_ids])
                    rerank_scores = {
                        pk: float(value) for pk, value in zip(rerank_ids, values, strict=False)
                    }
                    alpha = float(rerank_cfg.get("alpha", 0.3))
                    for pk, value in rerank_scores.items():
                        fused[pk] = (1 - alpha) * fused.get(pk, 0.0) + alpha * value
                except Exception as exc:  # noqa: BLE001 - optional
                    rerank_error = str(exc)
                    logger.warning("Cross-encoder rerank skipped: %s", exc)

        hits = []
        for pk, score in self.top_n(fused, k):
            item = {
                name: float(values.get(pk, 0.0))
                for name, values in components.items()
                if values
            }
            if pk in rerank_scores:
                item["rerank"] = float(rerank_scores[pk])
            item["fused"] = float(score)
            hits.append(Hit(image_pk=pk, score=float(score), components=item))

        coverage = min(1.0, self._indexed_docs() / max(1, self.store.count()))
        return SearchOutput(
            query=query,
            strategy=self.key,
            hits=hits,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
            expanded_queries=[query],
            debug={
                "doc_error": doc_error,
                "rerank_error": rerank_error,
                "coverage": round(coverage, 4),
                "num_candidates": len(candidates),
            },
        )

    def _indexed_docs(self, meta: dict | None = None) -> int:
        """Artifacts actually in the loaded index (what readiness checks)."""
        if meta is None:
            meta = self._meta or read_meta(self.index_dir)
        return int(meta.get("count") or 0)

    def stats(self) -> dict:
        self.ensure_loaded()
        # Report the artifacts actually loaded, not whatever a rebuild wrote to
        # disk while the process kept serving: a fresh read here would show a
        # coverage/ready pair that search does not use.
        meta = self._meta or read_meta(self.index_dir)
        total = max(1, self.store.count())
        coverage = min(1.0, self._indexed_docs(meta) / total)
        return {
            "strategy": self.key,
            "title": self.title,
            "description": self.description,
            "ready": self.ready,
            "error": self.load_error,
            "num_images": self.store.count(),
            "num_docs": meta.get("count"),
            "coverage": round(coverage, 4),
            "dim": meta.get("dim"),
            "model": meta.get("model"),
            "vlm_model": meta.get("vlm_model"),
            "built_at": meta.get("built_at"),
            "meta": meta,
        }


STRATEGY = VlmStrategy
