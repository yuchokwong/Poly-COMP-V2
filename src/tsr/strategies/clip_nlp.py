"""Strategy B: CLIP image score fused with human annotation signals.

Signals:
* CLIP text->image similarity (visual)
* dense vector similarity over the annotation document (semantic)
* SQLite FTS5 BM25 over field-weighted annotation columns (lexical)

Expansion terms only enter the BM25 query. The dense and CLIP encoders always
see the original query, which keeps the semantic intent stable.
"""

from __future__ import annotations

import time

from tsr.artifacts import dataset_hash, read_meta, validate_meta
from tsr.config import manifest_path
from tsr.index import VectorIndex
from tsr.logging_utils import get_logger
from tsr.nlp.analyze import expand_query
from tsr.nlp.fusion import fuse, normalize
from tsr.strategies.base import Hit, SearchOutput, Strategy

logger = get_logger(__name__)


class ClipNlpStrategy(Strategy):
    key = "clip_nlp"
    title = "CLIP + NLP"
    description = (
        "Fuses CLIP visual similarity with a dense index over Meme-Cap "
        "annotations and field-weighted BM25, then optionally reranks with a "
        "cross-encoder. Uses human-written captions/metaphors."
    )

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._doc_index: VectorIndex | None = None

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
                f"{error}. Build it with `python -m tsr build --strategy clip_nlp`."
            )
            return
        index_file = self.index_dir / "doc.faiss"
        ids_file = self.index_dir / "doc_ids.npy"
        if not index_file.exists() or not ids_file.exists():
            self.ready = False
            self.load_error = (
                "Annotation doc index files are missing. "
                "Run `python -m tsr build --strategy clip_nlp`."
            )
            return
        self._doc_index = VectorIndex.load(index_file, ids_file)
        meta = read_meta(self.index_dir)
        doc_count = self.store.count_strategy_docs(self.key)
        nlp_count = self.store.count_nlp_docs()
        if not doc_count or not nlp_count:
            self._doc_index = None
            self.ready = False
            self.load_error = (
                f"Strategy B has {doc_count} annotation documents and {nlp_count} BM25 "
                "rows. Run `python -m tsr build --strategy clip_nlp --force`."
            )
            return
        indexed = int(meta.get("count") or 0)
        if indexed != doc_count:
            self._doc_index = None
            self.ready = False
            self.load_error = (
                f"Annotation index has {indexed} vectors but the database has "
                f"{doc_count} documents. Run "
                "`python -m tsr build --strategy clip_nlp --force`."
            )
            return
        self.ready = True
        self.load_error = None

    # ------------------------------------------------------------------ #
    def _doc_scores(self, query: str, fetch: int) -> tuple[dict[int, float], str | None]:
        assert self._doc_index is not None
        try:
            vector = self.ctx.text.encode([query], is_query=True)
        except Exception as exc:  # noqa: BLE001 - missing transformers/model
            logger.warning("Dense doc encoding unavailable: %s", exc)
            return {}, str(exc)
        scores, ids = self._doc_index.search(vector, fetch)
        return {int(pk): float(score) for score, pk in zip(scores, ids, strict=False)}, None

    def search(self, query: str, k: int, **options) -> SearchOutput:
        self.ensure_loaded()
        if not self.ready:
            raise RuntimeError(self.load_error or "strategy clip_nlp is not ready")
        started = time.perf_counter()
        cfg = self.cfg.strategies.clip_nlp
        fetch = max(int(k) * int(cfg.get("fetch_multiplier", 5)), 100)

        clip_scores: dict[int, float] = {}
        try:
            if self.ctx.clip.index_valid()[0]:
                clip_scores = dict(
                    self.ctx.clip.search_text(query, fetch, self.ctx.clip_templates())
                )
        except Exception as exc:  # noqa: BLE001 - visual signal is optional
            logger.warning("CLIP signal unavailable for strategy B: %s", exc)

        doc_scores, doc_error = self._doc_scores(query, fetch)

        terms = expand_query(
            query,
            self.cfg.get("lexicon") or {},
            float(cfg.get("expansion_weight", 0.5)),
        )
        bm25_weights = cfg.get("bm25_weights") or {}
        weight_tuple = (
            float(bm25_weights.get("scene", 1.0)),
            float(bm25_weights.get("meaning", 1.6)),
            float(bm25_weights.get("metaphor", 1.8)),
            float(bm25_weights.get("title", 0.6)),
        )
        bm25_scores: dict[int, float] = dict(
            self.store.bm25(terms.seed, fetch, weight_tuple)
        )
        if bool(cfg.get("expand", True)) and terms.expansion:
            expansion = self.store.bm25(terms.expansion, fetch, weight_tuple)
            weight = float(cfg.get("expansion_weight", 0.5))
            for pk, score in expansion:
                bm25_scores[pk] = max(bm25_scores.get(pk, 0.0), weight * score)

        weights = cfg.get("weights") or {}
        signal_weights = {
            "clip": float(weights.get("clip", 0.5)),
            "doc": float(weights.get("doc", 0.3)),
            "bm25": float(weights.get("bm25", 0.2)),
        }
        signals = {"clip": clip_scores, "doc": doc_scores, "bm25": bm25_scores}
        fused = fuse(
            signals,
            signal_weights,
            method=str(cfg.get("fusion", "minmax")),
        )
        if not fused:
            return SearchOutput(
                query=query,
                strategy=self.key,
                hits=[],
                latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
                expanded_queries=[query] + terms.expansion,
                debug={"doc_error": doc_error},
            )

        candidates = sorted(fused)
        # Display components are always min-max scaled so the UI bars are comparable
        # even when fusion itself uses z-score or RRF.
        component_scores = {
            name: normalize(scores, "minmax", candidates=candidates)
            for name, scores in signals.items()
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
                except Exception as exc:  # noqa: BLE001 - reranking is optional
                    rerank_error = str(exc)
                    logger.warning("Cross-encoder rerank skipped: %s", exc)

        ranked = self.top_n(fused, k)
        hits = []
        for pk, score in ranked:
            components = {
                name: float(values.get(pk, 0.0))
                for name, values in component_scores.items()
                if values and any(values.get(candidate, 0.0) for candidate in candidates)
            }
            if pk in rerank_scores:
                components["rerank"] = float(rerank_scores[pk])
            components["fused"] = float(score)
            hits.append(Hit(image_pk=pk, score=float(score), components=components))

        return SearchOutput(
            query=query,
            strategy=self.key,
            hits=hits,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
            expanded_queries=[query] + terms.expansion,
            debug={
                "doc_error": doc_error,
                "rerank_error": rerank_error,
                "rerank_skipped": bool(rerank_cfg.get("enabled", False))
                and not rerank_scores,
                "num_candidates": len(candidates),
                "signal_sizes": {name: len(scores) for name, scores in signals.items()},
                "lexicon_phrases": terms.phrases,
            },
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
            "num_docs": meta.get("count"),
            "dim": meta.get("dim"),
            "model": meta.get("model"),
            "built_at": meta.get("built_at"),
            "meta": meta,
        }


STRATEGY = ClipNlpStrategy
