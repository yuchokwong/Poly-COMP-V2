from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tsr.strategies.clip_nlp import ClipNlpStrategy
from tsr.strategies.clip_only import ClipOnlyStrategy
from tsr.strategies.vlm import VlmStrategy


class FakeClip:
    def __init__(self, entries=(), available=True):
        self.entries = list(entries)
        self.available = available

    def index_valid(self):
        return (True, None) if self.available else (False, "CLIP index missing")

    def index(self):
        return object()

    def search_text(self, query, k, templates=None):
        return self.entries[:k]


class FakeText:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def encode(self, texts, *, is_query=False):
        self.calls.append((list(texts), is_query))
        if self.fail:
            raise RuntimeError("no transformers")
        return np.ones((len(list(texts)), 4), dtype=np.float32)


class FakeIndex:
    def __init__(self, entries):
        self.entries = list(entries)

    @property
    def size(self):
        return len(self.entries)

    def search(self, vector, k):
        entries = self.entries[:k]
        scores = np.array([score for _, score in entries], dtype=np.float32)
        ids = np.array([pk for pk, _ in entries], dtype=np.int64)
        return scores, ids


class FakeReranker:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def __call__(self, model_name):
        self.model_name = model_name
        return self

    def score(self, query, docs):
        self.calls.append((query, list(docs)))
        return self.values


class FakeStore:
    def __init__(
        self, bm25=None, docs=None, total=3, vlm_docs=3, nlp_docs=3, strategy_docs=None
    ):
        self._bm25 = bm25 or []
        self._docs = docs or {}
        self._total = total
        self._vlm_docs = vlm_docs
        self._nlp_docs = nlp_docs
        self._strategy_docs = total if strategy_docs is None else strategy_docs
        self.bm25_calls = []

    def count_strategy_docs(self, strategy):
        return self._strategy_docs

    def count_nlp_docs(self):
        return self._nlp_docs

    def bm25(self, terms, k, weights):
        self.bm25_calls.append((list(terms), k, tuple(weights)))
        return self._bm25[:k]

    def strategy_docs_for(self, strategy, pks):
        return {pk: self._docs[pk] for pk in pks if pk in self._docs}

    def count(self):
        return self._total

    def vlm_count(self, **_filters):
        return self._vlm_docs


def make_ctx(cfg, *, clip=None, text=None, store=None, reranker=None):
    return SimpleNamespace(
        cfg=cfg,
        store=store or FakeStore(),
        clip=clip or FakeClip(),
        text=text or FakeText(),
        reranker=reranker or FakeReranker([0.5]),
        clip_templates=lambda: ["{}"],
    )


def test_clip_only_filters_non_positive_scores(cfg):
    ctx = make_ctx(
        cfg,
        clip=FakeClip([(1, 0.9), (2, 0.8), (3, -0.1)]),
    )
    strategy = ClipOnlyStrategy(ctx)
    output = strategy.search("a cat", k=2)

    assert strategy.ready is True
    assert [hit.image_pk for hit in output.hits] == [1, 2]
    assert output.hits[0].components == {"clip": 0.9}
    assert output.strategy == "clip"
    assert output.latency_ms >= 0


def test_clip_only_reports_unavailable(cfg):
    ctx = make_ctx(cfg, clip=FakeClip(available=False))
    strategy = ClipOnlyStrategy(ctx)
    strategy.ensure_loaded()
    assert strategy.ready is False
    assert "build --strategy clip" in strategy.load_error

    with pytest.raises(RuntimeError):
        strategy.search("cat", k=5)


def test_clip_nlp_fuses_three_signals(cfg):
    cfg.strategies.clip_nlp.rerank.enabled = False
    ctx = make_ctx(
        cfg,
        clip=FakeClip([(1, 0.9), (2, 0.8)]),
        store=FakeStore(bm25=[(3, 3.0)]),
    )
    strategy = ClipNlpStrategy(ctx)
    strategy._loaded = True
    strategy.ready = True
    strategy._doc_index = FakeIndex([(2, 0.95), (1, 0.6)])

    output = strategy.search("cat at dinner", k=3)

    assert [hit.image_pk for hit in output.hits] == [1, 2, 3]
    top = output.hits[0].components
    assert set(top) == {"clip", "doc", "bm25", "fused"}
    assert top["clip"] == pytest.approx(1.0)
    assert top["fused"] == pytest.approx(output.hits[0].score)
    assert output.debug["num_candidates"] == 3
    assert output.debug["doc_error"] is None
    assert ctx.text.calls and ctx.text.calls[0][1] is True
    # BM25 receives only content terms from the original query.
    assert ctx.store.bm25_calls[0][0] == ["cat", "dinner"]


def test_clip_nlp_survives_missing_visual_and_dense_signals(cfg):
    cfg.strategies.clip_nlp.rerank.enabled = False
    ctx = make_ctx(
        cfg,
        clip=FakeClip(available=False),
        text=FakeText(fail=True),
        store=FakeStore(bm25=[(3, 2.0)]),
    )
    strategy = ClipNlpStrategy(ctx)
    strategy._loaded = True
    strategy.ready = True
    strategy._doc_index = FakeIndex([])

    output = strategy.search("fire", k=3)

    assert [hit.image_pk for hit in output.hits] == [3]
    assert "clip" not in output.hits[0].components
    assert "doc" not in output.hits[0].components
    assert output.debug["doc_error"] == "no transformers"


def test_clip_nlp_optional_rerank(cfg):
    cfg.strategies.clip_nlp.rerank.enabled = True
    cfg.strategies.clip_nlp.rerank.alpha = 0.3
    reranker = FakeReranker([0.1, 0.9])
    ctx = make_ctx(
        cfg,
        clip=FakeClip(),
        store=FakeStore(docs={1: "doc one", 2: "doc two"}),
        reranker=reranker,
    )
    strategy = ClipNlpStrategy(ctx)
    strategy._loaded = True
    strategy.ready = True
    strategy._doc_index = FakeIndex([(1, 0.9), (2, 0.8)])

    output = strategy.search("cat", k=2)

    assert reranker.calls
    assert "rerank" in output.hits[0].components
    assert output.hits[0].image_pk == 1
    assert output.debug["rerank_error"] is None


def test_clip_nlp_unavailable_when_index_missing(cfg):
    strategy = ClipNlpStrategy(make_ctx(cfg))
    strategy.ensure_loaded()
    assert strategy.ready is False
    assert "build --strategy clip_nlp" in strategy.load_error
    with pytest.raises(RuntimeError):
        strategy.search("cat", k=3)


def test_vlm_fuses_documents_with_optional_clip(cfg):
    cfg.strategies.vlm.clip_weight = 0.2
    ctx = make_ctx(
        cfg,
        clip=FakeClip([(2, 0.8), (1, 0.6)]),
        store=FakeStore(total=3, vlm_docs=3),
    )
    strategy = VlmStrategy(ctx)
    strategy._loaded = True
    strategy.ready = True
    strategy._doc_index = FakeIndex([(1, 0.9), (2, 0.7)])
    strategy._meta = {"count": 3}

    output = strategy.search("a cat", k=2)

    assert [hit.image_pk for hit in output.hits] == [1, 2]
    assert set(output.hits[0].components) == {"doc", "clip", "fused"}
    assert output.debug["coverage"] == 1.0


def test_vlm_pure_text_when_clip_weight_is_zero(cfg):
    cfg.strategies.vlm.clip_weight = 0.0
    ctx = make_ctx(cfg, clip=FakeClip([(2, 0.8)]))
    strategy = VlmStrategy(ctx)
    strategy._loaded = True
    strategy.ready = True
    strategy._doc_index = FakeIndex([(2, 0.9), (1, 0.4)])

    output = strategy.search("cat", k=2)

    assert [hit.image_pk for hit in output.hits] == [2, 1]
    assert set(output.hits[0].components) == {"doc", "fused"}


def test_vlm_refuses_an_index_that_only_covers_part_of_the_gallery(cfg, monkeypatch):
    """Captioning can be interrupted; a 2% index must not report ready."""
    import tsr.strategies.vlm as vlm_module

    index_dir = cfg.paths.data_dir + "/index/vlm"
    from pathlib import Path

    Path(index_dir).mkdir(parents=True)
    (Path(index_dir) / "doc.faiss").write_bytes(b"")
    (Path(index_dir) / "doc_ids.npy").write_bytes(b"")
    monkeypatch.setattr(vlm_module, "validate_meta", lambda *a, **k: (True, None))
    monkeypatch.setattr(
        vlm_module,
        "read_meta",
        lambda *_a, **_k: {
            "count": 10,
            "dim": 4,
            "model": "bge",
            "vlm_backend": "qwen2_5_vl",
            "vlm_model": "Qwen/Qwen2.5-VL-3B-Instruct",
            "prompt_version": "v1",
        },
    )
    monkeypatch.setattr(
        vlm_module,
        "VectorIndex",
        SimpleNamespace(load=lambda *a: FakeIndex([(i, 0.5) for i in range(1, 11)])),
    )

    strategy = VlmStrategy(make_ctx(cfg, store=FakeStore(total=100, vlm_docs=10)))
    strategy.load()

    assert strategy.ready is False
    assert "covers only 10 of 100" in strategy.load_error


def test_clip_nlp_refuses_an_index_without_documents(cfg, monkeypatch):
    """A meta.json and a FAISS file are not enough: the docs must exist."""
    import tsr.strategies.clip_nlp as nlp_module

    index_dir = Path(cfg.paths.data_dir) / "index" / "clip_nlp"
    index_dir.mkdir(parents=True)
    (index_dir / "doc.faiss").write_bytes(b"")
    (index_dir / "doc_ids.npy").write_bytes(b"")
    monkeypatch.setattr(nlp_module, "validate_meta", lambda *a, **k: (True, None))
    monkeypatch.setattr(nlp_module, "read_meta", lambda *_a, **_k: {"count": 3})
    monkeypatch.setattr(nlp_module, "VectorIndex", SimpleNamespace(load=lambda *a: object()))

    strategy = ClipNlpStrategy(
        make_ctx(cfg, store=FakeStore(nlp_docs=0, strategy_docs=0))
    )
    strategy.load()

    assert strategy.ready is False
    assert "0 annotation documents" in strategy.load_error
    assert "build --strategy clip_nlp --force" in strategy.load_error


def test_clip_nlp_refuses_an_index_that_lags_the_database(cfg, monkeypatch):
    import tsr.strategies.clip_nlp as nlp_module

    index_dir = Path(cfg.paths.data_dir) / "index" / "clip_nlp"
    index_dir.mkdir(parents=True)
    (index_dir / "doc.faiss").write_bytes(b"")
    (index_dir / "doc_ids.npy").write_bytes(b"")
    monkeypatch.setattr(nlp_module, "validate_meta", lambda *a, **k: (True, None))
    monkeypatch.setattr(nlp_module, "read_meta", lambda *_a, **_k: {"count": 2})
    monkeypatch.setattr(nlp_module, "VectorIndex", SimpleNamespace(load=lambda *a: object()))

    strategy = ClipNlpStrategy(
        make_ctx(cfg, store=FakeStore(nlp_docs=3, strategy_docs=3))
    )
    strategy.load()

    assert strategy.ready is False
    assert "2 vectors" in strategy.load_error and "3 documents" in strategy.load_error


def test_vlm_refuses_an_index_built_by_another_generator(cfg, monkeypatch):
    import tsr.strategies.vlm as vlm_module

    index_dir = Path(cfg.paths.data_dir) / "index" / "vlm"
    index_dir.mkdir(parents=True)
    (index_dir / "doc.faiss").write_bytes(b"")
    (index_dir / "doc_ids.npy").write_bytes(b"")
    monkeypatch.setattr(vlm_module, "validate_meta", lambda *a, **k: (True, None))
    monkeypatch.setattr(
        vlm_module,
        "read_meta",
        lambda *_a, **_k: {
            "count": 3,
            "vlm_backend": "qwen2_5_vl",
            "vlm_model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "prompt_version": "v1",
        },
    )
    monkeypatch.setattr(vlm_module, "VectorIndex", SimpleNamespace(load=lambda *a: object()))

    strategy = VlmStrategy(make_ctx(cfg, store=FakeStore(total=3, vlm_docs=3)))
    strategy.load()

    assert strategy.ready is False
    assert "different generator" in strategy.load_error
    assert "Qwen/Qwen2.5-VL-7B-Instruct" in strategy.load_error


def test_vlm_unavailable_when_index_missing(cfg):
    strategy = VlmStrategy(make_ctx(cfg))
    strategy.ensure_loaded()
    assert strategy.ready is False
    assert "vlm-caption" in strategy.load_error
    with pytest.raises(RuntimeError):
        strategy.search("cat", k=3)


def test_vlm_reports_unavailable_when_the_backend_is_disabled(cfg, monkeypatch):
    import tsr.strategies.vlm as vlm_module

    cfg.models.vlm.backend = "none"
    index_dir = Path(cfg.paths.data_dir) / "index" / "vlm"
    index_dir.mkdir(parents=True)
    (index_dir / "doc.faiss").write_bytes(b"")
    (index_dir / "doc_ids.npy").write_bytes(b"")
    monkeypatch.setattr(vlm_module, "validate_meta", lambda *a, **k: (True, None))
    monkeypatch.setattr(vlm_module, "read_meta", lambda *_a, **_k: {"count": 3})
    monkeypatch.setattr(vlm_module, "VectorIndex", SimpleNamespace(load=lambda *a: object()))

    strategy = VlmStrategy(make_ctx(cfg, store=FakeStore(total=3, vlm_docs=3)))
    strategy.load()

    assert strategy.ready is False
    assert "disabled" in strategy.load_error
    with pytest.raises(RuntimeError):
        strategy.search("cat", k=3)


def test_vlm_coverage_comes_from_the_loaded_index_not_the_database(cfg, monkeypatch):
    """A DB that gained docs after the build must not inflate index coverage."""
    import tsr.strategies.vlm as vlm_module

    cfg.models.vlm.backend = "qwen2_5_vl"
    index_dir = Path(cfg.paths.data_dir) / "index" / "vlm"
    index_dir.mkdir(parents=True)
    (index_dir / "doc.faiss").write_bytes(b"")
    (index_dir / "doc_ids.npy").write_bytes(b"")
    monkeypatch.setattr(vlm_module, "validate_meta", lambda *a, **k: (True, None))
    monkeypatch.setattr(
        vlm_module,
        "read_meta",
        lambda *_a, **_k: {
            "count": 1,
            "vlm_backend": "qwen2_5_vl",
            "vlm_model": "Qwen/Qwen2.5-VL-3B-Instruct",
            "prompt_version": "v1",
        },
    )
    monkeypatch.setattr(vlm_module, "VectorIndex", SimpleNamespace(load=lambda *a: FakeIndex([])))

    strategy = VlmStrategy(make_ctx(cfg, store=FakeStore(total=10, vlm_docs=10, strategy_docs=1)))
    stats = strategy.stats()

    assert stats["num_docs"] == 1
    assert stats["coverage"] == pytest.approx(0.1)
    assert stats["ready"] is False


def test_clip_only_refuses_an_empty_gallery(cfg):
    strategy = ClipOnlyStrategy(make_ctx(cfg, store=FakeStore(total=0)))
    strategy.load()

    assert strategy.ready is False
    assert "Gallery database is empty" in strategy.load_error


def _vlm_index_files(cfg):
    index_dir = Path(cfg.paths.data_dir) / "index" / "vlm"
    index_dir.mkdir(parents=True)
    (index_dir / "doc.faiss").write_bytes(b"")
    (index_dir / "doc_ids.npy").write_bytes(b"")
    return index_dir


def test_vlm_refuses_a_half_written_index(cfg, monkeypatch):
    """Index files are saved before meta.json; an interrupted build must not serve."""
    import tsr.strategies.vlm as vlm_module

    cfg.models.vlm.backend = "qwen2_5_vl"
    _vlm_index_files(cfg)
    monkeypatch.setattr(vlm_module, "validate_meta", lambda *a, **k: (True, None))
    monkeypatch.setattr(
        vlm_module,
        "read_meta",
        lambda *_a, **_k: {
            "count": 300,
            "vlm_backend": "qwen2_5_vl",
            "vlm_model": "Qwen/Qwen2.5-VL-3B-Instruct",
            "prompt_version": "v1",
        },
    )
    monkeypatch.setattr(
        vlm_module,
        "VectorIndex",
        SimpleNamespace(load=lambda *a: FakeIndex([(1, 0.5), (2, 0.5)])),
    )

    strategy = VlmStrategy(make_ctx(cfg, store=FakeStore(total=3, vlm_docs=3)))
    strategy.load()

    assert strategy.ready is False
    assert "2 vectors" in strategy.load_error and "300" in strategy.load_error


def test_vlm_refuses_when_the_documents_were_cleared(cfg, monkeypatch):
    """A forced gallery rebuild deletes vlm_docs; a leftover index must not serve."""
    import tsr.strategies.vlm as vlm_module

    cfg.models.vlm.backend = "qwen2_5_vl"
    _vlm_index_files(cfg)
    monkeypatch.setattr(vlm_module, "validate_meta", lambda *a, **k: (True, None))
    monkeypatch.setattr(
        vlm_module,
        "read_meta",
        lambda *_a, **_k: {
            "count": 3,
            "vlm_backend": "qwen2_5_vl",
            "vlm_model": "Qwen/Qwen2.5-VL-3B-Instruct",
            "prompt_version": "v1",
        },
    )
    monkeypatch.setattr(
        vlm_module,
        "VectorIndex",
        SimpleNamespace(load=lambda *a: FakeIndex([(1, 0.5), (2, 0.5), (3, 0.5)])),
    )

    strategy = VlmStrategy(make_ctx(cfg, store=FakeStore(total=3, vlm_docs=0)))
    strategy.load()

    assert strategy.ready is False
    assert "No VLM descriptions" in strategy.load_error


def test_vlm_stats_report_the_loaded_artifacts(cfg, monkeypatch):
    """stats() must describe what search uses, not a newer file on disk."""
    import tsr.strategies.vlm as vlm_module

    cfg.models.vlm.backend = "qwen2_5_vl"
    _vlm_index_files(cfg)
    monkeypatch.setattr(vlm_module, "validate_meta", lambda *a, **k: (True, None))
    monkeypatch.setattr(
        vlm_module,
        "VectorIndex",
        SimpleNamespace(load=lambda *a: FakeIndex([(1, 0.5), (2, 0.5), (3, 0.5)])),
    )
    loaded_meta = {
        "count": 3,
        "vlm_backend": "qwen2_5_vl",
        "vlm_model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "prompt_version": "v1",
    }
    fresh = dict(loaded_meta, count=1)
    calls = {"n": 0}

    def fake_read_meta(*_a, **_k):
        # The first read is the load; a later read on disk sees a 1-vector rebuild.
        calls["n"] += 1
        return loaded_meta if calls["n"] == 1 else fresh

    monkeypatch.setattr(vlm_module, "read_meta", fake_read_meta)

    strategy = VlmStrategy(make_ctx(cfg, store=FakeStore(total=3, vlm_docs=3)))
    strategy.ensure_loaded()
    stats = strategy.stats()

    assert stats["ready"] is True
    assert stats["num_docs"] == 3
    assert stats["coverage"] == pytest.approx(1.0)
