"""FAISS-backed vector index.

Vectors are L2-normalised so inner product equals cosine similarity. Every
index persists an ``ids.npy`` array mapping FAISS rows to ``images.id`` values.
The ids file is mandatory: the old silent ``np.arange`` fallback could misalign
search results without any error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tsr.logging_utils import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - import guard
    import faiss
except Exception:  # pragma: no cover
    faiss = None


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class VectorIndex:
    def __init__(
        self,
        dim: int,
        index_type: str = "flat",
        nlist: int = 1024,
        nprobe: int = 32,
        m: int = 32,
        ef_search: int = 64,
    ) -> None:
        if faiss is None:  # pragma: no cover
            raise RuntimeError("faiss is not installed. `pip install faiss-cpu`.")
        self.dim = int(dim)
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.m = m
        self.ef_search = ef_search
        self.index = self._make()
        self.ids: np.ndarray = np.empty((0,), dtype=np.int64)
        self._trained = index_type == "flat"

    # ------------------------------------------------------------------ #
    def _make(self):
        if self.index_type == "hnsw":
            idx = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efSearch = self.ef_search
            return idx
        if self.index_type in {"ivf", "pq"}:
            quantizer = faiss.IndexFlatIP(self.dim)
            if self.index_type == "pq":
                m = self.m if self.dim % self.m == 0 else (8 if self.dim % 8 == 0 else 1)
                idx = faiss.IndexIVFPQ(
                    quantizer, self.dim, self.nlist, m, 8, faiss.METRIC_INNER_PRODUCT
                )
            else:
                idx = faiss.IndexIVFFlat(
                    quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
                )
            idx.nprobe = self.nprobe
            return idx
        return faiss.IndexFlatIP(self.dim)

    def add(self, embeddings: np.ndarray, ids: np.ndarray) -> None:
        embeddings = normalize(embeddings)
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if embeddings.shape[0] != ids.shape[0]:
            raise ValueError(
                f"Got {embeddings.shape[0]} vectors but {ids.shape[0]} ids; "
                "refusing to build a misaligned index."
            )
        if self.index_type in {"ivf", "pq"} and not self._trained:
            if embeddings.shape[0] < self.nlist * 39:
                logger.warning(
                    "Only %d vectors for nlist=%d; falling back to a flat index.",
                    embeddings.shape[0],
                    self.nlist,
                )
                self.index_type = "flat"
                self.index = self._make()
                self._trained = True
            else:
                self.index.train(embeddings)
                self._trained = True
        self.index.add(embeddings)
        self.ids = np.concatenate([self.ids, ids])

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.index.ntotal == 0:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
        query = normalize(query)
        k = min(int(k), int(self.index.ntotal))
        scores, positions = self.index.search(query, k)
        scores = scores[0]
        positions = positions[0]
        valid = positions >= 0
        return scores[valid], self.ids[positions[valid]]

    def reconstruct(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        position = {int(v): i for i, v in enumerate(self.ids.tolist())}
        positions = [position[int(i)] for i in ids if int(i) in position]
        if not positions:
            return np.empty((0, self.dim), dtype=np.float32)
        try:
            vectors = np.vstack([self.index.reconstruct(int(p)) for p in positions])
        except Exception as exc:  # pragma: no cover - index-type dependent
            raise RuntimeError("This index type does not support reconstruction") from exc
        return normalize(vectors)

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    # ------------------------------------------------------------------ #
    def save(self, index_file: str | Path, ids_file: str | Path) -> None:
        index_file = Path(index_file)
        ids_file = Path(ids_file)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        idx_tmp = index_file.with_suffix(index_file.suffix + ".tmp")
        ids_tmp = ids_file.parent / (ids_file.name + ".tmp.npy")
        faiss.write_index(self.index, str(idx_tmp))
        np.save(str(ids_tmp), self.ids)
        idx_tmp.replace(index_file)
        ids_tmp.replace(ids_file)

    @classmethod
    def load(
        cls,
        index_file: str | Path,
        ids_file: str | Path,
        *,
        index_type: str = "flat",
        nprobe: int = 32,
        ef_search: int = 64,
    ) -> VectorIndex:
        index_file = Path(index_file)
        ids_file = Path(ids_file)
        if not index_file.exists():
            raise FileNotFoundError(f"Index not found: {index_file}")
        if not ids_file.exists():
            raise FileNotFoundError(
                f"Index id map not found: {ids_file}. Rebuild the index."
            )
        raw = faiss.read_index(str(index_file))
        # Trust the serialised FAISS object, not the caller's hint: a requested
        # IVF/PQ build may have fallen back to flat for a small gallery, and
        # applying IVF-only attributes to a flat index used to crash on load.
        name = type(raw).__name__
        if "IVFPQ" in name or "PQ" in name:
            detected = "pq"
        elif "IVF" in name:
            detected = "ivf"
        elif "HNSW" in name:
            detected = "hnsw"
        else:
            detected = "flat"
        obj = cls.__new__(cls)
        obj.dim = int(raw.d)
        obj.index_type = detected
        obj.nlist = int(getattr(raw, "nlist", 0) or 0)
        obj.nprobe = nprobe
        obj.m = 32
        obj.ef_search = ef_search
        if detected in {"ivf", "pq"}:
            raw.nprobe = nprobe
        if detected == "hnsw":
            raw.hnsw.efSearch = ef_search
        obj.index = raw
        obj._trained = True
        obj.ids = np.load(str(ids_file)).astype(np.int64)
        if obj.ids.shape[0] != raw.ntotal:
            raise ValueError(
                f"Index has {raw.ntotal} vectors but {obj.ids.shape[0]} ids; rebuild."
            )
        return obj
