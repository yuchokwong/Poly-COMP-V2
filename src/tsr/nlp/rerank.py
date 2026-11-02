"""Optional cross-encoder reranker (strategy B and the C+rerank ablation)."""

from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np

from tsr.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-base"

# Documents are scored in length-sorted chunks: a single batch pads every pair
# to the longest document, which on this corpus doubles the token count.
_PAIRS_PER_BATCH = 16


class CrossEncoderReranker:
    """Lazily-loaded HuggingFace cross-encoder.

    Scores are squashed with a sigmoid so they can be blended with fused scores
    in [0, 1]. If ``transformers`` is unavailable the caller receives a clear
    ``RuntimeError`` and the strategy falls back to the first-stage ranking.
    """

    def __init__(self, model_name: str | None = None, device: str = "auto") -> None:
        self.model_name = model_name or DEFAULT_MODEL
        self.device = device
        self._tokenizer = None
        self._model = None
        self._load_lock = threading.Lock()

    def load(self) -> None:
        # Preload and the first reranked query can race; loading twice would
        # hold two copies of the 1.1 GB checkpoint.
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "cross-encoder reranking needs `transformers`: pip install transformers"
            ) from exc
        logger.info("Loading cross-encoder %s ...", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self.device).eval()

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        if not docs:
            return np.empty((0,), dtype=np.float32)
        import torch

        self.load()
        tokenized = self._tokenizer(docs, truncation=True, max_length=512)
        lengths = [len(ids) for ids in tokenized["input_ids"]]
        scores = np.empty((len(docs),), dtype=np.float32)
        for batch in _length_batches(lengths, _PAIRS_PER_BATCH):
            inputs = self._tokenizer(
                [query] * len(batch), [docs[i] for i in batch], padding=True,
                truncation=True, max_length=512, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self._model(**inputs).logits.view(-1).float()
            scores[batch] = torch.sigmoid(logits).cpu().numpy()
        return scores


def _length_batches(lengths: Sequence[int], batch_size: int) -> list[list[int]]:
    """Group indices of similar length so padding is paid only within a chunk."""
    order = sorted(range(len(lengths)), key=lengths.__getitem__)
    return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
