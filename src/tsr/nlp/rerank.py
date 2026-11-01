"""Optional cross-encoder reranker (strategy B and the C+rerank ablation)."""

from __future__ import annotations

import numpy as np

from tsr.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-base"


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

        if self._model is None:
            self._load()
        inputs = self._tokenizer(
            [query] * len(docs), docs, padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits = self._model(**inputs).logits.view(-1).float()
        return torch.sigmoid(logits).cpu().numpy().astype(np.float32)
