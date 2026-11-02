"""Dense text encoder (BGE by default) used by strategies B and C."""

from __future__ import annotations

import threading
from collections.abc import Iterable

import numpy as np

from tsr.config import AttrDict
from tsr.encoders.clip import resolve_device
from tsr.logging_utils import get_logger

logger = get_logger(__name__)


class TextEncoder:
    """Lazily-loaded HuggingFace encoder with BGE CLS pooling.

    BGE checkpoints are trained with the CLS token, not mean pooling; using the
    pooled mean here would silently degrade every dense ranking. Set
    ``models.text.pooling: mean`` for checkpoints (e.g. E5) trained that way.
    The same model must encode gallery documents and queries; the artifact
    metadata records the model name and dimension so a mismatch is detected at
    load time instead of silently producing bad rankings.
    """

    def __init__(self, cfg: AttrDict) -> None:
        model_cfg = cfg.models.text
        self.model_name: str = str(model_cfg.name)
        self.batch_size: int = int(model_cfg.get("batch_size", 64))
        self.max_length: int = int(model_cfg.get("max_length", 512))
        self.pooling: str = str(model_cfg.get("pooling", "cls")).lower()
        self.query_instruction: str = str(model_cfg.get("query_instruction", "") or "")
        self.cache_dir: str | None = model_cfg.get("cache_dir")
        self.device: str = resolve_device(str(model_cfg.get("device", "auto")))
        self._tokenizer = None
        self._model = None
        self._dim: int | None = None
        self._load_lock = threading.Lock()

    def load(self) -> None:
        # Startup preload and the first visitor can race here.
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._load()

    def _load(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading text encoder %s on %s ...", self.model_name, self.device)
        kwargs = {}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        self._model = AutoModel.from_pretrained(self.model_name, **kwargs)
        self._model.to(self.device).eval()
        self._dim = int(self._model.config.hidden_size)
        logger.info("Text encoder ready. Dimension = %d", self._dim)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.load()
        assert self._dim is not None
        return self._dim

    def encode(self, texts: Iterable[str], *, is_query: bool = False) -> np.ndarray:
        import torch

        self.load()
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        if is_query and self.query_instruction:
            texts = [self.query_instruction + text for text in texts]
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                hidden = self._model(**inputs).last_hidden_state
            if self.pooling == "mean":
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden[:, 0]
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            out.append(pooled.detach().cpu().float().numpy())
        return np.concatenate(out, axis=0)
