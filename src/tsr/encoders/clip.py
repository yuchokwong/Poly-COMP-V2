"""OpenCLIP wrapper (shared image backbone for all three strategies)."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from tsr.config import AttrDict
from tsr.logging_utils import get_logger

logger = get_logger(__name__)


def resolve_device(device: str = "auto") -> str:
    import torch

    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClipEncoder:
    """Lazily-loaded OpenCLIP image/text encoder.

    The model is not loaded on construction so commands that only inspect
    artifacts (``tsr stats``) stay dependency-light.
    """

    def __init__(self, cfg: AttrDict) -> None:
        model_cfg = cfg.models.clip
        self.model_name: str = str(model_cfg.name)
        self.pretrained: str = str(model_cfg.pretrained)
        self.batch_size: int = int(model_cfg.get("batch_size", 64))
        self.normalize: bool = bool(model_cfg.get("normalize", True))
        self.cache_dir: str | None = model_cfg.get("cache_dir")
        self.device: str = resolve_device(str(model_cfg.get("device", "auto")))
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._dim: int | None = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        # Startup preload and the first visitor can race here; loading twice
        # would double peak memory for a 1.6 GB backbone.
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._load()

    def _load(self) -> None:
        import open_clip
        import torch

        logger.info(
            "Loading OpenCLIP %s (%s) on %s ...",
            self.model_name,
            self.pretrained,
            self.device,
        )
        kwargs = {}
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = self.cache_dir
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device, **kwargs
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        with torch.inference_mode():
            probe = model.encode_text(self._tokenizer(["x"]).to(self.device))
        self._dim = int(probe.shape[-1])
        logger.info("CLIP ready. Embedding dimension = %d", self._dim)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.load()
        assert self._dim is not None
        return self._dim

    # ------------------------------------------------------------------ #
    def _finalize(self, features) -> np.ndarray:
        import torch

        if self.normalize:
            features = torch.nn.functional.normalize(features, dim=-1)
        return features.detach().to("cpu", dtype=torch.float32).numpy()

    def _encode_images(self, batch) -> np.ndarray:
        import torch

        with torch.inference_mode():
            feats = self._model.encode_image(batch.to(self.device))
        return self._finalize(feats)

    def _encode_text(self, tokens) -> np.ndarray:
        import torch

        with torch.inference_mode():
            feats = self._model.encode_text(tokens.to(self.device))
        return self._finalize(feats)

    def encode_images(self, paths: Sequence[str | Path]) -> np.ndarray:
        from PIL import Image

        self.load()
        out: list[np.ndarray] = []
        batch = []
        for path in paths:
            try:
                with Image.open(path) as img:
                    batch.append(self._preprocess(img.convert("RGB")))
            except Exception as exc:  # pragma: no cover - corrupt file
                logger.warning("Failed to read image %s: %s", path, exc)
                batch.append(None)
            if len(batch) >= self.batch_size:
                out.append(self._encode_batch(batch))
                batch = []
        if batch:
            out.append(self._encode_batch(batch))
        if not out:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.concatenate(out, axis=0)

    def _encode_batch(self, batch) -> np.ndarray:
        import torch

        valid = [item for item in batch if item is not None]
        if not valid:
            # Keep one row per input: dropping rows would silently shift every
            # later vector relative to its image id.
            logger.warning("Whole batch of %d images was unreadable", len(batch))
            return np.zeros((len(batch), self.dim), dtype=np.float32)
        feats = self._encode_images(torch.stack(valid))
        if len(valid) != len(batch):
            full = np.zeros((len(batch), feats.shape[1]), dtype=np.float32)
            it = iter(feats)
            for i, item in enumerate(batch):
                if item is not None:
                    full[i] = next(it)
            return full
        return feats

    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        self.load()
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        out = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            out.append(self._encode_text(self._tokenizer(chunk)))
        return np.concatenate(out, axis=0)

    def encode_text_with_templates(self, text: str, templates: Sequence[str]) -> np.ndarray:
        prompts = [t.format(text) if "{}" in t else f"{t} {text}" for t in templates]
        feats = self.encode_texts(prompts)
        if feats.shape[0] == 0:
            return feats
        mean = feats.mean(axis=0, keepdims=True)
        norm = np.linalg.norm(mean, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return (mean / norm).astype(np.float32)
