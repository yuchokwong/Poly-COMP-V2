"""Offline VLM captioning for strategy C.

The VLM reads each image once and produces a structured description (literal
scene, overlaid text, intended meaning, metaphors, keywords). Retrieval then
uses the generated text only. The model is imported lazily so the API process
and strategies A/B never load several GB of weights.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tsr.config import AttrDict, images_dir, vlm_cache_path
from tsr.logging_utils import get_logger

logger = get_logger(__name__)

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are annotating a meme image for a text-to-image retrieval system. "
    "Describe only what is visible and the intended meaning. Do not identify "
    "private individuals or real people. Do not use outside knowledge of the "
    "meme template unless the visual evidence supports it."
)

USER_PROMPT = """Look at this meme and return ONLY a JSON object with these fields:
- "scene": literal people/objects/actions/setting, 1-2 sentences
- "ocr_text": all overlaid text verbatim in reading order; empty string if none
- "meaning": the intended joke, message, or emotion, 1-2 sentences
- "metaphors": array of at most 3 objects {"metaphor": "...", "meaning": "..."}; [] if none
- "emotions": array of 1-4 emotion words
- "keywords": 5-10 short retrieval terms (objects, template names, emotion, slang; no real people)

If text is unreadable, say so in "ocr_text" rather than guessing. Return valid JSON only."""


@dataclass
class VLMResult:
    scene: str = ""
    ocr_text: str = ""
    meaning: str = ""
    emotions: list[str] = field(default_factory=list)
    metaphors: list[dict] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    raw: str = ""

    def to_doc_text(self) -> str:
        parts = [f"scene: {self.scene}"]
        if self.ocr_text:
            parts.append(f"overlaid text: {self.ocr_text}")
        if self.meaning:
            parts.append(f"meaning: {self.meaning}")
        if self.metaphors:
            parts.append(
                "metaphors: "
                + "; ".join(f"{m['metaphor']} means {m['meaning']}" for m in self.metaphors)
            )
        if self.emotions:
            parts.append("emotions: " + ", ".join(self.emotions))
        if self.keywords:
            parts.append("keywords: " + ", ".join(self.keywords))
        return "\n".join(parts)


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,;]", value) if part.strip()]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("metaphor") or item.get("text")
                if text:
                    out.append(str(text))
        return out
    return []


# Qwen occasionally emits slightly broken JSON (an unescaped quote inside a
# value, a missing comma between members, a value split into two strings, or a
# generate call cut off by ``max_new_tokens``). Repairing those locally keeps a
# handful of images from losing their description; anything not understood is
# still rejected.
_MAX_REPAIRS = 40


def _object_span(text: str, start: int) -> str:
    """Return the first object, closing it when generation stopped mid-member."""
    depth = 0
    in_string = False
    escape = False
    last_separator = -1
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        elif char == "," and depth == 1:
            # Remember where a complete member ends in case the tail is cut.
            last_separator = i
    if last_separator > start:
        return text[start:last_separator] + "}"
    raise ValueError("unbalanced JSON object in model output")


def _previous_quote(text: str, before: int) -> int | None:
    """Index of the nearest unescaped double quote before ``before``."""
    index = before - 1
    while index >= 0:
        if text[index] == '"':
            backslashes = 0
            while index - backslashes - 1 >= 0 and text[index - backslashes - 1] == "\\":
                backslashes += 1
            if backslashes % 2 == 0:
                return index
        index -= 1
    return None


def _repair_at_error(text: str, error: json.JSONDecodeError) -> str | None:
    """Apply the repair that matches a JSON parser failure, or return None."""
    pos = error.pos
    if error.msg.startswith("Expecting ',' delimiter"):
        if text[pos : pos + 1] == '"':
            # A member value closed and the next key follows without a comma.
            return text[:pos] + ", " + text[pos:]
        quote = _previous_quote(text, pos)
        if quote is not None:
            # An unescaped quote inside a value ended the string early.
            return text[:quote] + '\\"' + text[quote + 1 :]
        return None
    if error.msg.startswith("Expecting ':' delimiter"):
        close = _previous_quote(text, pos)
        open_ = _previous_quote(text, close) if close is not None else None
        previous = _previous_quote(text, open_) if open_ is not None else None
        if previous is None:
            return None
        if not text[previous:open_].strip().startswith('",'):
            return None
        # '"key": "left", "right",' -> '"key": "left, right",'
        return text[:previous] + ", " + text[open_ + 1 :]
    return None


def _loads_json(text: str) -> dict:
    """Parse a repaired JSON object, raising ValueError when repair fails."""
    for _ in range(_MAX_REPAIRS):
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            repaired = _repair_at_error(text, error)
            if repaired is None or repaired == text:
                raise ValueError(
                    f"model output is not valid JSON: {error.msg} at char {error.pos}"
                ) from error
            text = repaired
    raise ValueError("model output is not valid JSON")


def _extract_json(text: str) -> dict:
    """Find and parse the first balanced JSON object in model output."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("model output contains no JSON object")
    return _loads_json(_object_span(cleaned, start))


def parse_vlm_output(text: str) -> VLMResult:
    data = _extract_json(text)
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    metaphors = []
    for item in data.get("metaphors") or []:
        if isinstance(item, dict):
            metaphor = str(item.get("metaphor") or "").strip()
            meaning = str(item.get("meaning") or "").strip()
            if metaphor:
                metaphors.append({"metaphor": metaphor, "meaning": meaning})
        elif isinstance(item, str) and item.strip():
            metaphors.append({"metaphor": item.strip(), "meaning": ""})
    return VLMResult(
        scene=str(data.get("scene") or "").strip(),
        ocr_text=str(
            data.get("ocr_text") or data.get("text_in_image") or data.get("ocr") or ""
        ).strip(),
        meaning=str(data.get("meaning") or "").strip(),
        emotions=_as_list(data.get("emotions")),
        metaphors=metaphors[:3],
        keywords=_as_list(data.get("keywords"))[:10],
        raw=text,
    )


class VLMBackend:
    name = "none"
    model = "none"
    available = False
    error: str | None = "VLM backend is disabled"

    def describe(self, image_path: str | Path) -> VLMResult:  # pragma: no cover - abstract
        raise NotImplementedError


class NoneBackend(VLMBackend):
    def __init__(self, reason: str = "VLM backend is disabled") -> None:
        self.error = reason


class QwenVLBackend(VLMBackend):
    name = "qwen2_5_vl"

    def __init__(self, cfg: AttrDict) -> None:
        vlm_cfg = cfg.models.vlm
        self.model = str(vlm_cfg.model)
        self.device = str(vlm_cfg.get("device", "auto"))
        self.dtype = str(vlm_cfg.get("dtype", "bfloat16"))
        self.attn_implementation = str(vlm_cfg.get("attn_implementation", "sdpa"))
        self.max_pixels = int(vlm_cfg.get("max_pixels", 1003520))
        self.min_pixels = int(vlm_cfg.get("min_pixels", 200704))
        self.max_new_tokens = int(vlm_cfg.get("max_new_tokens", 448))
        self.do_sample = bool(vlm_cfg.get("do_sample", False))
        self._model = None
        self._processor = None
        self.available = False
        self.error = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - optional dependency
            self.error = (
                "Qwen2.5-VL needs `transformers>=4.49`: pip install -r requirements.txt"
            )
            raise RuntimeError(self.error) from exc
        dtype = getattr(torch, self.dtype, torch.bfloat16)
        logger.info("Loading VLM %s on %s ...", self.model, self.device)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model,
            torch_dtype=dtype,
            attn_implementation=self.attn_implementation,
            device_map="auto" if self.device == "auto" else None,
        )
        if self.device != "auto":
            self._model.to(self.device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.model, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )
        # Decoder-only generation with padded batches needs left padding.
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        self.available = True
        self.device = str(next(self._model.parameters()).device)

    @staticmethod
    def _messages(image_path: str | Path) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]

    def describe(self, image_path: str | Path) -> VLMResult:
        return self.describe_many([image_path])[0]

    def describe_many(self, image_paths: Sequence[str | Path]) -> list[VLMResult]:
        """Caption several images in one generate call (batched prefill)."""
        paths = [str(path) for path in image_paths]
        if not paths:
            return []
        self.load()
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [self._messages(path) for path in paths]
        texts = [
            self._processor.apply_chat_template(
                item, tokenize=False, add_generation_prompt=True
            )
            for item in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
            )
        trimmed = [
            out[len(inp) :] for inp, out in zip(inputs.input_ids, generated, strict=False)
        ]
        outputs = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return [parse_vlm_output(output) for output in outputs]


def get_backend(cfg: AttrDict) -> VLMBackend:
    name = str(cfg.models.vlm.get("backend", "qwen2_5_vl")).lower()
    if name in {"none", "disabled", "off"}:
        return NoneBackend(
            "VLM backend disabled (models.vlm.backend=none); "
            "set TSR_MODELS__VLM__BACKEND=qwen2_5_vl to enable strategy C."
        )
    if name in {"qwen2_5_vl", "qwen", "qwen2.5-vl"}:
        return QwenVLBackend(cfg)
    return NoneBackend(f"Unknown VLM backend {name!r}")


# --------------------------------------------------------------------------- #
# Cache / batch generation
# --------------------------------------------------------------------------- #
def _load_cache(path: Path, backend: VLMBackend, prompt_version: str) -> dict[tuple, dict]:
    done: dict[tuple, dict] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("status") != "ok":
                continue
            if item.get("backend") != backend.name or item.get("prompt_version") != prompt_version:
                continue
            # A different model id is a different description; never reuse it.
            if item.get("model") not in (None, backend.model):
                continue
            key = (item.get("image_pk"), item.get("image_sha1"))
            done[key] = item
    return done


def _cached_doc_entry(entry: dict, backend: VLMBackend, prompt_version: str) -> dict:
    """Rebuild a ``vlm_docs`` row from a cache line (used when resuming)."""
    doc = entry.get("doc") or {}
    return {
        "image_pk": int(entry["image_pk"]),
        "backend": backend.name,
        "model": str(entry.get("model") or backend.model),
        "prompt_version": prompt_version,
        "scene": doc.get("scene", ""),
        "ocr_text": doc.get("ocr_text", ""),
        "meaning": doc.get("meaning", ""),
        "emotions": json.dumps(doc.get("emotions") or [], ensure_ascii=False),
        "metaphors": json.dumps(doc.get("metaphors") or [], ensure_ascii=False),
        "keywords": json.dumps(doc.get("keywords") or [], ensure_ascii=False),
        "doc_text": doc.get("doc_text", ""),
        "raw": "",
        "status": "ok",
        "latency_ms": entry.get("latency_ms"),
    }


def _new_entry(rec, backend: VLMBackend, prompt_version: str) -> dict:
    return {
        "image_pk": rec.id,
        "image_sha1": rec.sha1,
        "backend": backend.name,
        "model": backend.model,
        "prompt_version": prompt_version,
    }


def _flush_entry(fh, entry: dict) -> None:
    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def _record_success(
    store,
    fh,
    samples: list[dict],
    rec,
    entry: dict,
    result: VLMResult,
    elapsed_ms: float,
) -> None:
    doc_text = result.to_doc_text()
    entry.update(
        status="ok",
        doc={
            "scene": result.scene,
            "ocr_text": result.ocr_text,
            "meaning": result.meaning,
            "emotions": result.emotions,
            "metaphors": result.metaphors,
            "keywords": result.keywords,
            "doc_text": doc_text,
        },
        latency_ms=elapsed_ms,
    )
    store.upsert_vlm_doc(
        {
            "image_pk": rec.id,
            "backend": entry["backend"],
            "model": entry["model"],
            "prompt_version": entry["prompt_version"],
            "scene": result.scene,
            "ocr_text": result.ocr_text,
            "meaning": result.meaning,
            "emotions": json.dumps(result.emotions, ensure_ascii=False),
            "metaphors": json.dumps(result.metaphors, ensure_ascii=False),
            "keywords": json.dumps(result.keywords, ensure_ascii=False),
            "doc_text": doc_text,
            "raw": result.raw,
            "status": "ok",
            "latency_ms": elapsed_ms,
        }
    )
    if len(samples) < 20:
        samples.append({"record": rec, "doc": entry["doc"]})
    _flush_entry(fh, entry)


def _record_error(store, fh, rec, entry: dict, exc: Exception, elapsed_ms: float) -> None:
    entry.update(status="error", error=str(exc), latency_ms=elapsed_ms)
    store.upsert_vlm_doc(
        {
            "image_pk": rec.id,
            "backend": entry["backend"],
            "model": entry["model"],
            "prompt_version": entry["prompt_version"],
            "doc_text": "",
            "raw": "",
            "status": "error",
            "error": str(exc),
            "latency_ms": elapsed_ms,
        }
    )
    _flush_entry(fh, entry)


def caption_gallery(
    cfg: AttrDict,
    store,
    limit: int | None = None,
    backend: VLMBackend | None = None,
    prompt_version: str | None = None,
    out_html: str | Path | None = None,
) -> dict:
    """Generate VLM descriptions, persisting each result to a resumable cache."""
    backend = backend or get_backend(cfg)
    if not backend.available:
        load = getattr(backend, "load", None)
        if load is not None:
            try:
                load()
            except Exception as exc:  # noqa: BLE001 - surface a build error
                raise RuntimeError(backend.error or str(exc)) from exc
    if not backend.available:
        raise RuntimeError(backend.error or "VLM backend unavailable")
    prompt_version = prompt_version or str(cfg.models.vlm.get("prompt_version", PROMPT_VERSION))
    cache_path = vlm_cache_path(cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.parent / ".vlm.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_path.open("x").close()
    except FileExistsError as exc:
        raise RuntimeError(f"Another VLM run holds {lock_path}") from exc
    done = _load_cache(cache_path, backend, prompt_version)
    records = store.iter_all()
    if limit is not None:
        records = records[: int(limit)]
    root = images_dir(cfg)
    ok = errors = skipped = 0
    samples: list[dict] = []
    pending = []
    for rec in records:
        key = (rec.id, rec.sha1)
        if key in done:
            # The cache may outlive the database (a gallery rebuild drops
            # vlm_docs), so re-upsert cached descriptions on resume.
            store.upsert_vlm_doc(_cached_doc_entry(done[key], backend, prompt_version))
            skipped += 1
        else:
            pending.append(rec)
    batch_size = max(1, int(cfg.models.vlm.get("batch_size", 1)))
    describe_many = getattr(backend, "describe_many", None)
    try:
        with cache_path.open("a", encoding="utf-8") as fh:
            def note_progress() -> None:
                if (ok + errors) % 25 == 0:
                    logger.info(
                        "VLM progress: %d ok, %d errors, %d skipped", ok, errors, skipped
                    )

            for start in range(0, len(pending), batch_size):
                chunk = pending[start : start + batch_size]
                paths = [root / rec.path for rec in chunk]
                results: list[VLMResult] | None = None
                started = time.perf_counter()
                if len(chunk) > 1 and describe_many is not None:
                    try:
                        results = list(describe_many(paths))
                    except Exception as exc:  # noqa: BLE001 - retry individually
                        logger.warning(
                            "VLM batch of %d failed (%s); retrying one by one",
                            len(chunk),
                            exc,
                        )
                        results = None
                    if results is not None and len(results) != len(chunk):
                        results = None
                elapsed = (time.perf_counter() - started) * 1000.0

                if results is not None:
                    # One ``generate`` call; latency is shared across the batch.
                    per_item = elapsed / len(chunk)
                    for rec, result in zip(chunk, results, strict=True):
                        entry = _new_entry(rec, backend, prompt_version)
                        try:
                            _record_success(store, fh, samples, rec, entry, result, per_item)
                            ok += 1
                        except Exception as exc:  # noqa: BLE001 - keep going
                            _record_error(store, fh, rec, entry, exc, per_item)
                            errors += 1
                            logger.warning("VLM failed for %s: %s", rec.image_id, exc)
                        note_progress()
                    continue
                for rec, image_path in zip(chunk, paths, strict=True):
                    started = time.perf_counter()
                    entry = _new_entry(rec, backend, prompt_version)
                    try:
                        result = backend.describe(image_path)
                        _record_success(
                            store,
                            fh,
                            samples,
                            rec,
                            entry,
                            result,
                            (time.perf_counter() - started) * 1000.0,
                        )
                        ok += 1
                    except Exception as exc:  # noqa: BLE001 - keep the batch going
                        _record_error(
                            store, fh, rec, entry, exc, (time.perf_counter() - started) * 1000.0
                        )
                        errors += 1
                        logger.warning("VLM failed for %s: %s", rec.image_id, exc)
                    note_progress()
        if out_html:
            _write_sample_html(samples, out_html)
    finally:
        lock_path.unlink(missing_ok=True)
    logger.info("VLM finished: %d ok, %d errors, %d already cached", ok, errors, skipped)
    return {"ok": ok, "errors": errors, "skipped": skipped, "total": len(records)}


def _write_sample_html(samples: Sequence[dict], out: str | Path) -> None:
    rows = []
    for item in samples:
        rec = item["record"]
        doc = item["doc"]
        rows.append(
            f"<h3>{rec.image_id}</h3>"
            f'<img src="/api/image/{rec.id}" style="max-width:320px" />'
            f"<pre>{json.dumps(doc, indent=2, ensure_ascii=False)}</pre>"
        )
    Path(out).write_text(
        "<!doctype html><meta charset='utf-8'><title>VLM sample</title>"
        "<body style='font-family:system-ui;max-width:900px;margin:2rem auto'>"
        + "".join(rows)
        + "</body>",
        encoding="utf-8",
    )
