"""Meme-Cap dataset acquisition and preparation.

The dataset ships annotations as JSON and images as a large PNG archive. We keep
only the images that have annotations, convert them to bounded-size JPEGs, and
write a manifest that the rest of the pipeline consumes.

Sources
-------
* annotations: https://github.com/eujhwang/meme-cap (raw JSON on GitHub)
* images: HuggingFace mirror ``hieupth/memecap`` (zip), Google Drive fallback
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

from PIL import Image

from tsr.config import (
    AttrDict,
    archive_path,
    data_dir,
    images_dir,
    manifest_path,
    raw_captions_dir,
    raw_dir,
    raw_images_dir,
)
from tsr.logging_utils import get_logger
from tsr.manifest import ManifestRecord, manifest_sha256, write_manifest

logger = get_logger(__name__)

ANNOTATION_FILES = ("memes-trainval.json", "memes-test.json")
_ZIP_IMAGE_PREFIX = "memecap/images/"
_ZIP_CAPTION_PREFIX = "memecap/caption/"


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_annotations(cfg: AttrDict, force: bool = False) -> dict[str, Path]:
    """Download the two annotation JSON files if they are missing."""
    import requests

    out_dir = raw_captions_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = dict(cfg.dataset.annotations)
    paths: dict[str, Path] = {}
    for split, url in urls.items():
        dest = out_dir / f"memes-{split}.json"
        paths[split] = dest
        if dest.exists() and dest.stat().st_size > 0 and not force:
            continue
        logger.info("Downloading %s annotations -> %s", split, dest)
        resp = requests.get(str(url), timeout=120)
        resp.raise_for_status()
        tmp = dest.with_suffix(".json.part")
        tmp.write_bytes(resp.content)
        os.replace(tmp, dest)
    return paths


def download_archive(
    cfg: AttrDict,
    source: str = "hf",
    url: str | None = None,
    force: bool = False,
) -> Path:
    """Download the image archive with HTTP range resume.

    ``source='gdrive'`` requires ``gdown`` and falls back to an explicit error
    telling the user where to place the zip manually.
    """
    dest = archive_path(cfg)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        logger.info("Archive already present: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    if source == "gdrive":
        gdrive_id = cfg.dataset.get("gdrive_id")
        if not gdrive_id:
            raise RuntimeError("dataset.gdrive_id is not configured")
        try:
            import gdown
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Google Drive download needs gdown: pip install gdown. "
                f"Or download id={gdrive_id} manually and save it as {dest}."
            ) from exc
        tmp = dest.with_suffix(dest.suffix + ".part")
        gdown.download(id=str(gdrive_id), output=str(tmp), quiet=False, fuzzy=True)
        os.replace(tmp, dest)
        return dest

    target = url or str(cfg.dataset.zip_url)
    import requests

    part = dest.with_suffix(dest.suffix + ".part")
    downloaded = part.stat().st_size if part.exists() and not force else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    logger.info("Downloading %s (resume at %d bytes)", target, downloaded)
    with requests.get(target, stream=True, timeout=300, headers=headers) as resp:
        if downloaded and resp.status_code == 416:
            os.replace(part, dest)
            return dest
        resp.raise_for_status()
        mode = "ab" if downloaded and resp.status_code == 206 else "wb"
        if mode == "wb":
            downloaded = 0
        with part.open(mode) as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    os.replace(part, dest)
    return dest


def extract_archive(cfg: AttrDict, force: bool = False) -> Path:
    """Extract only the meme images and caption JSONs from the archive."""
    zip_path = archive_path(cfg)
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Archive not found: {zip_path}. Run `python -m tsr prepare --source hf`."
        )
    out_dir = raw_dir(cfg) / "memecap"
    marker = out_dir / ".extract_complete.json"
    if marker.exists() and not force:
        try:
            meta = json.loads(marker.read_text(encoding="utf-8"))
            if meta.get("zip_size") == zip_path.stat().st_size:
                logger.info("Archive already extracted: %s", out_dir)
                return out_dir
        except (json.JSONDecodeError, OSError):
            pass

    image_dir = raw_images_dir(cfg)
    caption_dir = raw_captions_dir(cfg)
    image_dir.mkdir(parents=True, exist_ok=True)
    caption_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or "__MACOSX/" in name:
                continue
            if name.startswith(_ZIP_IMAGE_PREFIX) and name.lower().endswith(".png"):
                target = image_dir / Path(name).name
            elif name.startswith(_ZIP_CAPTION_PREFIX) and name.lower().endswith(".json"):
                target = caption_dir / Path(name).name
            else:
                continue
            if target.exists() and target.stat().st_size == info.file_size and not force:
                continue
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
            extracted += 1
    marker.write_text(
        json.dumps(
            {"zip_size": zip_path.stat().st_size, "extracted": extracted}, indent=2
        ),
        encoding="utf-8",
    )
    logger.info("Extracted %d files into %s", extracted, out_dir)
    return out_dir


# --------------------------------------------------------------------------- #
# Annotations
# --------------------------------------------------------------------------- #
def _clean_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = _clean_text(value)
        return [text] if text else []
    out = []
    for item in value:
        text = _clean_text(item)
        if text:
            out.append(text)
    return out


def normalize_record(raw: dict, split: str) -> dict | None:
    """Normalise one Meme-Cap annotation record."""
    fname = _clean_text(raw.get("img_fname"))
    if not fname:
        return None
    metaphors = []
    for item in raw.get("metaphors") or []:
        if not isinstance(item, dict):
            continue
        metaphor = _clean_text(item.get("metaphor"))
        meaning = _clean_text(item.get("meaning"))
        if metaphor and meaning:
            metaphors.append({"metaphor": metaphor, "meaning": meaning})
    return {
        "img_fname": Path(fname).name,
        "post_id": _clean_text(raw.get("post_id")),
        "split": split,
        "title": _clean_text(raw.get("title")),
        "url": _clean_text(raw.get("url")),
        "img_captions": _as_list(raw.get("img_captions")),
        "meme_captions": _as_list(raw.get("meme_captions")),
        "metaphors": metaphors,
        "category": _clean_text(raw.get("category")) or "memes",
    }


def parse_annotations(captions_dir: str | Path) -> dict[str, dict]:
    """Read all annotation JSONs, keyed by image file name."""
    captions_dir = Path(captions_dir)
    records: dict[str, dict] = {}
    for path in sorted(captions_dir.glob("*.json")):
        name = path.name.lower()
        if "test" in name:
            split = "test"
        elif "trainval" in name or "train" in name or "val" in name:
            split = "trainval"
        else:
            split = "unknown"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed annotation file %s: %s", path, exc)
            continue
        if isinstance(data, dict):
            data = data.get("data") or data.get("annotations") or []
        for raw in data:
            if not isinstance(raw, dict):
                continue
            record = normalize_record(raw, split)
            if record is not None:
                records.setdefault(record["img_fname"], record)
    logger.info("Parsed %d annotated images from %s", len(records), captions_dir)
    return records


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dhash(image: Image.Image, size: int = 8) -> str:
    small = image.convert("L").resize((size + 1, size), Image.LANCZOS)
    if hasattr(small, "get_flattened_data"):  # Pillow >= 11.3
        pixels = list(small.get_flattened_data())
    else:
        pixels = list(small.getdata())
    bits = []
    for row in range(size):
        for col in range(size):
            left = pixels[row * (size + 1) + col]
            right = pixels[row * (size + 1) + col + 1]
            bits.append("1" if left > right else "0")
    return "".join(bits)


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def prepare_gallery(
    cfg: AttrDict,
    limit: int | None = None,
    force: bool = False,
) -> int:
    """Convert raw Meme-Cap PNGs to JPEGs and write ``manifest.jsonl``."""
    manifest_file = manifest_path(cfg)
    if manifest_file.exists() and not force:
        from tsr.manifest import read_manifest

        existing = read_manifest(manifest_file)
        if existing:
            logger.info("Manifest already exists (%d records); use --force to rebuild.",
                        len(existing))
            return len(existing)

    image_root = raw_images_dir(cfg)
    if not image_root.exists():
        raise FileNotFoundError(
            f"Raw images not found: {image_root}. Run `python -m tsr prepare` first."
        )
    annotations = parse_annotations(raw_captions_dir(cfg))
    out_dir = images_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_side = int(cfg.dataset.get("max_side", 768))
    quality = int(cfg.dataset.get("quality", 90))
    dedup = str(cfg.dataset.get("dedup", "sha1")).lower()
    if dedup not in {"sha1", "none"}:
        raise ValueError(f"dataset.dedup: expected 'sha1' or 'none', got {dedup!r}")

    files = sorted(p for p in image_root.glob("*.png") if p.name in annotations)
    missing = len(annotations) - len(files)
    if missing > 0:
        logger.warning("%d annotated images have no image file", missing)
    if limit is not None:
        files = files[: int(limit)]

    seen_sha1: dict[str, str] = {}
    records: list[ManifestRecord] = []
    skipped = 0
    for path in files:
        raw = annotations[path.name]
        try:
            with Image.open(path) as img:
                img.load()
                rgb = _to_rgb(img)
                if max(rgb.size) > max_side:
                    rgb.thumbnail((max_side, max_side), Image.LANCZOS)
                sha1 = _sha1_file(path)
                if dedup == "sha1" and sha1 in seen_sha1:
                    skipped += 1
                    logger.debug("Duplicate image %s (same as %s)", path.name, seen_sha1[sha1])
                    continue
                seen_sha1[sha1] = path.name
                image_id = f"memecap_{path.stem}"
                rel_path = f"memecap/{path.stem}.jpg"
                dest = out_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                rgb.save(dest, "JPEG", quality=quality, optimize=True, progressive=True)
                width, height = rgb.size
                phash = _dhash(rgb)
        except Exception as exc:  # noqa: BLE001 - one bad image must not stop the run
            logger.warning("Failed to prepare %s: %s", path.name, exc)
            continue
        records.append(
            ManifestRecord(
                image_id=image_id,
                img_fname=path.name,
                split=raw["split"],
                path=rel_path,
                width=width,
                height=height,
                sha1=sha1,
                phash=phash,
                title=raw["title"],
                source_url=raw["url"],
                post_id=raw["post_id"],
                img_captions=raw["img_captions"],
                meme_captions=raw["meme_captions"],
                metaphors=raw["metaphors"],
                category=raw["category"],
            )
        )
        if len(records) % 500 == 0:
            logger.info("Prepared %d / %d images", len(records), len(files))

    write_manifest(records, manifest_file)
    meta = {
        "num_records": len(records),
        "num_annotations": len(annotations),
        "num_without_annotations": missing,
        "num_duplicates": skipped,
        "manifest_sha256": manifest_sha256(manifest_file),
        "max_side": max_side,
        "quality": quality,
    }
    (data_dir(cfg) / "manifest.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %d records to %s (%d duplicates skipped)", len(records),
                manifest_file, skipped)
    return len(records)
