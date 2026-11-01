"""Dataset preparation and artifact build pipelines."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tsr.artifacts import config_hash, dataset_hash, read_meta, write_meta
from tsr.config import (
    AttrDict,
    cache_dir,
    db_path,
    images_dir,
    manifest_path,
    strategy_index_dir,
    vlm_doc_filters,
    vlm_provenance,
)
from tsr.encoders.clip import ClipEncoder
from tsr.encoders.text import TextEncoder
from tsr.index import VectorIndex
from tsr.logging_utils import get_logger
from tsr.manifest import content_hash, read_manifest
from tsr.memecap import download_annotations, download_archive, extract_archive, prepare_gallery
from tsr.store import GalleryStore

logger = get_logger(__name__)


def ensure_dataset(cfg: AttrDict, limit: int | None = None, force: bool = False) -> int:
    count = prepare_gallery(cfg, limit=limit, force=force)
    if count <= 0:
        raise RuntimeError(
            "No manifest records. Run `python -m tsr prepare` to download Meme-Cap."
        )
    return count


def prepare_dataset(
    cfg: AttrDict,
    source: str = "hf",
    limit: int | None = None,
    force: bool = False,
    skip_download: bool = False,
) -> int:
    if not skip_download:
        download_annotations(cfg, force=force)
        download_archive(cfg, source=source, force=force)
        extract_archive(cfg, force=force)
    return ensure_dataset(cfg, limit=limit, force=force)


def _dataset_hash(cfg: AttrDict) -> str:
    return dataset_hash(manifest_path(cfg))


def _index_layout(cfg: AttrDict) -> str:
    return str(cfg.index.get("type", "flat"))


# --------------------------------------------------------------------------- #
# Gallery + CLIP image index
# --------------------------------------------------------------------------- #
def open_store(cfg: AttrDict, reset: bool = False) -> GalleryStore:
    return GalleryStore(db_path(cfg), reset=reset)


def build_gallery(cfg: AttrDict, force: bool = False) -> int:
    """Insert the manifest into SQLite and build the shared CLIP image index."""
    manifest = read_manifest(manifest_path(cfg))
    if not manifest:
        raise RuntimeError("Manifest is empty. Run `python -m tsr prepare` first.")

    store = GalleryStore(db_path(cfg), reset=force)
    try:
        if store.count() == 0 or force:
            if force:
                store.clear("images")
            # Primary keys are assigned again on insert, both on a forced rebuild
            # and when inserting into an empty table; thumbnails are keyed by pk,
            # so old files would be served for the wrong images.
            thumbs = cache_dir(cfg) / "thumbs"
            if thumbs.exists():
                shutil.rmtree(thumbs, ignore_errors=True)
                logger.info("Cleared thumbnail cache before rebuilding the gallery")
            store.insert_images(manifest)
        records = store.iter_all()
        if not records:
            raise RuntimeError("Gallery database is empty after insertion.")
        manifest_ids = [rec.image_id for rec in manifest]
        if [rec.image_id for rec in records] != manifest_ids:
            raise RuntimeError(
                "Gallery database does not match the manifest. Rerun "
                "`python -m tsr build --strategy clip --force` to rebuild it."
            )
        enriched = store.enrich({rec.id: rec for rec in records})
        if content_hash(enriched.values()) != content_hash(manifest):
            raise RuntimeError(
                "Gallery rows were inserted from a different version of the "
                "manifest: image ids match but annotations differ. Rerun "
                "`python -m tsr build --strategy clip --force` to rebuild it."
            )
        ids = [rec.id for rec in records]
        if ids != list(range(ids[0], ids[0] + len(ids))):
            raise RuntimeError(f"images.id is not contiguous: {ids[:3]} ... {ids[-3:]}")

        encoder = ClipEncoder(cfg)
        root = images_dir(cfg)
        vectors: list[np.ndarray] = []
        batch_size = int(cfg.models.clip.get("batch_size", 64))
        for start in tqdm(range(0, len(records), batch_size), desc="CLIP embedding"):
            batch = records[start : start + batch_size]
            vectors.append(encoder.encode_images([root / rec.path for rec in batch]))
        matrix = np.concatenate(vectors, axis=0).astype(np.float32)
        if matrix.shape[0] != len(ids):
            raise RuntimeError(
                f"CLIP produced {matrix.shape[0]} vectors for {len(ids)} images; "
                "refusing to write a misaligned index."
            )
        zeros = int((np.linalg.norm(matrix, axis=1) == 0).sum())
        if zeros:
            logger.warning("%d images produced zero vectors (unreadable files)", zeros)

        index_type = _index_layout(cfg)
        index = VectorIndex(
            dim=matrix.shape[1],
            index_type=index_type,
            nlist=int(cfg.index.get("nlist", 1024)),
            nprobe=int(cfg.index.get("nprobe", 32)),
            m=int(cfg.index.get("m", 32)),
            ef_search=int(cfg.index.get("ef_search", 64)),
        )
        index.add(matrix, np.asarray(ids, dtype=np.int64))
        out_dir = strategy_index_dir(cfg, "clip")
        index.save(out_dir / "image.faiss", out_dir / "image_ids.npy")
        write_meta(
            out_dir,
            {
                "kind": "clip_image",
                "strategy": "clip",
                "model": f"{cfg.models.clip.name}/{cfg.models.clip.pretrained}",
                "dim": int(matrix.shape[1]),
                "count": int(matrix.shape[0]),
                "index_type": index.index_type,
                "dataset_hash": _dataset_hash(cfg),
                "config_hash": config_hash(cfg.models.clip),
            },
        )
        logger.info("CLIP index built: %d vectors (dim=%d) in %s",
                    index.size, matrix.shape[1], out_dir)
        return index.size
    finally:
        store.close()


def build_clip(cfg: AttrDict, force: bool = False) -> int:
    return build_gallery(cfg, force=force)


# --------------------------------------------------------------------------- #
# Strategy docs + dense indexes
# --------------------------------------------------------------------------- #
def _annotation_fields(rec) -> tuple[str, str, str, str]:
    scene = " / ".join(rec.img_captions)
    meaning = " / ".join(rec.meme_captions)
    metaphor = "; ".join(
        f"{item['metaphor']} means {item['meaning']}"
        for item in rec.metaphors
        if str(item.get("meaning", "")).strip().lower()
        != "not related to the meme context"
    )
    return scene, meaning, metaphor, rec.title or ""


def _embed_docs(
    cfg: AttrDict,
    encoder: TextEncoder,
    docs: list[str],
    out_dir: Path,
    ids: list[int],
    meta: dict,
) -> int:
    if not docs:
        raise RuntimeError(f"No documents to embed for {out_dir}; nothing to index.")
    vectors: list[np.ndarray] = []
    batch_size = int(cfg.models.text.get("batch_size", 64))
    for start in tqdm(range(0, len(docs), batch_size), desc="doc embedding"):
        vectors.append(encoder.encode(docs[start : start + batch_size]))
    matrix = np.concatenate(vectors, axis=0).astype(np.float32)
    if matrix.shape[0] != len(ids):
        raise RuntimeError(
            f"Encoded {matrix.shape[0]} documents for {len(ids)} ids; "
            "refusing to write a misaligned index."
        )
    index = VectorIndex(
        dim=matrix.shape[1],
        index_type=_index_layout(cfg),
        nlist=int(cfg.index.get("nlist", 1024)),
        nprobe=int(cfg.index.get("nprobe", 32)),
        m=int(cfg.index.get("m", 32)),
        ef_search=int(cfg.index.get("ef_search", 64)),
    )
    index.add(matrix, np.asarray(ids, dtype=np.int64))
    index.save(out_dir / "doc.faiss", out_dir / "doc_ids.npy")
    write_meta(
        out_dir,
        {
            "kind": "doc_index",
            "dim": int(matrix.shape[1]),
            "count": int(matrix.shape[0]),
            "index_type": index.index_type,
            "dataset_hash": _dataset_hash(cfg),
            **meta,
        },
    )
    return index.size


def _assert_gallery_matches_manifest(cfg: AttrDict, store: GalleryStore) -> None:
    """Refuse to attach strategy artifacts to a gallery from another dataset.

    Only ``build_gallery`` inserts rows; if the manifest changed since, the
    strategy would index documents for one image set and be searched against
    another gallery, silently misaligning every row id.
    """
    manifest = read_manifest(manifest_path(cfg))
    if not manifest:
        raise RuntimeError("Manifest is empty. Run `python -m tsr prepare` first.")
    rows = store.iter_all()
    stored = [rec.image_id for rec in rows]
    expected = [rec.image_id for rec in manifest]
    if stored != expected:
        raise RuntimeError(
            f"Gallery has {len(stored)} images but the manifest has {len(expected)} "
            "and they differ. Run `python -m tsr build --strategy clip --force` first."
        )
    enriched = store.enrich({rec.id: rec for rec in rows})
    if content_hash(enriched.values()) != content_hash(manifest):
        raise RuntimeError(
            "Gallery annotations are out of date with the manifest (same ids, "
            "different text). Run `python -m tsr build --strategy clip --force` first."
        )


def build_clip_nlp(cfg: AttrDict, force: bool = False) -> int:
    store = GalleryStore(db_path(cfg))
    try:
        _assert_gallery_matches_manifest(cfg, store)
        records = store.enrich(store.get_many([rec.id for rec in store.iter_all()]))
        if not records:
            raise RuntimeError("Gallery is empty. Run `python -m tsr build --strategy clip` first.")
        rows = []
        nlp_rows = []
        for pk in sorted(records):
            scene, meaning, metaphor, title = _annotation_fields(records[pk])
            dense = (
                f"scene: {scene}\nmeaning: {meaning}\n"
                f"metaphor: {metaphor}\ntitle: {title}"
            )
            rows.append((pk, dense))
            nlp_rows.append((pk, scene, meaning, metaphor, title))
        store.write_nlp_docs(nlp_rows)
        store.write_strategy_docs("clip_nlp", rows)
        encoder = TextEncoder(cfg)
        out_dir = strategy_index_dir(cfg, "clip_nlp")
        size = _embed_docs(
            cfg,
            encoder,
            [text for _, text in rows],
            out_dir,
            [pk for pk, _ in rows],
            {
                "strategy": "clip_nlp",
                "model": str(cfg.models.text.name),
                "config_hash": config_hash(cfg.strategies.clip_nlp),
            },
        )
        logger.info("Strategy B doc index built: %d documents", size)
        return size
    finally:
        store.close()


def build_vlm(cfg: AttrDict, force: bool = False) -> int:
    store = GalleryStore(db_path(cfg))
    try:
        _assert_gallery_matches_manifest(cfg, store)
        provenance = vlm_provenance(cfg)
        if not provenance["enabled"]:
            raise RuntimeError(
                "models.vlm.backend is 'none'. Set a backend before building strategy C."
            )
        docs = store.iter_vlm_docs(**vlm_doc_filters(cfg))
        if not docs:
            raise RuntimeError(
                "No VLM descriptions for the configured backend/model/prompt_version. "
                "Run `python -m tsr vlm-caption` first, or set models.vlm.backend=none "
                "to disable strategy C."
            )
        rows = [(int(doc["image_pk"]), str(doc["doc_text"])) for doc in docs]
        rows = [(pk, text) for pk, text in rows if text]
        if not rows:
            raise RuntimeError("VLM descriptions are all empty; regenerate them.")
        docs = [doc for doc in docs if doc.get("doc_text")]
        store.write_strategy_docs("vlm", rows)
        encoder = TextEncoder(cfg)
        out_dir = strategy_index_dir(cfg, "vlm")
        size = _embed_docs(
            cfg,
            encoder,
            [text for _, text in rows],
            out_dir,
            [pk for pk, _ in rows],
            {
                "strategy": "vlm",
                "model": str(cfg.models.text.name),
                "vlm_model": provenance["model"],
                "vlm_backend": provenance["backend"],
                "prompt_version": provenance["prompt_version"],
                "config_hash": config_hash(cfg.strategies.vlm),
            },
        )
        logger.info("Strategy C doc index built: %d documents", size)
        return size
    finally:
        store.close()


def build(cfg: AttrDict, strategy: str = "all", force: bool = False) -> dict[str, int]:
    strategy = strategy.lower()
    if strategy in {"a", "gallery"}:
        strategy = "clip"
    if strategy == "b":
        strategy = "clip_nlp"
    if strategy == "c":
        strategy = "vlm"

    results: dict[str, int] = {}
    if strategy in {"clip", "all"}:
        results["clip"] = build_clip(cfg, force=force)
    if strategy in {"clip_nlp", "all"}:
        if "clip" not in results and not strategy_index_dir(cfg, "clip").exists():
            raise RuntimeError("Build strategy clip before clip_nlp.")
        results["clip_nlp"] = build_clip_nlp(cfg, force=force)
    if strategy in {"vlm", "all"}:
        if str(cfg.models.vlm.get("backend", "qwen2_5_vl")).lower() == "none":
            logger.warning("VLM backend disabled; skipping strategy C.")
        else:
            try:
                results["vlm"] = build_vlm(cfg, force=force)
            except RuntimeError as exc:
                if strategy == "all":
                    logger.warning("Skipping strategy C: %s", exc)
                else:
                    raise
    if not results:
        raise ValueError(f"Nothing built for strategy={strategy!r}")
    return results


# --------------------------------------------------------------------------- #
# Thumbnails / stats
# --------------------------------------------------------------------------- #
def pregenerate_thumbnails(cfg: AttrDict, size: int = 480) -> int:
    from PIL import Image

    store = GalleryStore(db_path(cfg))
    root = images_dir(cfg)
    out_root = cache_dir(cfg) / "thumbs" / str(size)
    out_root.mkdir(parents=True, exist_ok=True)
    made = 0
    try:
        for rec in tqdm(store.iter_all(), desc=f"thumbs@{size}"):
            dest = out_root / f"{rec.id}.jpg"
            if dest.exists():
                continue
            src = root / rec.path
            if not src.exists():
                continue
            try:
                with Image.open(src) as img:
                    img = img.convert("RGB")
                    img.thumbnail((size, size), Image.LANCZOS)
                    img.save(dest, "JPEG", quality=80, optimize=True)
                made += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Thumbnail failed for %s: %s", rec.image_id, exc)
    finally:
        store.close()
    logger.info("Generated %d thumbnails at %s", made, out_root)
    return made


def stats(cfg: AttrDict) -> dict:
    path = db_path(cfg)
    if not path.exists():
        return {"ready": False, "error": "database missing"}
    try:
        store = GalleryStore(path)
    except Exception as exc:  # noqa: BLE001 - old schema is a normal state
        return {"ready": False, "error": str(exc)}
    try:
        out = {
            "ready": True,
            "num_images": store.count(),
            "vlm_docs": store.vlm_count(**vlm_doc_filters(cfg)),
            "strategies": {},
            "manifest": str(manifest_path(cfg)),
        }
        for key in ("clip", "clip_nlp", "vlm"):
            out["strategies"][key] = read_meta(strategy_index_dir(cfg, key))
        return out
    finally:
        store.close()


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
