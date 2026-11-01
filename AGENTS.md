# AGENTS.md

Guidance for coding agents working in this repository.

## Project

Meme-Cap text-to-image retrieval (COMP5523). Three strategies — **A `clip`** (pure
OpenCLIP), **B `clip_nlp`** (CLIP + annotation dense/BM25 + optional rerank),
**C `vlm`** (offline Qwen2.5-VL descriptions) — share one gallery, one benchmark and
one FastAPI app. Python package `tsr` under `src/`, UI in `web/`, SQLite for metadata,
FAISS for vectors.

## Environment

- Docker (Python 3.11) is the reproducible target for deployment. Native installs
  also work where PyTorch/OpenCLIP wheels exist (verified on Python 3.14 + torch
  2.11 on Windows).
- Package source lives in `src/`; run modules with `PYTHONPATH=src` or
  `pip install -e .`.
- The base image does not install the VLM extras (`requirements-vlm.txt`); offline
  captioning runs on a GPU host natively. Serving C needs only the BGE text encoder.

## Common commands

```bash
# install (dev)
pip install -r requirements-dev.txt && pip install -e .

# small end-to-end loop (network required, ~300 images)
python -m tsr prepare --limit 300
python -m tsr build --strategy gallery
python -m tsr build --strategy clip_nlp
python -m tsr make-eval --num 60
python -m tsr eval --strategy clip,clip_nlp
python -m tsr serve            # http://localhost:8000

# full dataset (6,382 images)
python -m tsr prepare
python -m tsr vlm-caption      # GPU host, resumable
python -m tsr build --strategy all
python -m tsr compare --ablations

# tests + lint (always run before finishing; use PYTHONPATH when not installed)
PYTHONPATH=src pytest
PYTHONPATH=src ruff check src tests
```

Docker equivalents:

```bash
docker compose --profile tools run --rm build python -m tsr prepare --limit 300
docker compose --profile tools run --rm build python -m tsr build --strategy all
docker compose up -d --build
```

## Conventions

- Python 3.10+ syntax, `from __future__ import annotations`, 100-char lines.
- Config is a nested `AttrDict` loaded by `tsr.config.load_config`; add new options to
  `configs/default.yaml` and read them with `cfg.<section>.<key>` / `.get(default)`.
  Env overrides are `TSR_` + upper-cased path joined by `__`.
- Never commit `data/`, `.env` or generated indexes (see `.gitignore`).
- Tests that need FAISS/torch/transformers use `pytest.importorskip` so the suite
  degrades gracefully; unit tests use fake encoders and never download models.
- Keep the API response schemas in `src/tsr/schemas.py` in sync with the web UI.
- Runtime-adjustable knobs belong in `settings.py`; build-time/structural options
  belong in `configs/default.yaml`.
- Strategy keys are a closed set (`clip`, `clip_nlp`, `vlm`): unknown keys → 404,
  known-but-unbuilt → 503.

## Architecture notes

- `memecap.py` acquires annotations and images, converts PNG→JPEG, dedupes by SHA-1
  and writes `manifest.jsonl`; `manifest.py` defines the record and hashing.
- `store.py` owns SQLite schema v2 (`PRAGMA user_version = 2`). There is no
  migration from the old COCO schema — `GalleryStore.open()` raises
  `SchemaVersionError`; rebuild after deleting `data/`.
- `images.id` is the shared key across SQLite, FAISS row ids and `image_ids.npy`.
  Insert images in `image_id` order once; strategy rebuilds only rewrite
  `strategy_docs`/vector files and must never renumber `images`. Strategy builds
  compare a content hash of the stored annotations against the manifest (ids alone
  would accept corrected text), and `VectorIndex.add` refuses mismatched vector/id
  counts, so a misaligned index cannot be written.
- `index.py` stores L2-normalised vectors so inner product == cosine similarity and
  requires an id map (`FileNotFoundError`/count mismatch rather than `np.arange`).
  Every built index writes `meta.json` (schema, model, dim, count, manifest/config
  hash); `artifacts.py` validates it before loading.
- `build` ordering: `gallery` (or `clip`) inserts rows and builds the shared CLIP
  image index; `clip_nlp` reuses that index; `vlm` requires `vlm_docs` and is skipped
  with a warning under `--strategy all` when `models.vlm.backend: none`.
- `encoders/vlm.py` imports transformers/qwen-vl-utils lazily and is used only by
  `tsr vlm-caption` and the VLM build path. Never import it on API startup.
- Every read of `vlm_docs` (build filters, detail panels, stats counts) goes through
  `config.vlm_provenance` so rows from an older backend/model/prompt version can
  never be served or counted as coverage; the VLM index `meta.json` records the same
  triple and loading refuses a mismatch.
- `strategies/*` implement `Strategy` (`stats`, `search`) and return `Hit`s with
  per-signal `components`; `service.StrategyManager` loads shared resources lazily
  and reports readiness without crashing when an index is absent.
- `nlp/fusion.py` min-max/z-score/RRF fusion drops disabled or empty signals and
  renormalises the remaining weights. `nlp/analyze.py` query expansion feeds BM25
  only — never CLIP or the dense encoder.
- Evaluation is leakage-aware: `make-eval` rewrites test-split captions and rejects
  drafts whose 8-gram overlaps an annotation, whose ≥ `eval.min_copy_tokens` run is
  copied from the source, or whose Jaccard ≥ 0.6. Raw-caption
  queries are a separate, labelled diagnostic set — do not mix them into the main
  table.
- `compare.py` runs the same frozen queries through every ready strategy, uses paired
  bootstrap with a fixed seed for confidence intervals, and applies Holm–Bonferroni
  across the three pairwise comparisons. It also folds any
  `reports/eval_<key>_raw_captions.json` into a separately labelled
  `diagnostics` table (raw captions are leaky by construction; never mix them into
  the main metrics). Reports are the only source for the `/evaluation` page.
- Strategy C reports as unavailable when its doc index covers under 95% of the
  gallery, so an interrupted captioning run cannot masquerade as a built strategy.
- `api.py` serves the static pages from `web/`; `/api/search` is a compatibility
  alias for `/api/strategies/clip/search`.
