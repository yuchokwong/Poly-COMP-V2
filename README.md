# Meme Retrieval: CLIP vs CLIP+NLP vs VLM (COMP5523)

A text-to-image retrieval system built on the [Meme-Cap](https://github.com/eujhwang/meme-cap)
dataset. Three independent pipelines share one gallery, one evaluation set and one
FastAPI service, so their retrieval quality can be compared directly:

| Page | Strategy | Query-time signals | Extra knowledge |
| --- | --- | --- | --- |
| `/clip` | **A — CLIP only** | OpenCLIP text→image cosine similarity | none (pixels only) |
| `/clip-nlp` | **B — CLIP + NLP** | CLIP ⊕ dense annotation embeddings ⊕ field-weighted BM25, optional cross-encoder | human captions & metaphors |
| `/vlm` | **C — VLM** | dense index over Qwen2.5-VL descriptions, optional CLIP fusion | offline VLM descriptions |

`/compare` runs a query through all three pipelines side by side, and `/evaluation`
renders the same frozen benchmark across strategies.

```
                        ┌──────────── shared gallery ────────────┐
 Meme-Cap annotations ─►│ manifest → SQLite v2 → images + FAISS  │
 Meme-Cap PNGs ────────►│            (CLIP image vectors)        │
                        └───────┬────────────┬────────────┬──────┘
                                │            │            │
                        A: CLIP │  B: + NLP  │  C: + VLM  │
                                ▼            ▼            ▼
                          text→image   + BGE docs     + Qwen2.5-VL
                          cosine       + BM25         docs → BGE
                                       + rerank       + optional CLIP
                                └──── same queries / metrics ────┘
```

## Features

- **Meme-Cap ingestion** — downloads the 6,382 annotated memes (5,823 trainval +
  559 test), converts transparent PNGs to bounded-size JPEGs, deduplicates by SHA-1
  (6,378 unique → 5,820 trainval + 558 test) and writes a versioned manifest.
- **Three comparable strategies** — each reports per-signal score components
  (`clip`, `doc`, `bm25`, `rerank`, `fused`) so a ranking can be explained in the UI.
- **One gallery, one benchmark** — all strategies search identical images; evaluation
  uses the same queries, qrels and top-k for every pipeline.
- **Leakage-aware evaluation** — paraphrased drafts are rejected when they copy an
  8-gram from an annotation; a separate raw-caption set is kept as a labelled
  diagnostic only. qrels are graded (2 = source meme, 1 = near-duplicate intent).
- **Statistical comparison** — paired bootstrap confidence intervals, win/tie/loss,
  and Holm–Bonferroni correction across the three pairwise comparisons.
- **Ablations** — disable prompt ensembling or each B channel; compare pure VLM text
  retrieval against VLM + CLIP.
- **Offline captioning** — Qwen2.5-VL writes structured JSON per image
  (`scene`, `ocr_text`, `meaning`, `metaphors`, `emotions`, `keywords`) to a resumable
  cache. Batched generation (`models.vlm.batch_size`, default 1) cuts the full 6.4k
  run from ~6 s to ~1.5 s per image on a 12 GB GPU; a failed batch is retried
  image-by-image, so one bad file cannot stall the run. The web server never loads
  VLM weights.
- **Runtime settings** — `/settings` edits fusion weights, over-fetch multipliers and
  reranking live; changes persist to `data/settings.json` and are re-applied on
  restart. Build-time options stay read-only.
- **Docker** — CPU or CUDA images, persistent `./data` and model-cache volume.

## Dataset

Meme-Cap provides, for each meme, human-written literal captions, intended-meaning
captions, a category, a title, the source URL and a list of metaphor/meaning pairs.
Annotations are fetched from the project's GitHub raw JSON; images come from the
HuggingFace mirror `hieupth/memecap` (Google Drive fallback / manual zip supported).

After preparation:

| Path | Contents |
| --- | --- |
| `data/images/memecap/*.jpg` | 6,378 JPEGs (max side 768, quality 90) |
| `data/manifest.jsonl` | one normalised record per image |
| `data/gallery.db` | SQLite schema v2: images, annotations, metaphors, FTS5 |
| `data/index/clip/` | shared CLIP image vectors (`image.faiss`, `image_ids.npy`) |
| `data/index/clip_nlp/` | BGE annotation-document vectors |
| `data/vlm/descriptions.jsonl` | resumable VLM output cache |
| `data/index/vlm/` | BGE VLM-document vectors |
| `data/eval/` | queries, graded qrels and the raw-caption diagnostic set |
| `data/reports/` | per-strategy eval JSON, `compare.json`, `compare.html`, `failures.html` |

All of `data/` is gitignored. Cite arXiv:2305.13703 when using Meme-Cap.

## Quick start

### Docker (recommended for serving)

```bash
cp .env.example .env            # optional; edit models/weights if needed
docker compose --profile tools run --rm build python -m tsr prepare --limit 300
docker compose --profile tools run --rm build python -m tsr build --strategy gallery
docker compose --profile tools run --rm build python -m tsr build --strategy clip_nlp
docker compose --profile tools run --rm build python -m tsr make-eval --num 60
docker compose --profile tools run --rm build python -m tsr eval --strategy clip,clip_nlp
docker compose up -d --build    # http://localhost:8000
```

Replace `--limit 300` with no limit for the full gallery. Strategy C additionally
needs offline captioning on a GPU host:

```bash
# GPU host: install the CUDA torch wheel for your driver first
pip install -r requirements-vlm.txt
python -m tsr vlm-caption --limit 20 --out data/vlm_sample.html   # spot-check first
TSR_MODELS__VLM__BATCH_SIZE=8 python -m tsr vlm-caption            # resume-safe, hours
python -m tsr build --strategy vlm
```

Copy `data/vlm/descriptions.jsonl` (or the resulting `data/index/vlm/`) into the
deployment host, then `build --strategy vlm` there — the server only needs the text
encoder, not the 3B VLM. An index covering less than 95% of the gallery is reported
as unavailable (with the exact counts) rather than serving a half-captioned gallery;
finish captioning and rerun `build --strategy vlm --force`.

### Native development

Python 3.10+ with PyTorch/OpenCLIP wheels (verified on Python 3.14 + torch 2.11 on
Windows). Without installing the package, prefix commands with `PYTHONPATH=src`.

```bash
python -m venv .venv && . .venv/Scripts/activate     # Windows
pip install -r requirements-dev.txt
pip install -e .

python -m tsr prepare --limit 300
python -m tsr build --strategy gallery        # SQLite rows + shared CLIP index
python -m tsr build --strategy clip_nlp        # annotation docs + BGE index + BM25
python -m tsr make-eval --num 60
python -m tsr compare --ablations
python -m tsr serve                           # http://localhost:8000
```

The full pipeline:

```bash
python -m tsr prepare                         # download + convert 6,382 images
python -m tsr build --strategy gallery        # CLIP image index
python -m tsr vlm-caption                     # Qwen2.5-VL, resumable
python -m tsr build --strategy all            # A + B + C document indexes
python -m tsr make-eval --sources data/eval/draft_sources.json \
                        --rewrites data/eval/rewrites --force   # frozen 240-query benchmark
python -m tsr eval --strategy all             # reports/eval_<key>.json
python -m tsr compare --ablations             # reports/compare.{json,html}
```

## CLI

| Command | Description |
| --- | --- |
| `tsr prepare [--limit N] [--force] [--source hf\|gdrive]` | Download, extract and normalise Meme-Cap |
| `tsr build --strategy gallery\|clip\|clip_nlp\|vlm\|all` | Build the gallery and strategy indexes |
| `tsr vlm-caption [--limit N] [--out sample.html]` | Offline Qwen2.5-VL descriptions (resumable) |
| `tsr make-eval [--num N] [--seed S] [--force]` | Draft queries, qrels and the diagnostic set |
| `tsr make-eval --sources F --rewrites DIR --force` | Freeze the reviewed benchmark from drafted sources |
| `tsr eval --strategy clip,clip_nlp,vlm` | Metrics + bootstrap CIs per strategy |
| `tsr compare [--ablations] [--boot N]` | Paired comparison report (JSON + HTML) |
| `tsr serve [--host H] [--port P] [--reload]` | Run the FastAPI server |
| `tsr stats [--json]` | Dataset, model and index readiness |
| `tsr thumbnails [--size 480]` | Pre-generate thumbnail cache |
| `tsr clean [--all] [--raw] [--yes]` | Remove caches / generated artifacts |

Add `--config path.yaml` or set `TSR_CONFIG` to layer overrides; `--log-level` sets
verbosity.

## Evaluation

**Set.** `memecap_eval_v1` holds 240 queries drawn only from the 559-image test
split — 80 visual (rewritten `img_captions`), 80 intent (rewritten `meme_captions`)
and 80 metaphor (rewritten metaphor annotations). Queries are written per draft, then
run through the leakage gate; the frozen set keeps every query that clears it and the
build fails rather than silently dropping the rest. Drafts
live in `data/eval/draft_sources.json` and the reviewed rewrites in
`data/eval/rewrites/`; `make-eval --sources ... --rewrites ...` rebuilds the frozen
set deterministically.

**Relevance.** A query is written for one test-split meme. That meme is grade 2.
Grade 1 is assigned by deterministic rules only (near-identical image caption with
token Jaccard ≥ 0.8, identical intended meaning, or a shared non-literal metaphor
that carries the *same meaning* and appears in at most `eval.metaphor_max_df`
images — corpus-common labels such as "a man" are literal depictions, not shared
symbolism, and a shared label alone is not enough). No model-judged expansion is
used, so "looks similar to a model" never becomes ground truth.

**Leakage.** Strategy B indexes human annotations, so evaluation queries must not
copy them. Drafts rewrite the source caption and are rejected if an 8-gram overlaps
any annotation field of the source *or any gallery image*, if the query embeds a
whole short annotation (≥ 4 tokens, which the n-gram rule cannot see), if the query
shares a contiguous run of ≥ `eval.min_copy_tokens` (default 5) tokens with its own
source annotation (the 8-gram rule cannot see a 5-token copy inside a longer
caption), or if token Jaccard with the source is ≥ 0.6 (that rule has no length
floor, so short captions cannot be pasted verbatim). Every verdict is stored in
`metadata.leakage`. `make-eval` also writes `*.raw_captions.json`: raw
captions used as queries, explicitly labelled as a leaky upper-bound diagnostic and
reported separately (`tsr eval --queries data/eval/memecap_eval_v1.raw_captions.json`
writes `reports/eval_<key>_raw_captions.json` and never overwrites the main table).

**Metrics.** P@1/5/10, R@1/5/10, mAP, MRR, nDCG@10 with graded gains, median rank and
misses; grouped by query category (visual / intent / metaphor / independent) and split.
Retrieval depth is `max(top_k, largest relevant set, 100)` so recall@R and mAP are not
truncated by the reported @k cuts. Latency is reported as p50/p95.

**Significance.** `compare` runs identical frozen queries through every ready
strategy, caches each strategy's results, and applies a paired bootstrap
(`eval.bootstrap`, fixed seed) to per-query metric values. Holm–Bonferroni is applied
once across every pairwise × metric test, so the three strategy comparisons form a
single family, and p-values are floored at the bootstrap resolution `1/n_boot`.
`--ablations` adds prompt-ensemble, per-channel B ablations and pure-VLM vs VLM+CLIP.
`failures.html` shows the ten worst queries per strategy with the graded relevant set
and the top-5 retrieved images side by side. If `reports/eval_<key>_raw_captions.json`
exists, the report also carries a clearly separated diagnostic table: raw-caption mAP
next to main-set mAP, so the annotation-overlap effect is visible and cannot be
mistaken for retrieval quality.

Reports are written to `data/reports/`. The `/evaluation` page reads the same
`compare.json`, so displayed numbers always match the CLI report.

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/api/health` | Overall + per-strategy readiness |
| GET | `/api/stats` | Dataset, models, index and strategy stats |
| GET | `/api/strategies` | Strategy metadata and readiness |
| GET | `/api/strategies/{key}/search?q=&k=` | Search one strategy (`clip`, `clip_nlp`, `vlm`) |
| GET | `/api/compare?q=&k=` | Three-column comparison |
| GET | `/api/search?q=&k=&strategy=clip` | Backward-compatible alias |
| GET | `/api/similar/{pk}?strategy=` | Image-to-image over the shared CLIP index |
| GET | `/api/detail/{pk}` | Annotations, metaphors, VLM summary |
| GET | `/api/image/{pk}` · `/api/thumbnail/{pk}` | Image files / cached thumbnails |
| GET | `/api/random?n=` | Random gallery sample |
| GET | `/api/eval/summary` · `/api/eval/report.html` · `/api/eval/failures.html` | Comparison report and per-strategy failure gallery |
| GET | `/api/eval/queries` | Current frozen query set |
| GET/PUT | `/api/settings` | Runtime-editable weights and rerank switches |

Search responses include `score_components`, `img_captions`, `meme_captions`,
`metaphors`, `source_url` and a `vlm` summary. Unknown strategies return 404,
unbuilt strategies 503, and invalid settings 422.

## Configuration

`configs/default.yaml` is the single source of defaults. Environment variables
prefixed `TSR_` override it; nested keys use `__`:

```bash
TSR_APP__PORT=9000
TSR_MODELS__CLIP__NAME=ViT-B-32
TSR_STRATEGIES__CLIP_NLP__WEIGHTS__BM25=0.3
```

Key sections:

- `dataset.*` — archive, annotation URLs, max side, JPEG quality, optional limit.
- `models.clip.*` — OpenCLIP backbone used by the shared image index and strategy A.
- `models.text.*` — BGE encoder for B and C document indexes.
- `models.vlm.*` — Qwen2.5-VL backend/model and offline cache file.
- `index.*` — `flat`, `hnsw`, `ivf` or `pq`; all vectors are L2-normalised.
- `strategies.*` — enabled strategies, B fusion weights/BM25 field weights/rerank,
  C CLIP fusion weight.
- `lexicon` — hand-authored meme phrases expanded into BM25 terms only.
- `eval.*` — set name, query counts per category, top-k, bootstrap resamples, seed.

## Project layout

```
configs/default.yaml       all defaults
src/tsr/
  cli.py                   command-line entry point
  config.py                AttrDict config + path helpers
  memecap.py               download, extract, normalise, manifest
  manifest.py              ManifestRecord / JSONL / hashing
  store.py                 SQLite schema v2 + FTS5 BM25
  index.py                 FAISS index with mandatory id map + meta checks
  pipeline.py              gallery / strategy build pipelines
  encoders/{clip,text,vlm}.py
  nlp/{analyze,fusion,rerank}.py
  strategies/{base,clip_only,clip_nlp,vlm}.py
  service.py               lazy StrategyManager
  evaluate.py              metrics, leakage checks, bootstrap
  compare.py               comparison + ablations + HTML report
  api.py                   FastAPI routes
  schemas.py               response models
web/                       index, search, compare, evaluation pages
tests/                     unit tests (heavy deps are importorskip-guarded)
```

## Testing & linting

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src pytest
PYTHONPATH=src ruff check src tests
```

Tests use fake encoders/indexes; they never download models or need the gallery.
FAISS tests skip automatically when the wheel is unavailable.

## Limitations

- Strategy B depends on human annotations; it has no cold-start path for new memes.
  Strategy C is the annotation-free approximation, and A is the zero-cost baseline.
- CLIP reads the literal image, not the joke; OCR overlaid text is only available to C.
- VLM captioning is a one-off GPU cost: ~1.5 s/image with a 3B model at batch 8 on a
  12 GB card (~2 h 35 min for the full gallery); batch 1 costs roughly 4–10 s/image.
- With 6.4k images, absolute metric differences can be small; use the confidence
  intervals and per-category breakdowns rather than a single mAP number.
- Meme-Cap is a research dataset; respect its licence and do not build face search.

## Deployment

See [`DEPLOY.md`](DEPLOY.md) for serving with Docker, direct IP or a domain with
automatic HTTPS.

## Report

[`docs/report.md`](docs/report.md) is the written report: protocol, full results with
confidence intervals, pairwise tests, ablations, the leakage diagnostic and the
knowledge-cost comparison. Every number in it is reproducible from the commands above.

## Schematic

[`docs/system-schematic.pdf`](docs/system-schematic.pdf) is a five-page diagram set —
system architecture, the three strategies side by side (build and query pipeline for
each, plus a comparison table), offline build flow, online query flow and the evaluation
pipeline. It is generated from the repository, not drawn by hand:

```bash
python docs/make_schematic.py     # writes docs/system-schematic.pdf
```

[`docs/system-detailed.pdf`](docs/system-detailed.pdf) is the long-form version: 36 landscape
pages in which the same five diagrams each get a page of their own at the end of their
chapter, alongside prose, tables and worked examples covering the dataset, the three strategies, the build and
query pipelines, the evaluation protocol, the results, and how to operate and reproduce the
system. Every number in it is read from the repository at build time, and Appendix B maps each
figure back to the file it came from:

```bash
python docs/make_detailed.py     # writes docs/system-detailed.pdf
```

[`docs/system-walkthrough.md`](docs/system-walkthrough.md) is the prose-only version of the
same material: one section per diagram page, explaining each box, what it writes and why the
guard exists.
