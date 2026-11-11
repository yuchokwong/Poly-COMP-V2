# System walkthrough — the schematic in words

[`system-schematic.pdf`](system-schematic.pdf) shows the system as five diagrams. This
document says the same things in prose. Sections 1–5 correspond to pages 1–5 of the PDF:
each explains every box on that page, what it writes, and why the box exists in that form
rather than a simpler one.

Nothing here is aspirational. Every count, weight and latency is read from `configs/default.yaml`,
`data/index/*/meta.json` or `data/reports/compare.json`; §7 lists which figure comes from
which file.

---

## 0. The system in one paragraph

Meme-Cap gives 6,382 memes, each with human-written literal captions, intended-meaning
captions and a title, and most with a list of metaphor/meaning pairs. The system turns that into a
searchable gallery of 6,378 unique images and answers natural-language queries with three
different retrieval strategies over **the same images, the same frozen query set and the
same metric code**. The only thing that varies between strategies is what text — human
annotation, model-generated description, or none at all — represents an image at query
time. That is what makes the resulting comparison (mAP 0.44 / 0.76 / 0.45) an attributable
difference rather than three unrelated numbers.

---

## 1. Architecture (page 1)

### 1.1 The offline band — how the gallery is made

**Meme-Cap source.** Annotations are fetched as two JSON files from the project's GitHub
repository (trainval + test); images come from the HuggingFace mirror `hieupth/memecap`
(1.7 GB zip, Google Drive as fallback, resumable). The zip is streamed to disk with a
`.part` suffix, so an interrupted download resumes instead of restarting.

**Normalise.** Every PNG is converted to RGB — transparent pixels are composited onto
white, because a meme's white background is part of the image and a transparent crop would
turn black in some viewers. The longest side is bounded at 768 px (never upscaled) and the
result is written as JPEG quality 90. The bound is a compromise: it keeps overlaid text
legible for the VLM's OCR field while producing a 492 MiB gallery instead of a 1.7 GB one.

**De-duplicate.** A SHA-1 of the source PNG identifies byte-identical files; four
duplicates are dropped, leaving 6,378 images from 6,382 annotations. A perceptual dHash is
also stored on every record, but it is *not* used to merge images — it is kept so that
near-duplicate analysis can be done later without re-decoding the gallery.

**manifest.jsonl.** One JSON record per image, in `image_id` order: the identifier, the
source filename, the split (`trainval` or `test`), the relative path, width, height, SHA-1,
perceptual hash, title, source URL, post id, the raw caption lists, the metaphor pairs and
the category. `manifest.meta.json` sits next to it with the record count, the annotation
counts and the manifest hash; the schema version lives in each index's own `meta.json`.
Everything downstream reads the manifest, never the raw zip.

The split is what makes evaluation honest: 5,820 trainval and 558 test images. Queries are
drawn **only** from the test split, so no strategy can have been tuned on a query's source
image.

### 1.2 The artefacts panel — what exists on disk

| Artefact | Written by | Read by | Holds |
| --- | --- | --- | --- |
| `gallery.db` | `build --strategy gallery` | every strategy, every request | SQLite schema v2: images, annotations, metaphors, `vlm_docs`, `strategy_docs`, `nlp_docs`, the `meta` key/value table + the FTS5 mirror |
| `index/clip/` | same command | A, B, C | 6,378 × 768-d image vectors, the id map, `meta.json` |
| `index/clip_nlp/` | `build --strategy clip_nlp` | B | 384-d BGE vectors over the annotation documents |
| `index/vlm/` | `build --strategy vlm` | C | 384-d BGE vectors over the VLM descriptions |
| `vlm/descriptions.jsonl` | `vlm-caption` | `vlm-caption` itself, as its resume cache — `build --strategy vlm` reads `vlm_docs` in SQLite, not this file | the resumable caption cache — an append-only JSONL log with one object per captioning attempt, so a retried image can carry an earlier `status='error'` line as well as its later `status='ok'` line |

The panel is drawn between the two bands because it *is* the interface: the offline band
only ever writes into it, the online band only ever reads from it, and the two never run
in the same process. That is why the server can be started while captioning is still
running on another machine.

### 1.3 The online band — how one request is served

**Browser.** Eight static pages (`/`, `/clip`, `/clip-nlp`, `/vlm`, `/compare`,
`/evaluation`, `/tuning`, `/settings`) served from `web/`. They are plain HTML + vanilla JS;
all state comes from the JSON API, so nothing has to be rebuilt when the data changes.

**FastAPI.** 19 routes under `/api` — 15 JSON endpoints plus two image and two HTML
responses. The important ones are `/api/strategies/{key}/search`,
`/api/compare`, `/api/eval/summary` and `GET`/`PUT /api/settings`. `k` is limited to 1…100
and defaults to 12 on `/api/strategies/{key}/search`, `/api/search` and `/api/similar/{pk}`;
`/api/compare` limits `k` to 1…50 and defaults to 10. Unknown strategy keys raise before any
model is touched.

**StrategyManager.** Owns the SQLite store, creates the three strategy objects, and loads
nothing else at startup. Each strategy loads itself lazily through `ensure_loaded()`, which
both `search()` and `stats()` call — so `/api/health` and `/api/stats` validate every index
on page load, and encoder weights wait for the first actual search. The strategy records
either `ready` or a `load_error` string; a failure is reported through `/api/health`, never
raised into the page. A process-wide `RLock` serialises searches, because the encoders and
FAISS indexes are shared mutable state.

**The three strategies.** A/B/C, drawn with their real weights. Note that all three read
the *same* `index/clip/` image vectors: A uses them alone, B fuses them with annotation
signals, C fuses them with VLM-description signals. That shared backbone is the reason the
comparison holds the visual representation constant.

**SearchResponse.** The payload every column returns: rank, fused score, per-signal
`score_components`, the image's human captions and metaphors, the VLM summary, and the
latency of that call. The per-signal components are what lets the UI show *why* an image
ranked where it did instead of a single opaque number.

### 1.4 The four invariants

The strip at the bottom of page 1 carries two invariants. Two further guards are enforced
by the build but not drawn there. All four prevent a specific, silent failure.

1. **`images.id` is assigned once, in `image_id` order, and never renumbered.** It is the
   only key shared by SQLite, the FAISS row ids and `image_ids.npy`. Strategy rebuilds
   rewrite `strategy_docs` and vector files only — they never touch `images` — so a
   rebuild cannot invalidate the qrels that point at image ids.
2. **Every index has an id map. There is no `arange` fallback.** Without a file, a search
   would return row positions 0…n and silently answer with the wrong images; the loader
   raises instead. *(Not drawn on page 1 — enforced in `index.py`.)*
3. **Every index writes `meta.json`** (schema version, kind, model, dimension, count,
   dataset hash, config hash, build time), and loading validates it. A vector file from a
   different model or a different manifest is refused rather than served.
4. **A strategy build compares a content hash of the stored annotations against the
   manifest.** Comparing ids alone would accept edited text — the ids still line up, but
   the vectors describe the old wording. *(Not drawn on page 1 — it is the "Verify" step
   on page 3.)*

---

## 2. The three strategies (page 2)

### 2.1 A — `clip`

**Build.** One pass over the gallery through the CLIP image tower (OpenCLIP ViT-L-14,
`laion2b_s32b_b82k`), batch 64, L2-normalised, written to `index/clip/`. No annotation text
feeds the CLIP embedding pass or the query path; the build that produces `index/clip/`
also writes the manifest's annotations into `gallery.db` and hashes them to detect stale
rows, but strategy A never uses them.

**Query.** The query goes through the CLIP text tower (77 tokens), and the resulting 768-d
vector is compared with every image vector by inner product — which equals cosine
similarity because both sides are L2-normalised. It over-fetches `max(2k, k)` and then
drops every candidate whose similarity is ≤ 0, keeping the top *k*.

**What it cannot see.** The joke, and any text drawn on the image. A query that describes
the visible scene works; a query that describes the meme's meaning only works if the
meaning happens to be visible. This is the zero-knowledge baseline: minutes of CPU, no
human labour, and it generalises to any new image.

### 2.2 B — `clip_nlp`

**Build.** Each image becomes four fields — `scene` (literal captions), `meaning`
(intended meaning), `metaphor` (`"X means Y"` joined by `; `) and `title`. Metaphor rows
whose meaning is `"not related to the meme context"` describe literal elements, not
symbolism; there are 1,968 of them in the 6,378-image gallery and they are dropped before
anything is indexed. The four fields are written twice: as one dense document (`scene: …
\nmeaning: …\nmetaphor: …\ntitle: …`) embedded with BGE-small-en-v1.5 (CLS pooling, 384-d),
and as four columns of an FTS5 table for BM25.

**Query.** Three candidate lists are fetched with `fetch = max(5k, 100)` each:

- **CLIP**, as in A — the visual signal;
- **dense doc**, the query encoded by BGE with its query instruction, against
  `index/clip_nlp/`;
- **BM25**, FTS5 over the four columns with weights `scene 1.0 / meaning 1.6 / metaphor 1.8
  / title 0.6` — the intended meaning and the metaphor are what a retrieval query usually
  expresses, so they count for more than a scene description.

A hand-written meme lexicon (`"distracted boyfriend"` → couple, jealousy, distraction,
ignoring) adds expansion terms, but **only to the BM25 query**, at weight 0.5 and merged by
taking the maximum per image. CLIP and BGE always see the original sentence, so an
expansion that guesses wrong cannot drag the semantic signals with it.

The three lists are min-max normalised over their union of candidates (a missing candidate
scores 0), combined as `0.5·clip + 0.3·doc + 0.2·bm25`, and — because weights are
renormalised after empty or disabled signals are dropped — the strategy degrades to a
two-signal or one-signal ranker instead of returning zeros when, say, CLIP is unavailable.

Finally the top 60 are reranked by a cross-encoder (`bge-reranker-base`) that reads the
query and the document together: `fused ← 0.7·fused + 0.3·rerank`. Reranking is the single
most expensive and most valuable part of B (removing it costs 0.072 mAP but cuts latency
five-fold).

### 2.3 C — `vlm`

**Build (offline, GPU).** Qwen2.5-VL-3B-Instruct reads each image once and returns a JSON
object with `scene`, `ocr_text`, `meaning`, `metaphors`, `emotions` and `keywords`. The
prompt forbids identifying real people and forbids using outside knowledge of a meme
template unless the picture supports it. Malformed JSON (unescaped quotes, a missing comma,
a `max_new_tokens` cut-off) is repaired by locating the object span and re-parsing; the
first full pass still left 8 of 6,378 images (0.13%) that local repair could not fix, and
re-running those images recovered all of them. Each result is appended to
`vlm/descriptions.jsonl` and fsynced, keyed by `(image_pk, image_sha1)` and filtered by
`backend`/`model`/`prompt_version`, so an interrupted run resumes exactly where it stopped
and a different generator's rows are never reused.

Those fields are flattened into `doc_text` (`scene: … / overlaid text: … / meaning: … /
metaphors: … / emotions: … / keywords: …`) and embedded with the same BGE model as B.

**Query.** The query is encoded with BGE and searched against `index/vlm/`, over-fetching
`max(3k, 100)` candidates before fusing, then fused with CLIP at `0.8·doc + 0.2·clip`. If
CLIP is unavailable the doc weight becomes 1.0. No reranker by default. **The 3B model is never loaded by the server** — inference cost was
paid once, offline; serving needs only a 384-d text encoder.

**Two gates specific to C.** The index must cover at least 95% of the gallery (otherwise a
half-finished captioning run would serve as a "ready" strategy), and `meta.json`'s recorded
`backend`/`model`/`prompt_version` must match the current configuration (otherwise
descriptions from two different generators would be mixed without anyone noticing).

### 2.4 What actually differs

Every strategy ends in the same place — a ranked list of the same images, with
per-signal components. What differs is the knowledge each one is allowed to use:

| | A | B | C |
| --- | --- | --- | --- |
| Text per image | none | human captions and metaphors | model-written description |
| Signals | 1 | 3 + rerank | 2 |
| One-off cost | minutes of CPU | human annotation of 6.4k images | 2 h 35 min of GPU |
| Works on unseen memes | yes | **no** — new memes arrive unannotated | yes |

Two ablation results make the difference concrete rather than rhetorical:

- **B is better without CLIP** (0.7608 → 0.7802). Once human annotations are indexed, the
  0.5 CLIP weight dilutes a stronger text signal. The weight was fixed before evaluation;
  this is the clearest tuning opportunity in the system.
- **C is much worse without CLIP** (0.4519 → 0.3922). A VLM description compresses the
  image to its gist and drops visible detail, which the pixels then have to supply.

So the same CLIP channel is noise for one strategy and signal for the other. That does not
make VLM text a copy of annotation text: C recovers part of the intent channel (intent
0.539 and metaphor 0.570, both above A) but loses visual detail (0.247, below A's 0.315),
which is why it lands beside A rather than beside B. A 3B VLM is a cheap approximation of
the knowledge a human annotator supplies, not a substitute for it.

---

## 3. Build flow (page 3)

Five commands, in this order. Each row on the page is one command: what it does, what it
writes, and the rule that stops a bad build from being written.

### 3.1 `tsr prepare` — once, network required (~15 min)

Downloads the annotation JSON and the image archive (resumable), extracts only
`memecap/images/*.png` and `memecap/caption/*.json`, normalises (RGB on white, 768 px
longest side, JPEG q90), de-duplicates by SHA-1, and writes `manifest.jsonl` +
`manifest.meta.json`.

*Guards.* An existing manifest is reused unless `--force`; extraction is skipped when the
recorded zip size still matches, so re-running after a failure does not re-extract 6,416
files.

### 3.2 `tsr build --strategy gallery` — once (~10 min on CPU)

Inserts the 6,378 image rows in `image_id` order inside one transaction, then encodes every
image with CLIP into `index/clip/`.

*Guards.* The build asserts that ids are contiguous, that the vector count matches the row
count, and that a content hash of the stored annotations matches the manifest. Any of these
failing aborts before an index is written — the alternative is an index whose row *i* is a
different image than id *i*, which no downstream check would catch.

### 3.3 `tsr build --strategy clip_nlp` (~5 min on CPU)

Builds the four annotation fields, writes `nlp_docs` (with the FTS5 triggers) and
`strategy_docs`, and encodes the dense documents into `index/clip_nlp/`.

*Guard.* The 1,968 literal-metaphor rows are excluded; without that rule, "a man" would be
indexed as this meme's symbolism.

### 3.4 `tsr vlm-caption` — GPU host, hours (~2 h 35 min)

Batched generation (`models.vlm.batch_size`; the shipped default is 1, the full run used 8
images per call, 448 new tokens), JSON repair, upsert into `vlm_docs` and append to the
JSONL cache with an fsync per line. A failed batch is retried image by image, so one
unreadable file cannot stall the run. The reported run finished with 0 outstanding parse
errors and **6,378 / 6,378** images described.

*Guard.* The resume key is image + SHA-1; `backend`, `model` and `prompt_version` are stored
on every row and reject rows from a different generator — an image that changed on disk, or
a prompt revision, invalidates exactly the affected rows.

### 3.5 `tsr build --strategy vlm` (~5 min on CPU)

Filters `vlm_docs` to `status = ok` and the current provenance triple, encodes `doc_text`
with BGE, and records the triple in `meta.json`.

*Guard.* A different generator, or coverage under 95%, makes C read as unavailable rather
than serving a partially captioned gallery.

`--strategy all` runs clip → clip_nlp → vlm and skips C with a warning when
`models.vlm.backend` is `none` or when its build raises a `RuntimeError` — for example no
`vlm_docs` match the configured backend/model/prompt_version, or the gallery no longer
matches the manifest. B and C never touch `images`: they rewrite only their own rows and
vector files.

`python -m tsr stats --json` prints readiness, row and vector counts and the model string
for every stage — the quickest way to see which index is built and what provenance it
records.

---

## 4. Query flow (page 4)

Nine steps for one request, drawn as a sequence diagram.

1. The page calls `GET /api/strategies/{key}/search?q=…&k=…`.
2. The route calls `StrategyManager.search(key, query, k)`.
3. The manager resolves the strategy (`UnknownStrategy` → 404) and calls
   `ensure_loaded()`.
4. The strategy encodes the query — CLIP text tower for A, BGE with the query instruction
   for B and C.
5. It searches its index(es), fetching `max(2k, k)` for A and `max(mult·k, 100)` for B
   (mult 5) and C (mult 3).
6. The index returns ids and scores (inner product on L2-normalised vectors).
7. The strategy returns a `SearchOutput` with hits, per-signal components and its own
   latency.
8. The manager enriches the hits from `gallery.db` — captions, metaphors, title, source
   URL, VLM summary — because an index only knows ids and scores.
9. The route returns a `SearchResponse`; the page renders cards, score bars and a lightbox.

### 4.1 Failure semantics

| Status | Meaning |
| --- | --- |
| 404 | unknown strategy key — the set is closed (`clip`, `clip_nlp`, `vlm`) |
| 503 | known but not ready: index missing, stale, or failing a coverage/provenance gate |
| 422 | invalid payload (empty query, out-of-range setting) |

`/api/compare` never fails as a whole: a column that cannot answer is returned as
`{ready: false, error: …}` and the other columns still serve. That is what lets the
comparison page work while strategy C is still being captioned.

### 4.2 What loads when

Startup loads configuration, opens SQLite and constructs the three strategy objects — no
model weights, no FAISS files. The first `stats()` or `search()` call on a strategy loads
its indexes (the home page's `/api/health` and `/api/stats` calls do the former); the first
search loads its encoder and, for B, the cross-encoder reranker. The load is guarded by
`ensure_loaded()`, so the cost is paid once and never repeated. The VLM stack is never loaded by the serving process at all: captioning is an
offline job, and the serving image deliberately does not ship those extras.

### 4.3 The latency budget

Measured p50 / p95 over the 240 frozen queries: **A 14.4 / 20.8 ms**, **C 35.3 / 41.6 ms**,
**B 225.6 / 285.9 ms**. B's extra time is almost entirely the cross-encoder over 60
candidates; turning it off drops latency to roughly 43 ms at a cost of 0.072 mAP. All three
are far inside an interactive budget, which is why the default configuration keeps the
reranker on.

---

## 5. Evaluation (page 5)

### 5.1 Build the benchmark

240 queries are drafted for 80 visual, 80 intent and 80 metaphor sources, all drawn from
the 558-image test split. Each draft is rewritten by hand, then must pass the leakage gate
before it is frozen:

- an 8-gram from the query overlapping **any** annotation field of the source **or of any
  gallery image**;
- the query embedding a whole short annotation (≥ 4 tokens) — no 8-gram can catch a
  3-word caption pasted verbatim;
- a contiguous run of ≥ 5 tokens copied from the source caption — a short copy inside a
  longer sentence slips under the 8-gram rule;
- token Jaccard ≥ 0.6 with the source caption — no length floor, so short captions cannot
  be pasted either.

The freeze also verifies that every source is still in the test split and that its
annotation snapshot is unchanged, and it aborts rather than writing a partial set if any
reviewed query is rejected. Verdicts are stored in `metadata.leakage`.

### 5.2 Measure

All three strategies run the same queries against the same gallery with
`depth = max(top_k, largest relevant set, 100)`, so recall and mAP are never truncated by
the reported @k cuts. Every strategy is scored on P@1/5/10, R@1/5/10, mAP, MRR, nDCG@10,
median rank and misses; §5.4 tabulates the headline subset.

Relevance is graded: grade 2 is the source meme, and grade 1 is assigned only by
deterministic rules (near-identical image caption with Jaccard ≥ 0.8, identical intended
meaning, or a shared non-literal metaphor that carries the same meaning and appears in at
most 10 images). No model judges relevance.

### 5.3 Compare

A paired bootstrap (1,000 resamples, fixed seed 42) over per-query metric values gives the
confidence intervals; Holm–Bonferroni is applied once across the three pairwise × five
metric tests, so the three comparisons form a single family, and p-values are floored at
the bootstrap resolution 0.001.

`--ablations` re-runs the benchmark with seven single changes, each isolating one design
decision: prompt ensembling for A; B without BM25 (0.704), without the dense channel
(0.704), without CLIP (0.780) or without the reranker (0.689); and C as pure VLM (0.392) or
with the CLIP weight raised to 0.35 (0.472).

The run writes `data/reports/compare.json`, renders it to `compare.html` and
`failures.html`, and serves the same JSON through `/api/eval/summary`, which the
`/evaluation` page reads — so the CLI and the page cannot disagree.

### 5.4 Results

| | mAP [95% CI] | MRR | nDCG@10 | P@1 | R@10 | median rank | misses | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A `clip` | 0.4395 [0.385, 0.498] | 0.4461 | 0.4744 | 0.358 | 0.607 | 2 | 46 | 14 ms |
| B `clip_nlp` | **0.7608** [0.713, 0.808] | 0.7647 | 0.7804 | 0.700 | 0.870 | 1 | 13 | 226 ms |
| C `vlm` | 0.4519 [0.399, 0.507] | 0.4632 | 0.4881 | 0.383 | 0.626 | 2 | 38 | 35 ms |

B − A is +0.321 mAP [0.270, 0.371] and B − C is +0.309 [0.255, 0.363], both significant on
every metric after correction (p_holm = 0.015 for both). C − A is +0.012 [−0.038, 0.064] and
not significant on any metric: the two annotation-free strategies are indistinguishable.

The category split explains the ranking. B improves every category — visual 0.631 vs 0.315
(A) / 0.247 (C), intent 0.739 vs 0.444 / 0.539, metaphor 0.912 vs 0.559 / 0.570 — with its
largest gain over A on metaphor (+0.353) and its smallest on intent (+0.295), because its
documents contain the human-written meaning and the metaphor pairs. C gains on intent and
metaphor over A (0.539 / 0.570 vs 0.444 / 0.559) but **loses on the visual category** (0.247
vs 0.315): a 3B model describes the gist and the overlaid text, not every visible detail, so
"two dogs, the left one standing" is better served by pixel-level alignment.

### 5.5 The leakage diagnostic

Running the untouched human annotations as queries is leaky by construction — B indexes
those annotations — so it is reported separately and never enters the main table. It
measures the size of the annotation-copy effect: B +0.139 mAP, A −0.085, C −0.193. Only B
improves when queries copy annotation wording, and the gain is large enough to reorder the
strategies on its own; that is precisely why the frozen set uses rewritten queries and why
the comparison refuses to publish a set in which any query failed the gate.

### 5.6 Failure analysis

`failures.html` shows the ten worst queries per strategy with the graded relevant set and
the top-5 retrieved images. A's and C's ten worst queries are all visual — fine-grained
structure that a single embedding cannot carry, such as two-panel layouts, a specific
expression or a costume. CLIP matches the scene gist and misses composition; the VLM
compresses the image to "a man reacts" and loses the same details for a different reason.
B's ten worst split evenly, five visual and five intent queries about abstract meaning, and
its 13 misses overall spread across the categories — 7 intent, 5 visual and 1 metaphor — so
intent, not visual, is its largest gap despite the large overall lead.

---

## 6. Operating it

**Build-time choices** (models, index type, dataset, prompt version) live in
`configs/default.yaml` and require a rebuild. **Ranking knobs** — fusion weights, the
BM25 weight, over-fetch multipliers, expansion on/off and its weight, rerank top-N and
alpha, C's CLIP weight, prompt ensembling, default `k` — are editable at runtime from
`/settings` or `PUT /api/settings`, persisted to `data/settings.json` and re-applied at
startup. Edits take effect on the next search — except `app.default_top_k`, which
`create_app` reads once into the search route's query default, so changing it has no effect
until the process restarts.

**Rebuild rules.** Rebuild an index when the dataset, the model or the document text
changes; then restart the API. Readiness is a load-time snapshot, so a running process keeps
serving the indexes it loaded. Stats are not: strategies A and B re-read `meta.json` and
the live row count on every request, so after a rebuild they describe the newer files on
disk even though search still uses the old loaded index. Only strategy C reports the
artifacts it actually loaded.

---

## 7. Where every figure comes from

| Figure | Source |
| --- | --- |
| 6,382 annotations → 6,378 images, 4 duplicates | `data/manifest.meta.json` |
| 5,820 / 558 split | `gallery.db` (`images.split`), `data/manifest.jsonl` |
| 492 MiB gallery, 768 px, quality 90 | `configs/default.yaml` · `dataset.*`, `data/images/` |
| Model names and dimensions | `data/index/*/meta.json` |
| Fusion weights, BM25 column weights, fetch multipliers, rerank settings | `configs/default.yaml` · `strategies.*` |
| Build guards (id contiguity, content hash, provenance, 95% coverage) | `src/tsr/pipeline.py`, `src/tsr/strategies/*.py` |
| Latencies, mAP, CIs, Holm-adjusted p-values, ablations | `data/reports/compare.json` |
| Query set and qrels | `data/eval/memecap_eval_v1.{json,qrels.jsonl}` |
| Captioning throughput, parse failures | `data/vlm/descriptions.jsonl`, `vlm_docs` |

To regenerate the PDF after a rebuild:

```bash
python docs/make_schematic.py            # writes docs/system-schematic.pdf
```
