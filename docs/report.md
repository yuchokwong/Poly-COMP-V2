# Meme Retrieval: CLIP vs CLIP + NLP vs VLM (COMP5523)

This report compares three ways of representing memes for text-to-image retrieval on
the same 6,382-image Meme-Cap gallery:

- **A — `clip`**: OpenCLIP ViT-L-14 text and image embeddings only. No annotations,
  no OCR, no extra knowledge.
- **B — `clip_nlp`**: CLIP fused with a dense index over the human annotation
  documents (scene / meaning / metaphor / title) and field-weighted BM25 over the same
  fields, with an optional cross-encoder rerank stage.
- **C — `vlm`**: an offline Qwen2.5-VL-3B pass writes a structured description of every
  image (literal scene, overlaid text, intended meaning, metaphors, emotions,
  keywords); retrieval matches that text with BGE and optionally fuses CLIP.

All three share one image table, one CLIP image index, one frozen query set and one
metric implementation, so differences come from the representation and not from the
data or the protocol.

## 1. Dataset and preprocessing

Meme-Cap provides, per meme, human-written literal captions, intended-meaning
captions, a category, a title, a source URL and metaphor/meaning pairs. Annotations
come from the project's GitHub JSON; images come from the HuggingFace mirror
`hieupth/memecap` (Google Drive fallback and manual zip are supported).

Preparation converts transparent PNGs onto white, bounds the longest side at 768 px,
writes JPEG quality 90, deduplicates by SHA-1, and records a normalised JSONL manifest.
6,382 annotated images yield 6,378 unique images after four SHA-1 duplicates are
dropped; the gallery holds 5,820 trainval and 558 test images.
The SQLite schema (v2) stores images, annotations, metaphors, VLM documents, strategy
documents and an FTS5 index. `images.id` is the single key shared by SQLite, FAISS row
ids and the id map, and is assigned once, in `image_id` order.

## 2. Architecture invariants

- FAISS indexes store L2-normalised vectors, so inner product equals cosine similarity,
  and every index carries a mandatory id map — there is no `arange` fallback.
- Each index writes `meta.json` (schema, model, dimension, count, dataset hash); loading
  validates it and refuses mismatched or incomplete artifacts.
- Strategy rebuilds only rewrite strategy documents and vector files; they never
  renumber images, so evaluation qrels stay valid across rebuilds.
- Strategy keys are a closed set: unknown keys return 404, known-but-unbuilt return 503.
  Strategy C additionally reports unavailable when its index covers less than 95% of the
  gallery, so an interrupted captioning run cannot masquerade as a built strategy.

## 3. Retrieval details

**A.** `q = L2norm(clip_text(query))`; inner product against the shared image index.
Optional prompt ensembling is an ablation.

**B.** Each image becomes four fields — `scene`, `meaning`, `metaphor`, `title` — stored
both as a dense document (BGE-small-en-v1.5, CLS pooling) and in an FTS5 table scored
with column weights `(scene 1.0, meaning 1.6, metaphor 1.8, title 0.6)`. At query time
three candidate lists are fetched, min-max normalised and fused with weights
`clip 0.5 / doc 0.3 / bm25 0.2` (renormalised when a signal is empty), then optionally
reranked by a cross-encoder over the top 60. Query expansion from a small hand-written
meme lexicon feeds BM25 only — never CLIP or BGE — so expansion cannot drift the
semantic signal. Metaphor rows whose meaning is "not related to the meme context"
(literal elements, 1,968 rows in the 6,378-image gallery) are excluded from documents.

**C.** Captioning runs offline and resumably: one JSON object per image, stored in
`data/vlm/descriptions.jsonl` keyed by image, SHA-1, backend, model and prompt version.
Batched generation (8 images per call) runs at roughly 1.5 s/image on a 12 GB
RTX 4070 SUPER (2 h 35 min for the full gallery); a failed batch is retried image by
image so one bad file cannot stall the run. The serving process never imports the VLM.
Retrieval embeds the query with BGE, searches the description index, and fuses with
CLIP at weight 0.2 (pure-VLM and CLIP 0.35 are ablations).

## 4. Evaluation protocol

**Query set.** `memecap_eval_v1` has 240 queries drawn only from the 559-image test
split: 80 visual (rewritten literal captions), 80 intent (rewritten intended meanings),
80 metaphor (rewritten metaphor annotations). Each query is written for one source meme,
in natural language, and must clear the leakage gate below before it is frozen. The
frozen build also checks that every source is still in the test split and that its
annotation snapshot is unchanged, and it aborts instead of writing a partial set if
any reviewed query is rejected.

**Leakage control.** Because B indexes human annotations, any query that copies them
would inflate B for the wrong reason. Every query passes a gate: it is rejected if an
8-gram overlaps any annotation field of the source or of any gallery image, if it
embeds a whole short annotation (≥ 4 tokens, which no 8-gram can catch), if it shares
a contiguous run of ≥ 5 tokens with its own source annotation (a 5-token copy inside a
longer caption slips under the 8-gram rule), or if its token Jaccard with the source
caption is ≥ 0.6. The Jaccard rule has no length floor, so short captions cannot be
pasted verbatim either. All verdicts are stored in `metadata.leakage`, and `compare`
refuses to publish a query set in which any query failed the gate. The raw-caption
queries are kept as a *separate, labelled* diagnostic and are never mixed into the
main table.

**Relevance.** Grade 2 is the source meme. Grade 1 is assigned only by deterministic
rules: token Jaccard ≥ 0.8 with the source image caption, identical normalised intended
meaning, or a shared non-literal metaphor label that also carries the same meaning and
has document frequency at most 10. A shared label alone is not a shared metaphor —
"a man" or "having" appear in unrelated memes, and matching only the label produced
false positives in earlier drafts. No model-judged expansion is used. On the frozen
set, the metaphor and intent queries therefore end up with a single grade-2 image each,
while visual queries retain graded near-duplicates.

**Metrics.** P@1/5/10, R@1/5/10, mAP, MRR, nDCG@10 with graded gains, median rank and
misses, grouped by query category and split. Retrieval depth is
`max(top_k, largest relevant set, 100)`, so recall and mAP are not truncated by the
reported @k cuts.

**Statistics.** `compare` runs the identical frozen queries through every ready strategy
with a fixed seed, applies a paired bootstrap (1,000 resamples) to per-query metric
values, and applies Holm–Bonferroni once across every pairwise × metric test, treating
the three pairs as one family. p-values are floored at the bootstrap resolution
`1/n_boot`.

## 5. Results

All 240 frozen queries, three strategies, identical gallery and retrieval depth
(`compare.json`, 1,000 bootstrap resamples, seed 42).

| Strategy | mAP [95% CI] | MRR [95% CI] | nDCG@10 [95% CI] | P@1 | R@10 | median rank | misses | p50 / p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A `clip` | **0.4395** [0.385, 0.498] | 0.4461 [0.391, 0.499] | 0.4744 [0.418, 0.525] | 0.3583 | 0.6074 | 2 | 46 | 14.4 / 20.8 ms |
| B `clip_nlp` | **0.7608** [0.713, 0.808] | 0.7647 [0.718, 0.813] | 0.7804 [0.737, 0.822] | 0.7000 | 0.8702 | 1 | 13 | 225.6 / 285.9 ms |
| C `vlm` | **0.4519** [0.399, 0.507] | 0.4632 [0.410, 0.517] | 0.4881 [0.433, 0.544] | 0.3833 | 0.6259 | 2 | 38 | 35.3 / 41.6 ms |

| Category (mAP) | A `clip` | B `clip_nlp` | C `vlm` |
| --- | --- | --- | --- |
| visual (80 queries) | 0.3153 | 0.6313 | 0.2469 |
| intent (80) | 0.4444 | 0.7389 | 0.5390 |
| metaphor (80) | 0.5588 | 0.9121 | 0.5697 |

**Pairwise tests** (paired bootstrap over per-query values, Holm–Bonferroni across the
three pairs × five metrics; p-values floored at 1/1,000 = 0.001):

- **B − A:** mAP **+0.3213** [0.2702, 0.3708], p_holm = 0.015; significant on every
  metric (mAP wins/ties/losses 140/94/6).
- **B − C:** mAP **+0.3089** [0.2550, 0.3632], p_holm = 0.015; significant on every
  metric (128/93/19 for B).
- **C − A:** mAP **+0.0124** [−0.0382, 0.0641], p_holm = 1.000; not significant on any
  metric (84/82/74) — the two annotation-free strategies are indistinguishable.

**Reading.** The annotation-backed strategy wins by a wide, statistically robust margin,
and the category breakdown says why: B's gains are largest exactly where intent lives
(metaphor 0.912 vs 0.559/0.570, intent 0.739 vs 0.444/0.539) because its documents
contain the human-written meaning and metaphor pairs. C converts a model's paraphrase of
the image into a moderate advantage on intent and metaphor over A, but it *loses* on the
visual category (0.247 vs 0.315): a 3B VLM describes the gist and overlaid text, not
every visible detail, so a query that enumerates objects ("two dogs, the left one
standing") is served better by pixel-level CLIP alignment. The net C − A difference is
within noise, so at this scale a VLM is not a drop-in substitute for human annotation —
it is closer to a cheap, annotation-free approximation of the intent signal.

Latency separates the cost side: A answers in 14 ms, C in 35 ms, B in 226 ms (its
cross-encoder rerank over 60 candidates dominates), all far below interactive budgets.

## 6. Ablations

`compare --ablations` reruns the identical queries against seven single-change variants.

| Variant | mAP | MRR | nDCG@10 | p50 |
| --- | --- | --- | --- | --- |
| A `clip` (reference) | 0.4395 | 0.4461 | 0.4744 | 14.4 ms |
| A + prompt ensemble | 0.4415 | 0.4535 | 0.4795 | 7.1 ms |
| B `clip_nlp` (reference) | 0.7608 | 0.7647 | 0.7804 | 225.6 ms |
| B − BM25 | 0.7040 | 0.7130 | 0.7286 | 198.5 ms |
| B − dense doc | 0.7043 | 0.7119 | 0.7224 | 199.7 ms |
| B − CLIP | **0.7802** | 0.7829 | 0.7992 | 228.2 ms |
| B − reranker | 0.6887 | 0.6926 | 0.7181 | 43.0 ms |
| C `vlm` (reference) | 0.4519 | 0.4632 | 0.4881 | 35.3 ms |
| C pure VLM (CLIP 0) | 0.3922 | 0.4032 | 0.4266 | 8.1 ms |
| C + CLIP 0.35 | **0.4720** | 0.4837 | 0.5095 | 18.0 ms |

**B's channels.** Removing either text channel costs about 0.056 mAP (BM25 0.7040,
dense doc 0.7043), and removing the reranker costs 0.072 (0.6887) while cutting latency
five-fold — reranking is the single most valuable component of B. The surprise is
CLIP: dropping the visual signal *raises* mAP from 0.7608 to 0.7802. With human
annotations indexed, the fused CLIP weight (0.5) dilutes a stronger text signal; the
current weight was chosen before evaluation and is the clearest tuning opportunity in
the system. Prompt ensembling on A is neutral (+0.002 mAP), so the single template is
kept for speed and simplicity.

**C's fusion.** Pure VLM text retrieval is 0.392; adding CLIP at 0.2 reaches 0.452 and
at 0.35 reaches 0.472. Unlike B, C genuinely needs the visual channel: the description
misses visible detail and the pixels supply it. The two strategies therefore use CLIP
in opposite directions — as noise for B, as signal for C — which is itself evidence
that the annotation and VLM documents cover different parts of the query space.

## 7. Leakage diagnostic and the cost of knowledge

The raw-caption diagnostic set uses the untouched human annotations as queries. It is
leaky by construction (B indexes those annotations), so it is reported separately and
never enters the main table.

| Strategy | Main-set mAP | Raw-caption mAP | Δ |
| --- | --- | --- | --- |
| A `clip` | 0.4395 | 0.3546 | −0.0849 |
| B `clip_nlp` | 0.7608 | 0.8998 | **+0.1390** |
| C `vlm` | 0.4519 | 0.2591 | −0.1928 |

Only B improves when queries copy annotation wording — the +0.139 mAP is the size of
the annotation-copy effect, and it is large enough to manufacture a different ranking if
it were allowed into the main table. A and C fall, and C falls furthest: raw captions
are human phrasing, while C's documents are model paraphrases, so the wording overlap
that helps B actively hurts C. This is also why the frozen set uses rewritten queries
and why `compare` refuses to publish a set in which any query failed the leakage gate.

| Strategy | Extra knowledge | One-off cost | Generalises to new images |
| --- | --- | --- | --- |
| A `clip` | none (pixels only) | minutes | yes |
| B `clip_nlp` | human captions and metaphors | human annotation of 6.4k images | no — new memes arrive without annotations |
| C `vlm` | VLM-generated descriptions | ~2.6 h GPU for 6.4k images (batch 8) | yes |

The three strategies are therefore three points on a knowledge-cost curve, not three
attempts at the same thing: B is the ceiling when annotations exist, C approximates
annotations with a model, A is the zero-cost baseline.

## 8. Failure analysis

`data/reports/failures.html` shows the ten worst queries per strategy with the graded
relevant set and the top-5 retrieved images. The failures are dominated by *visual*
queries in all three strategies, and they share one shape: the query names fine-grained
structure that a single embedding cannot carry — two-panel layouts ("above … and then
below …"), a specific expression ("biting his lip, lifting his eyebrows"), a costume
("full Batman costumes"). CLIP matches the scene gist and misses composition; the VLM
description compresses the image to "a man reacts", so it loses the same details for a
different reason; B's one-sentence annotation documents have the same limit, which is
why its 13 misses are still mostly visual despite its large overall lead. A's metaphor
and intent queries do comparatively well (0.559 / 0.444) because the rewritten query
usually still names the literal scene, while B's failures there are nearly gone
(0.912 metaphor mAP): with the intended meaning in the index, wording overlap between
query and annotation is enough.

## 9. Reproducing

```bash
pip install -r requirements-dev.txt && pip install -e .
python -m tsr prepare
python -m tsr build --strategy gallery
TSR_MODELS__VLM__BATCH_SIZE=8 python -m tsr vlm-caption
python -m tsr build --strategy all
python -m tsr make-eval --sources data/eval/draft_sources.json \
                        --rewrites data/eval/rewrites --force
python -m tsr eval --strategy all
python -m tsr compare --ablations
PYTHONPATH=src pytest && PYTHONPATH=src ruff check src tests
```

## 10. Limitations

- B has no cold-start path: new memes arrive without annotations.
- CLIP reads the literal image, not the joke; overlaid text is only available to C
  (via the VLM's OCR field).
- At 6.4k images, differences should be read with the confidence intervals and the
  per-category table, not a single mAP.
- The VLM description is itself a lossy annotation; C's quality is bounded by the 3B
  model and by caption errors. The first full pass left 8 of 6,378 images (0.13%) with
  malformed JSON (unescaped quotes, a missing comma, or a `max_new_tokens` cut-off); a
  repair pass recovered all of them, and the two truncated descriptions keep only
  their scene field.
- Meme-Cap is a research dataset; use is limited to research and no face search.

## 11. Use of AI tools

The three pipelines, the evaluation tooling and this report were written by the author
with the assistance of an AI coding assistant (Claude). Query drafts for the benchmark
were generated by an LLM from annotation sources and then passed through the automated
leakage gate; the frozen set, its qrels and every reported number are reproducible with
the commands above. No LLM judged relevance at evaluation time.
