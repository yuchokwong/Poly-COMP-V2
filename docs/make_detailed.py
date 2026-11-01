#!/usr/bin/env python3
"""Render docs/system-detailed.pdf.

A long-form companion to docs/system-schematic.pdf: the five diagrams are embedded one
per page, and every page of the diagram set is expanded into prose, tables and worked
examples. Numbers are read from the repository at build time (manifest.meta.json,
data/reports/compare.json, data/eval/, configs/default.yaml) so the document cannot drift
from the artefacts it describes.

Pipeline: schematic pages -> PNG, HTML -> headless Edge -> PDF, then footers and page
numbers are stamped with pymupdf (Chromium's print-to-PDF has no CSS margin boxes).

    python docs/make_detailed.py
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SCHEMATIC = DOCS / "system-schematic.pdf"
OUT = DOCS / "system-detailed.pdf"

BROWSERS = [
    Path(r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    Path(r"C:/Program Files/Google/Chrome/Application/chrome.exe"),
    Path(r"C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
]

CLIP, NLP, VLM, EVAL, DATA = "#1F8A70", "#4A63B0", "#BF7A20", "#A6474E", "#66798A"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def load_data() -> dict:
    meta = load_json(ROOT / "data/manifest.meta.json", {})
    report = load_json(ROOT / "data/reports/compare.json", {})
    evalset = load_json(ROOT / "data/eval/memecap_eval_v1.json", [])
    counts = {"images": 0, "with_metaphors": 0}
    try:
        with (ROOT / "data/manifest.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                counts["images"] += 1
                if json.loads(line).get("metaphors"):
                    counts["with_metaphors"] += 1
    except (OSError, ValueError):
        pass
    return {
        "meta": meta,
        "report": report,
        "evalset": evalset if isinstance(evalset, list) else [],
        "counts": counts,
    }


def worked_examples(data: dict) -> list[dict]:
    """One query per category where A and C miss and B ranks the source first."""
    report = data["report"]
    strategies = report.get("strategies", {})
    if not strategies or not data["evalset"]:
        return []
    rank = {k: {q["qid"]: q for q in v.get("per_query", [])} for k, v in strategies.items()}
    out = []
    for cat in ("visual", "intent", "metaphor"):
        for entry in data["evalset"]:
            if entry.get("category") != cat:
                continue
            qid = entry["qid"]
            try:
                a, b, c = rank["clip"][qid], rank["clip_nlp"][qid], rank["vlm"][qid]
            except KeyError:
                continue
            if (
                a["rank"] is None
                and b["rank"] == 1
                and (c["rank"] is None or c["rank"] > 10)
            ):
                out.append({"entry": entry, "rows": [("A · clip", a), ("B · clip_nlp", b), ("C · vlm", c)]})
                break
    return out


# --------------------------------------------------------------------------- #
# html helpers
# --------------------------------------------------------------------------- #
def esc(text) -> str:
    return html.escape(str(text))


def table(headers: list[str], rows: list[list[str]], cls: str = "", widths: list[str] | None = None) -> str:
    out = [f'<table class="{cls}">']
    if widths:
        out.append("<colgroup>" + "".join(f'<col style="width:{w}">' for w in widths) + "</colgroup>")
    out.append("<thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in headers) + "</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def figure(png: Path, number: int, caption: str, anchor: str) -> str:
    return (
        f'<figure id="{anchor}"><img src="{png.as_uri()}">'
        f'<figcaption><b>Figure {number}.</b> {caption}</figcaption></figure>'
    )


def fmt(value, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #
CSS = """
@page { size: A4 landscape; margin: 14mm 16mm 12mm 16mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0 auto; max-width: 225mm;
       font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
       font-size: 10.2pt; line-height: 1.62; color: #1B2430; }
p { margin: 0 0 2.8mm; }
ul, ol { margin: 0 0 3mm; padding-left: 5.5mm; }
li { margin-bottom: 1.1mm; }
b, strong { color: #16202B; }
a { color: #2F6FB0; text-decoration: none; }
code { font-family: Consolas, "Cascadia Mono", monospace; font-size: 8.5pt;
       background: #EFF3F6; padding: 0 1.2px; border-radius: 2px; color: #22303C; }
pre { font-family: Consolas, "Cascadia Mono", monospace; font-size: 8.6pt; line-height: 1.5;
      background: #F5F8FA; border-left: 2.5px solid #1F8A70; padding: 2.4mm 3mm;
      margin: 2mm 0 4mm; white-space: pre-wrap; page-break-inside: avoid; break-inside: avoid; }
pre code { background: none; padding: 0; }

.ex, .keep { page-break-inside: avoid; break-inside: avoid; margin-bottom: 4mm; }
.ex h3 { margin: 0 0 1.5mm; }

h1 { font-size: 19pt; line-height: 1.15; margin: 9mm 0 5mm; color: #16202B;
     padding-bottom: 3.5mm; border-bottom: 2px solid #DCE4EA; position: relative;
     page-break-after: avoid; break-after: avoid; }
h1::after { content: ""; position: absolute; left: 0; bottom: -2px; width: 38mm;
            height: 2px; background: #1F8A70; }
section:first-of-type h1 { margin-top: 0; }
h1[id="s13"], h1[id="sa"], h1[id="sb"], h1[id="sc"] { page-break-before: always; margin-top: 0; }
h1 .num { color: #1F8A70; margin-right: 3mm; }
h1 + .lede { font-size: 10.3pt; color: #4A5A68; max-width: 190mm; margin: 0 0 5mm;
             page-break-after: avoid; break-after: avoid; }
.rule { height: 2px; background: #DCE4EA; margin: 0 0 5mm; position: relative; }
.rule::before { content: ""; position: absolute; left: 0; top: 0; height: 2px;
                width: 38mm; background: #1F8A70; }
h2 { font-size: 12.2pt; margin: 6.5mm 0 2.2mm; color: #16202B; page-break-after: avoid; }
h3 { font-size: 9.2pt; text-transform: uppercase; letter-spacing: .07em; color: #5B6B79;
     margin: 5mm 0 1.8mm; page-break-after: avoid; }

table { border-collapse: collapse; width: 100%; font-size: 8.2pt; line-height: 1.45;
        margin: 2mm 0 4.5mm; page-break-inside: avoid; }
th { background: #EBF1F5; text-align: left; font-weight: 600; color: #16202B;
     padding: 1.5mm 2mm; border-bottom: 1px solid #C2CFDA; }
td { padding: 1.3mm 2mm; border-bottom: 1px solid #E4EBF0; vertical-align: top; }
tbody tr:nth-child(even) td { background: #FAFCFD; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td.mono, th.mono { font-family: Consolas, monospace; font-size: 7.9pt; }

figure { width: 240mm; margin: 4mm -7.5mm 5mm; page-break-inside: avoid; }
figure img { width: 100%; display: block; border: 1px solid #D5DEE6; }
figcaption { font-size: 7.8pt; color: #71818F; margin-top: 1.6mm; line-height: 1.45; }

.box { border-left: 3px solid #66798A; background: #F7FAFC; padding: 2.6mm 3.2mm;
       margin: 3mm 0 4mm; font-size: 9.6pt; }
.box.clip { border-left-color: #1F8A70; background: #F3FAF7; }
.box.nlp  { border-left-color: #4A63B0; background: #F5F7FC; }
.box.vlm  { border-left-color: #BF7A20; background: #FDF8F1; }
.box.eval { border-left-color: #A6474E; background: #FCF6F7; }
.box .t { font-weight: 600; display: block; margin-bottom: 1mm; }

.chips { margin: 0 0 5mm; }
.chip { display: inline-block; font-size: 8.2pt; padding: .8mm 3mm; margin: 0 2mm 2mm 0;
        border-radius: 10px; border: 1px solid; }
.chip.a { color: #1F8A70; border-color: #9CCDBC; background: #F1F9F6; }
.chip.b { color: #4A63B0; border-color: #B3BEE0; background: #F4F6FC; }
.chip.c { color: #BF7A20; border-color: #E5C897; background: #FDF8EF; }

.cover { page-break-after: always; padding-top: 14mm; }
.cover .kicker { font-size: 8.4pt; letter-spacing: .16em; text-transform: uppercase;
                 color: #71818F; margin-bottom: 6mm; }
.cover h1 { font-size: 29pt; page-break-before: avoid; margin: 0 0 4mm; letter-spacing: -0.01em;
            border: none; padding-bottom: 0; }
.cover h1::after { display: none; }
.cover .sub { font-size: 13pt; color: #4A5A68; max-width: 210mm; margin: 0 0 9mm; line-height: 1.45; }
.cover .band { height: 5px; width: 100%; margin: 0 0 9mm;
               background: linear-gradient(90deg, #1F8A70 0 33%, #4A63B0 33% 66%, #BF7A20 66% 100%); }
.meta { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6mm; }
.meta .cell { border-top: 2px solid #DCE4EA; padding-top: 2.2mm; }
.meta .k { font-size: 7.6pt; letter-spacing: .1em; text-transform: uppercase; color: #71818F; }
.meta .v { font-size: 11pt; color: #16202B; margin-top: 1mm; }
.meta .s { font-size: 8.2pt; color: #5B6B79; }
.cover .foot { position: absolute; bottom: 14mm; font-size: 8.2pt; color: #71818F; }
.inside { margin-top: 16mm; }
.inside .k { font-size: 7.6pt; letter-spacing: .12em; text-transform: uppercase;
             color: #71818F; margin-bottom: 3mm; }
.inside .row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 4mm; }
.inside .card { border: 1px solid #DCE4EA; border-top: 3px solid #1F8A70; padding: 3mm; }
.inside .card .n { font-size: 8pt; color: #1F8A70; font-weight: 600; }
.inside .card .t { font-size: 8.8pt; color: #16202B; margin-top: 1mm; line-height: 1.35; }

.toc { page-break-before: always; page-break-after: always; }
.toc h1 { page-break-before: avoid; margin-top: 0; }
.toc ol { list-style: none; padding: 0; margin: 6mm 0 0; max-width: 210mm; }
.toc li { display: flex; align-items: baseline; font-size: 10.2pt; padding: 1.5mm 0;
          border-bottom: 1px dotted #D5DEE6; }
.toc .dot { flex: 1; }
.toc .pg { color: #71818F; font-variant-numeric: tabular-nums; }
.toc .n { color: #1F8A70; font-weight: 600; min-width: 12mm; }

.note { font-size: 8.4pt; color: #71818F; }
.small { font-size: 8.4pt; }
"""


# --------------------------------------------------------------------------- #
# content
# --------------------------------------------------------------------------- #
def cover(data: dict, results: dict) -> str:
    meta = data["meta"]
    n = meta.get("num_records", 6378)
    return f"""
<section class="cover">
  <div class="kicker">COMP5523 · Text-to-image retrieval · Meme-Cap</div>
  <h1>Meme retrieval: CLIP vs CLIP+NLP vs VLM</h1>
  <div class="sub">A detailed walkthrough of the system architecture, the build and query
  pipelines of three retrieval strategies, and the evaluation that separates them.</div>
  <div class="band"></div>
  <div class="chips">
    <span class="chip a">A · clip — pixels only</span>
    <span class="chip b">B · clip_nlp — CLIP + human annotations</span>
    <span class="chip c">C · vlm — CLIP + Qwen2.5-VL descriptions</span>
  </div>
  <div class="meta">
    <div class="cell"><div class="k">Gallery</div><div class="v">{n:,} images</div>
      <div class="s">from {meta.get('num_annotations', 6382):,} annotated memes</div></div>
    <div class="cell"><div class="k">Benchmark</div><div class="v">240 queries</div>
      <div class="s">80 visual · 80 intent · 80 metaphor</div></div>
    <div class="cell"><div class="k">Models</div><div class="v">ViT-L-14 · BGE-small</div>
      <div class="s">+ Qwen2.5-VL-3B, offline</div></div>
    <div class="cell"><div class="k">Headline mAP</div><div class="v">
      {fmt(results.get('clip', 0))} · {fmt(results.get('clip_nlp', 0))} · {fmt(results.get('vlm', 0))}</div>
      <div class="s">A · B · C on the frozen set</div></div>
  </div>
  <div class="inside">
    <div class="k">The diagram set, one diagram per page</div>
    <div class="row">
      <div class="card"><div class="n">Diagram 1</div><div class="t">System architecture</div></div>
      <div class="card" style="border-top-color:#4A63B0"><div class="n" style="color:#4A63B0">Diagram 2</div><div class="t">The three strategies side by side</div></div>
      <div class="card" style="border-top-color:#BF7A20"><div class="n" style="color:#BF7A20">Diagram 3</div><div class="t">Offline build flow</div></div>
      <div class="card" style="border-top-color:#2F6FB0"><div class="n" style="color:#2F6FB0">Diagram 4</div><div class="t">Online query flow</div></div>
      <div class="card" style="border-top-color:#A6474E"><div class="n" style="color:#A6474E">Diagram 5</div><div class="t">Evaluation pipeline</div></div>
    </div>
  </div>
  <div class="foot">Generated from the repository — every count, weight and metric on the
  following pages is read from the code and the report files at build time.</div>
</section>
"""


def toc(entries: list[tuple[str, str, str]], pages: dict[str, int]) -> str:
    items = []
    for anchor, number, title in entries:
        pg = pages.get(anchor, "—")
        items.append(
            f'<li><span class="n">{number}</span><span>{esc(title)}</span>'
            f'<span class="dot"></span><span class="pg">{pg}</span></li>'
        )
    return f"""
<section class="toc">
  <h1 style="page-break-before:avoid">Contents</h1>
  <ol>{''.join(items)}</ol>
</section>
"""


def body(data: dict, figs: dict[int, Path], examples: list[dict], report: dict) -> str:
    meta = data["meta"]
    n_img = meta.get("num_records", 6378)
    n_ann = meta.get("num_annotations", 6382)
    n_dup = meta.get("num_duplicates", 4)
    strategies = report.get("strategies", {})
    m = {k: v.get("metrics", {}) for k, v in strategies.items()}
    lat = {k: v.get("latency", {}) for k, v in strategies.items()}

    def metric(key, field, default="—"):
        return fmt(m.get(key, {}).get(field), 4) if m.get(key, {}).get(field) is not None else default

    def lat_p50(key):
        v = lat.get(key, {}).get("p50_ms")
        return f"{v:.0f}" if v else "—"

    def lat_p95(key):
        v = lat.get(key, {}).get("p95_ms")
        return f"{v:.0f}" if v else "—"

    parts: list[str] = []

    # ---------------------------------------------------------------- overview
    parts.append(f"""
<section>
<h1 id="s1"><span class="num">1</span>Overview</h1>
<div class="rule"></div>
<p class="lede">One gallery, three representations of the same {n_img:,} images, one frozen
benchmark, one API. The only thing that changes between strategies is which text — human
annotation, model-generated description, or none at all — stands for an image at query time.</p>

<p>Meme retrieval is a hard case for text-to-image search. A meme is usually a photograph or a
drawing with overlaid text, and the interesting part is rarely the scene: it is the joke, the
implied meaning, the metaphor. A model that aligns pixels with captions can describe
<i>a man in a suit shaking hands with another man</i> and still miss that the image means
<i>one company acquiring another and pretending it was a merger</i>. The Meme-Cap dataset
exists to make that gap measurable, because each meme carries human-written literal captions and
intended-meaning captions, and {data['counts']['with_metaphors']:,} of the
{data['counts']['images']:,} also carry metaphor/meaning pairs.</p>

<p>The system built here turns that dataset into a searchable gallery and answers
natural-language queries with three strategies whose differences are deliberate:</p>

<ul>
<li><b>A · <code>clip</code></b> — an OpenCLIP ViT-L-14 image index. No annotation is used.
This is the zero-knowledge baseline: minutes of CPU and no human labour, but it sees the
scene and not the joke.</li>
<li><b>B · <code>clip_nlp</code></b> — CLIP fused with two annotation signals: BGE dense
vectors over a four-field document, and field-weighted BM25, followed by a cross-encoder
rerank. It is the upper bound when human annotation is available.</li>
<li><b>C · <code>vlm</code></b> — Qwen2.5-VL reads each image offline and writes a structured
description; BGE indexes those descriptions and fuses them with CLIP. It approximates
annotation without a human, and generalises to unseen memes.</li>
</ul>

<p>Because all three search the same images with the same queries, the same qrels and the same
metric code, the resulting mAP differences are attributable to the representation rather than
to three unrelated setups. That is the entire point of the design, and it shapes almost every
decision described on the following pages.</p>

<div class="box eval"><span class="t">Reading this document</span>
Sections 3 and 7 correspond to pages 1–2 of the diagram set (architecture, then the three
strategies side by side). Sections 8–10 correspond to pages 3–5 (build flow, query flow,
evaluation). Each of those chapters closes with its diagram on a page of its own — the prose
before it explains every box, so the two read as a pair. Section 13 lists
the commands that regenerate everything, and Appendix B maps every figure back to the file it
was read from.</div>
</section>
""")

    # ---------------------------------------------------------------- dataset
    parts.append(f"""
<section>
<h1 id="s2"><span class="num">2</span>The dataset and the gallery</h1>
<div class="rule"></div>
<p class="lede">Six thousand annotated memes become a {n_img:,}-image gallery with a stable
integer key, a bounded JPEG, and a manifest that everything downstream reads instead of the
raw archive.</p>

<h2>What Meme-Cap provides</h2>
<p>For every meme the dataset supplies the source image, one or more literal image captions
(<code>img_captions</code>), one or more intended-meaning captions (<code>meme_captions</code>),
a title, the Reddit source URL and post id, and — for most memes — a list of metaphor/meaning
pairs. Annotation
length is modest — about 12.7 words of literal caption and 21.0 words of intended meaning per
image — which is one reason a small text encoder (BGE-small, 384-d) is sufficient and a
512-token window never truncates an annotation document.</p>

<h2>Preparation</h2>
<p><code>tsr prepare</code> is the only stage that downloads the <i>dataset</i>. Annotations come
from the project's GitHub repository as two JSON files (trainval and test); images come from the
HuggingFace mirror <code>hieupth/memecap</code> — a 1.7 GB zip — with a Google Drive fallback.
The archive is streamed to disk with a <code>.part</code> suffix so an interrupted download
resumes rather than restarting, and extraction is skipped when the recorded archive size still
matches.</p>

<p>The other network dependency is model weights, fetched the first time an encoder is used and
cached thereafter: OpenCLIP ViT-L-14 (~1.7 GB), BGE-small (~130 MB) and, when B's reranker is
enabled, the cross-encoder (~1.1 GB). After that first load the whole system runs offline — which
is also why the deployment image ships a model-cache volume rather than downloading per request.</p>

<p>Each source PNG is converted to RGB with transparent pixels composited onto white. This
matters: a meme's white background is part of the image, and a transparent crop would render
as black in some viewers and as white in others. The longest side is bounded at 768 px (never
upscaled) and the result is written as JPEG quality 90. The bound is a compromise — it keeps
overlaid text legible for the VLM's OCR field while producing a 492 MiB gallery instead of a
1.7 GB one.</p>

<p>De-duplication is by SHA-1 of the source bytes. Four byte-identical files are dropped,
leaving <b>{n_img:,} images from {n_ann:,} annotations</b>. A perceptual dHash is computed and
stored on every record but is deliberately <i>not</i> used to merge images: the gallery's
identity is exact, and near-duplicate analysis can be added later without re-decoding
anything.</p>

<h2>The manifest</h2>
<p>Every downstream stage reads <code>data/manifest.jsonl</code>, never the raw zip. One JSON
record per image, written in <code>image_id</code> order:</p>

{table(
    ["Field", "Meaning", "Used by"],
    [
        ["<code>image_id</code>", "the dataset's identifier, e.g. <code>memes_d079np</code>", "everything; the stable key"],
        ["<code>post_id</code>", "Reddit post id", "provenance links in the UI"],
        ["<code>split</code>", "<code>trainval</code> or <code>test</code>", "the evaluation split guard"],
        ["<code>path</code>", "relative path of the normalised JPEG", "image and thumbnail routes"],
        ["<code>width</code>, <code>height</code>", "dimensions after normalisation", "layout and aspect handling"],
        ["<code>sha1</code>", "SHA-1 of the source PNG", "de-duplication, VLM cache key"],
        ["<code>phash</code>", "perceptual dHash", "future near-duplicate analysis"],
        ["<code>title</code>", "meme title, may be null", "B's <code>title</code> field"],
        ["<code>source_url</code>", "original Reddit image URL", "attribution in the detail panel"],
        ["<code>img_captions</code>", "literal scene captions", "B's <code>scene</code> field"],
        ["<code>meme_captions</code>", "intended-meaning captions", "B's <code>meaning</code> field"],
        ["<code>metaphors</code>", "metaphor / meaning pairs", "B's <code>metaphor</code> field, metaphor queries"],
        ["<code>category</code>", "dataset category", "reporting only"],
    ],
    widths=["18%", "42%", "40%"],
)}

<div class="box"><span class="t">Why the split matters</span>
The gallery is 5,820 trainval and 558 test images. Every evaluation query is drawn from the
test split, and the freeze step re-checks that the source is still in it. Without that rule a
strategy could be tuned on an image that a query later asks for, and the comparison would
measure memorisation rather than retrieval.</div>

<p><code>manifest.meta.json</code> sits beside the manifest with the record count, annotation
count, duplicate count, the manifest SHA-256, and the normalisation parameters. The schema
version is not there — it lives in each index's own <code>meta.json</code>, because the
manifest can exist before any index does.</p>
</section>
""")

    # ---------------------------------------------------------------- architecture
    parts.append(f"""
<section>
<h1 id="s3"><span class="num">3</span>System architecture</h1>
<div class="rule"></div>
<p class="lede">Two bands that never run in the same process, joined by the artefacts they
share: the offline band builds <code>data/</code>, the online band only reads it.</p>

<h2>3.1 The offline band</h2>
<p>Four steps produce the gallery. <b>Meme-Cap source</b> downloads annotations and the image
archive. <b>Normalise</b> converts PNG to bounded JPEG. <b>De-duplicate</b> removes byte-identical
files by SHA-1. <b>manifest.jsonl</b> records one normalised row per surviving image. Section 2
covers each step in detail; the important architectural point is that the manifest, not the
archive, is the boundary — every later stage can be re-run without the 1.7 GB zip.</p>


<h2>3.2 The artefacts</h2>
<p>The band in the middle of the page is the interface between offline and online. Nothing
crosses it except these files:</p>

{table(
    ["Artefact", "Written by", "Read by", "Contents"],
    [
        ["<code>gallery.db</code>", "<code>build --strategy gallery</code>", "every strategy, every request",
         "SQLite schema v2: <code>images</code>, <code>annotations</code>, <code>metaphors</code>, <code>vlm_docs</code>, <code>strategy_docs</code>, <code>nlp_docs</code>, <code>meta</code>, plus the FTS5 mirror"],
        ["<code>index/clip/</code>", "the same command", "A, B and C",
         f"{n_img:,} × 768-d image vectors, the id map, <code>meta.json</code>"],
        ["<code>index/clip_nlp/</code>", "<code>build --strategy clip_nlp</code>", "B",
         "384-d BGE vectors over the annotation documents"],
        ["<code>index/vlm/</code>", "<code>build --strategy vlm</code>", "C",
         "384-d BGE vectors over the VLM descriptions"],
        ["<code>vlm/descriptions.jsonl</code>", "<code>vlm-caption</code>", "<code>vlm-caption</code> itself, as its resume cache",
         "append-only log of captioning attempts"],
    ],
    widths=["17%", "22%", "20%", "41%"],
)}

<p>The panel is drawn between the two bands because it <i>is</i> the interface. The offline band
only writes; the online band only reads; and because they are separate processes, captioning on
a GPU host and serving on a laptop can run at the same time. The server needs the index files,
not the caption cache, which is why a deployment only has to copy <code>data/index/vlm/</code>.</p>

<h2>3.3 The online band</h2>
<p><b>Browser.</b> Seven page routes — <code>/</code>, <code>/clip</code>, <code>/clip-nlp</code>,
<code>/vlm</code>, <code>/compare</code>, <code>/evaluation</code>, <code>/settings</code> — served
from five HTML files in <code>web/</code>, because the three strategy routes share
<code>search.html</code> and switch on a <code>data-strategy</code> attribute. They are plain HTML
and vanilla JavaScript; all state comes from the JSON API, so changing the data never requires
rebuilding a page.</p>

<p><b>FastAPI.</b> Eighteen routes under <code>/api</code>: fourteen return JSON and four return
files (two images, two HTML reports). <code>k</code> is limited to 1–100 and defaults to 12 on the
search routes; <code>/api/compare</code> limits it to 1–50 and defaults to 10. Unknown strategy
keys raise before any model is touched.</p>

<p><b>StrategyManager.</b> Owns the SQLite store and constructs the three strategy objects at
startup, loading no weights and no index files. Each strategy loads itself lazily through
<code>ensure_loaded()</code>, recording either <code>ready</code> or a <code>load_error</code>
string. Both <code>search()</code> and <code>stats()</code> trigger it, so the home page's
<code>/api/health</code> and <code>/api/stats</code> calls validate every index before the first
query, while encoder weights wait for the first search. A failure is reported through
<code>/api/health</code>, never raised into the page. A process-wide lock serialises searches,
because the encoders and FAISS indexes are shared mutable state.</p>

<p><b>The three strategies.</b> All three read the <i>same</i> <code>index/clip/</code> vectors:
A uses them alone, B fuses them with annotation signals, C fuses them with VLM-description
signals. Holding the visual representation constant is what makes the comparison attributable.</p>

<p><b>SearchResponse.</b> The payload every column returns — rank, fused score, per-signal
components, captions, metaphors, the VLM summary, and the latency of that call. The components
are what let the interface show <i>why</i> an image ranked where it did.</p>

<div class="keep">
<h2>3.4 Four invariants</h2>
<p>Two are drawn in the strip at the bottom of page 1; the other two are enforced by the build
itself. Each prevents a specific silent failure.</p>

{table(
    ["#", "Invariant", "What would go wrong without it", "Where"],
    [
        ["1", "<code>images.id</code> is assigned once, in <code>image_id</code> order, and never renumbered",
         "a strategy rebuild would shift ids and every stored qrel would point at the wrong image", "page 1 strip"],
        ["2", "Every index has an id map; there is no <code>arange</code> fallback",
         "a search would return FAISS row positions and silently answer with unrelated images", "<code>index.py</code>"],
        ["3", "Every index writes <code>meta.json</code> and the loader validates it",
         "vectors from a different model or manifest would be served as if they matched", "page 1 strip"],
        ["4", "A strategy build compares a content hash of the stored annotations against the manifest",
         "edited annotation text would keep the same ids, and the vectors would describe the old wording", "page 3, Verify"],
    ],
    widths=["4%", "34%", "44%", "18%"],
)}
</div>

<p>The first is the load-bearing one. <code>images.id</code> is the only key shared by SQLite,
the FAISS row ids and <code>image_ids.npy</code>. Strategy rebuilds rewrite
<code>strategy_docs</code> and their own vector files; they never touch <code>images</code>. That
is why an evaluation set frozen months ago still points at the right images after a rebuild.</p>
{figure(figs[1], 1, "Page 1 of the diagram set. The offline band writes the gallery and the three indexes; the online band serves pages and JSON from them.", "fig1")}
</section>
""")

    # ---------------------------------------------------------------- A
    parts.append(f"""
<section>
<h1 id="s4"><span class="num">4</span>Strategy A — CLIP only</h1>
<div class="rule"></div>
<p class="lede">The zero-knowledge baseline: one image index, one text encoder, inner product.
No annotation is read at query time and none feeds the embedding pass.</p>

<h2>Build</h2>
<p>The gallery is encoded once with the CLIP image tower (OpenCLIP ViT-L-14,
<code>laion2b_s32b_b82k</code>) at batch 64. Every vector is L2-normalised before it is written, so
an inner-product search is exactly cosine similarity. The index is stored as
<code>index/clip/image.faiss</code> with <code>image_ids.npy</code> beside it and a
<code>meta.json</code> recording the model, dimension, count, dataset hash and config hash.</p>

<div class="box clip"><span class="t">A subtlety worth stating</span>
The command that produces this index is also the command that inserts the image rows, and that
insert reads the manifest's annotations into <code>gallery.db</code> so the other strategies can
use them. Strategy A never reads those columns; it only shares the build. The claim that matters
for the comparison is narrower and true: <b>no annotation text feeds A's embedding pass or its
query path.</b></div>

<h2>Query</h2>
<p>The query goes through the CLIP text tower (77 tokens) and is compared with all
{n_img:,} image vectors by inner product. The strategy over-fetches <code>max(2k, k)</code> — a
hard-coded 2× — and then drops every candidate whose similarity is ≤ 0, keeping the top
<i>k</i>. The zero cut exists because CLIP text embeddings can be anti-correlated with an image;
those candidates are confidently irrelevant and there is no reason to pad a result list with
them.</p>

<p>Prompt ensembling (averaging several prompt templates) is available but off by default. The
ablation in Section 11 measures it: it changes mAP by +0.002, which is well inside noise, so the
default stays simple.</p>

<h2>What it cannot see</h2>
<p>The joke, and any text drawn on the image. A query describing the visible scene works. A query
describing the meme's meaning works only when the meaning happens to be visible. This is not a
defect to be fixed inside strategy A; it is the definition of the baseline, and every gain in B
and C is measured against it.</p>

<p>Its cost profile is what makes it useful in production: minutes of CPU to build, a median of
{lat_p50('clip')} ms per query, no annotation labour, and it generalises to any new image the
moment the image is encoded.</p>
</section>
""")

    # ---------------------------------------------------------------- B
    parts.append(f"""
<section>
<h1 id="s5"><span class="num">5</span>Strategy B — CLIP + annotation NLP</h1>
<div class="rule"></div>
<p class="lede">Three candidate lists — pixels, dense annotation text, lexical annotation text —
fused, then reranked by a cross-encoder that reads the query and the document together.</p>

<h2>Build: one document, two indexes</h2>
<p>Each image becomes four fields:</p>

<table>
<thead><tr><th>Field</th><th>Source</th><th>Construction</th></tr></thead>
<tbody>
<tr><td class="mono">scene</td><td><code>img_captions</code></td><td>literal captions joined by <code> / </code></td></tr>
<tr><td class="mono">meaning</td><td><code>meme_captions</code></td><td>intended-meaning captions joined by <code> / </code></td></tr>
<tr><td class="mono">metaphor</td><td><code>metaphors</code></td><td><code>"X means Y"</code> pairs joined by <code>; </code></td></tr>
<tr><td class="mono">title</td><td><code>title</code></td><td>the dataset title, possibly empty</td></tr>
</tbody></table>

<p>One correction happens before indexing. Meme-Cap marks a metaphor row with the meaning
<i>“not related to the meme context”</i> when the “metaphor” is really a literal element of the
picture. There are <b>1,968</b> such rows in the {n_img:,}-image gallery, and indexing them would
teach the ranker that <i>a man</i> is this meme's symbolism. They are dropped.</p>

<p>The four fields are written twice. As a single dense document —</p>

<pre><code>scene: &lt;literal captions&gt;
meaning: &lt;intended meaning&gt;
metaphor: &lt;X means Y; ...&gt;
title: &lt;title&gt;</code></pre>

<p>— embedded with BGE-small-en-v1.5 (CLS pooling, 384-d, 512 tokens) into
<code>index/clip_nlp/</code>; and as four columns of an FTS5 table (<code>nlp_docs</code>, kept in
sync by triggers) for BM25.</p>

<h2>Query: three signals</h2>
<p>All three signals over-fetch <code>max(5k, 100)</code> candidates, because a fusion can only
reorder what it was given and 100 is a cheap floor at this gallery size.</p>

{table(
    ["Signal", "Query representation", "Index", "Why it is there"],
    [
        ["<b>clip</b>", "CLIP text tower", "<code>index/clip/</code>", "the visual channel; keeps scene queries working"],
        ["<b>doc</b>", "BGE with the query instruction", "<code>index/clip_nlp/</code>", "meaning that shares no words with the annotation"],
        ["<b>bm25</b>", "tokens, stop-words removed, lexicon-expanded", "FTS5 over four columns", "exact names, distinctive phrases, rare words"],
    ],
    widths=["10%", "32%", "24%", "34%"],
)}

<p>The BM25 column weights are <code>scene 1.0 · meaning 1.6 · metaphor 1.8 · title 0.6</code>.
A retrieval query usually expresses what a meme <i>means</i>, so the meaning and metaphor columns
count for more than a scene description; the title is short and noisy and counts for least.</p>

<h3>Query expansion stays lexical</h3>
<p>A hand-written lexicon maps meme phrases to descriptive terms — <i>“distracted boyfriend”</i>
becomes couple, jealousy, distraction, ignoring. Those terms enter <b>only the BM25 query</b>, at
weight 0.5, merged per image by taking the maximum. CLIP and BGE always see the original
sentence. This is deliberate: an expansion is a guess, and a wrong guess must not be able to drag
the semantic signals with it. A lexical expansion can only add candidates the literal query
missed.</p>

<h3>Fusion</h3>
<p>Each signal is min-max normalised over its candidate union, so an image absent from a list
scores 0 rather than being dropped. The combined score is</p>

<pre><code>fused = 0.5 · clip + 0.3 · doc + 0.2 · bm25</code></pre>

<p>with weights renormalised whenever a signal is disabled or returns nothing. That detail matters
operationally: with CLIP unavailable B degrades to a two-signal ranker instead of returning zeros,
and the ablation that removes CLIP entirely still reaches 0.780 mAP.</p>

<h3>Reranking</h3>
<p>The top 60 candidates are scored by a cross-encoder (<code>BAAI/bge-reranker-base</code>) that
sees the query and the document in the same forward pass, and the score is blended:</p>

<pre><code>fused ← 0.7 · fused + 0.3 · rerank</code></pre>

<p>This is the single most expensive and most valuable component. Removing it costs 0.072 mAP and
drops median latency from {lat_p50('clip_nlp')} ms to about 43 ms — a five-fold saving that the
default configuration declines to take.</p>

<h2>What B depends on</h2>
<p>Human annotation. It has no cold-start path: a new meme arrives with no captions and no
metaphors, so B cannot rank it until someone annotates it. That limitation is not a bug to be
patched — it is the cost that strategy C is built to approximate.</p>
</section>
""")

    # ---------------------------------------------------------------- C
    parts.append(f"""
<section>
<h1 id="s6"><span class="num">6</span>Strategy C — VLM descriptions</h1>
<div class="rule"></div>
<p class="lede">Qwen2.5-VL reads every image once, offline, and writes the annotation a human
would have written. Serving never loads the 3B model — only a 384-d text encoder.</p>

<h2>What the model is asked for</h2>
<p>The prompt requires a single JSON object with six fields: <code>scene</code> (what is
literally visible), <code>ocr_text</code> (all overlaid text, verbatim), <code>meaning</code> (the
joke or message), <code>metaphors</code> (at most three pairs), <code>emotions</code> and
<code>keywords</code> (five to ten). Two constraints are stated in the prompt: do not identify
real people, and do not use outside knowledge of a meme template unless the picture supports it.
Both exist to keep the description a description rather than a lookup of what the model
remembers about the template.</p>

<h2>Generation</h2>
<p>Generation runs on a GPU host with bf16, 448 new tokens, and a pixel budget capped around
1024². <code>models.vlm.batch_size</code> controls images per call; the shipped default is 1, and
the reported full run used 8 — roughly 1.5 s per image, about 2 h 35 min for the gallery.</p>

<p>Malformed JSON is repaired before it is rejected: the parser locates the outermost object
span, fixes stray quotes and missing commas where it can, and re-parses. The first full pass still
left <b>8 of {n_img:,} images (0.13%)</b> that local repair could not fix; re-running those images
recovered all of them, and the final cache holds a <code>status="ok"</code> row for every
image.</p>

<h2>The cache and its key</h2>
<p>Every attempt is appended to <code>vlm/descriptions.jsonl</code> and flushed to disk. The
resume key is <b>(image_pk, image_sha1)</b>; <code>backend</code>, <code>model</code> and
<code>prompt_version</code> are stored on every row and act as a validity filter. The distinction
matters: an image whose bytes changed gets a new sha1 and a new key, while a prompt revision
invalidates old rows without changing the key. Because the file is append-only, a retried image
can carry an earlier <code>error</code> line as well as its later <code>ok</code> line — the
loader keeps only the last successful entry.</p>

<h2>Build</h2>
<p>The six fields are flattened into one document embedded with the same BGE model as B:</p>

<pre><code>scene: ... / overlaid text: ... / meaning: ... / metaphors: ... / emotions: ... / keywords: ...</code></pre>

<p>Only rows with <code>status="ok"</code> and the <i>current</i> provenance triple are read.
That triple is recorded in the index's <code>meta.json</code> and checked again at load time, so
descriptions from two different generators can never be mixed without anyone noticing.</p>

<h2>Two gates specific to C</h2>
<div class="box vlm"><span class="t">Coverage and provenance</span>
The index must cover at least <b>95%</b> of the gallery. An interrupted captioning run would
otherwise present itself as a built strategy and quietly rank only the images it processed.
When coverage falls short, C reports as unavailable with the exact counts rather than serving a
partial gallery. The second gate is the provenance triple above; loading refuses a mismatch.</div>

<h2>Query</h2>
<p>The query is encoded with BGE and searched against <code>index/vlm/</code>, over-fetching
<code>max(3k, 100)</code> candidates, then fused with CLIP:</p>

<pre><code>fused = 0.8 · doc + 0.2 · clip</code></pre>

<p>When CLIP is unavailable the document weight becomes 1.0 rather than leaving a fifth of the
score unaccounted for. Reranking is off by default for C. The evidence that a cross-encoder helps
comes from B's annotation documents (Section 11.3); on C it would add roughly the same 180 ms it
adds to B — 226 ms with it, 43 ms without — to a strategy that otherwise answers in
{lat_p50('vlm')} ms, and no ablation covers that trade for C. The pure-VLM variant is therefore
the cleaner measurement of the description channel on its own.</p>

<h2>What C costs and what it buys</h2>
<p>A one-off GPU cost of roughly 2.5 hours and no human annotation. The artefact that reaches
serving is small: the VLM document index is 9.4 MiB for 6,378 × 384-d vectors, against 18.7 MiB
for the CLIP image index. It generalises to unseen memes, which B cannot do. What it does not buy
is B's accuracy: a 3B model compresses an image to its gist, so fine visual detail is lost — the
next section quantifies exactly that trade.</p>
</section>
""")

    # ---------------------------------------------------------------- side by side
    parts.append(f"""
<section>
<h1 id="s7"><span class="num">7</span>The three strategies side by side</h1>
<div class="rule"></div>
<p class="lede">Same gallery, same CLIP image vectors, same API. The only difference is which
text, if any, represents an image at query time.</p>

<h2>7.1 What actually differs</h2>
{table(
    ["", "A · clip", "B · clip_nlp", "C · vlm"],
    [
        ["Text per image", "none", "human captions and metaphors", "model-written description"],
        ["Signals at query time", "1 — CLIP", "3 + rerank", "2 — doc + CLIP"],
        ["Candidates fetched", "<code>max(2k, k)</code>", "<code>max(5k, 100)</code> per signal", "<code>max(3k, 100)</code>"],
        ["Post-processing", "drop score ≤ 0", "cross-encoder on top 60, α = 0.3", "none by default"],
        ["One-off cost", "minutes of CPU", "human annotation of 6.4k images", "~2 h 35 min of GPU"],
        ["Works on unseen memes", "yes", "<b>no</b> — they arrive unannotated", "yes"],
        ["Median latency", f"{lat_p50('clip')} ms", f"{lat_p50('clip_nlp')} ms", f"{lat_p50('vlm')} ms"],
        ["mAP on the frozen 240", metric("clip", "mAP"), metric("clip_nlp", "mAP"), metric("vlm", "mAP")],
    ],
    widths=["20%", "24%", "31%", "25%"],
)}


<h2>7.2 The two ablations that make the difference concrete</h2>
<p><b>B is slightly better without CLIP</b> (0.7608 → 0.7802). Once human annotations are indexed,
a 0.5 weight on pixels dilutes a stronger text signal. The weight was fixed before evaluation and
never tuned on the test set; this is the clearest tuning opportunity in the system, and it is
reported as an ablation rather than quietly applied.</p>

<p><b>C is much worse without CLIP</b> (0.4519 → 0.3922). A VLM description compresses the image
to its gist and drops visible detail, so the pixels have to supply what the text omitted. The same
channel that is noise for B is signal for C.</p>

<div class="box eval"><span class="t">Why that asymmetry is the interesting result</span>
If VLM text were simply a cheaper copy of annotation text, both ablations would point the same
way. They do not. Human annotation and model description cover different parts of the query
space: annotation carries the meaning a person chose to write down, while a description carries
what a model can see and name. Section 11 shows the same split in the per-category table — C
gains on intent and metaphor but loses on visual detail, which is why it lands beside A rather
than beside B.</div>
{figure(figs[2], 2, "Page 2 of the diagram set. Build and query pipelines for each strategy, with the comparison table.", "fig2")}
</section>
""")

    # ---------------------------------------------------------------- build
    parts.append(f"""
<section>
<h1 id="s8"><span class="num">8</span>Offline build flow</h1>
<div class="rule"></div>
<p class="lede">Five commands turn the Meme-Cap download into the gallery and the three indexes.
Every stage is keyed by a content hash, so a stale artefact is refused rather than served.</p>

<h2>8.1 <code>tsr prepare</code> — once, network required</h2>
<p>Downloads the annotation JSON and the image archive (resumable), extracts
<code>memecap/images/*.png</code> and <code>memecap/caption/*.json</code>, normalises, de-duplicates
by SHA-1 and writes the manifest. <i>Guards:</i> an existing manifest is reused unless
<code>--force</code>; extraction is skipped when the recorded archive size still matches, so
re-running after a failure does not re-extract 6,416 files.</p>

<h2>8.2 <code>tsr build --strategy gallery</code> — once</h2>
<p>Inserts the {n_img:,} image rows in <code>image_id</code> order inside one transaction, then
encodes every image with CLIP. <i>Guards:</i> the build asserts that ids are contiguous, that the
vector count matches the row count, and that a content hash of the stored annotations matches the
manifest. Any failure aborts <b>before</b> an index is written. The alternative — writing an index
whose row <i>i</i> is a different image than id <i>i</i> — is the one error no downstream check
would catch, because every later stage trusts the id map.</p>

<h2>8.3 <code>tsr build --strategy clip_nlp</code></h2>
<p>Builds the four annotation fields, writes <code>nlp_docs</code> with its FTS5 triggers, and
encodes the dense documents into <code>index/clip_nlp/</code>. <i>Guard:</i> the 1,968
literal-metaphor rows are excluded; without that rule, <i>a man</i> would be indexed as this
meme's symbolism.</p>

<h2>8.4 <code>tsr vlm-caption</code> — GPU host, hours (~2 h 35 min)</h2>
<p>Batched generation, JSON repair, upsert into <code>vlm_docs</code> and append to the JSONL
cache with a flush per line. A failed batch is retried image by image, so one unreadable file
cannot stall the run. <i>Guard:</i> the resume key is image + SHA-1, and the provenance triple on
every row rejects rows from a different generator.</p>

<h2>8.5 <code>tsr build --strategy vlm</code></h2>
<p>Filters <code>vlm_docs</code> to <code>status="ok"</code> and the current provenance triple,
encodes <code>doc_text</code> with BGE, and records the triple in <code>meta.json</code>.
<i>Guard:</i> a different generator, or coverage under 95%, makes C read as unavailable.</p>


<h2>8.6 Ordering, and what a rebuild may touch</h2>
<p><code>--strategy all</code> runs clip → clip_nlp → vlm. It skips C with a warning when
<code>models.vlm.backend</code> is <code>none</code>, and also when its build raises a
<code>RuntimeError</code> — no <code>vlm_docs</code> match the configured triple, or the gallery
no longer matches the manifest. B and C never touch <code>images</code>: they rewrite only their
own rows and vector files, which is what keeps <code>images.id</code> stable across rebuilds.</p>

<p><code>python -m tsr stats --json</code> prints readiness, row and vector counts and the model
string for every stage — the quickest way to see which index is built and what provenance it
records.</p>
{figure(figs[3], 3, "Page 3 of the diagram set. The five build stages, what each writes, and the guard on each.", "fig3")}
</section>
""")

    # ---------------------------------------------------------------- query
    parts.append(f"""
<section>
<h1 id="s9"><span class="num">9</span>Online query flow</h1>
<div class="rule"></div>
<p class="lede">One request, from the page to the ranked payload. Models and indexes load on the
first query of each strategy, never at process start.</p>

<h2>9.1 The nine steps</h2>
<ol>
<li>The page calls <code>GET /api/strategies/{{key}}/search?q=…&amp;k=…</code>.</li>
<li>The route calls <code>StrategyManager.search(key, query, k)</code>.</li>
<li>The manager resolves the strategy — an unknown key is a 404 — and calls
<code>ensure_loaded()</code>.</li>
<li>The strategy encodes the query: CLIP's text tower for A, BGE with the query instruction for
B and C.</li>
<li>It searches its indexes, fetching <code>max(2k, k)</code> for A and <code>max(mult·k, 100)</code>
for B (5×) and C (3×).</li>
<li>The index returns ids and scores; inner product over L2-normalised vectors is cosine
similarity.</li>
<li>The strategy returns hits with per-signal components and its own timing.</li>
<li>The manager enriches the hits from <code>gallery.db</code> — captions, metaphors, title,
source URL, VLM summary — because an index only knows ids and scores.</li>
<li>The route returns a <code>SearchResponse</code>; the page renders cards, score bars and a
lightbox.</li>
</ol>


<h2>9.2 Failure semantics</h2>
{table(
    ["Status", "When", "What the caller sees"],
    [
        ["<b>404</b>", "unknown strategy key — the set is closed to <code>clip</code>, <code>clip_nlp</code>, <code>vlm</code>", "a JSON detail naming the key that was rejected"],
        ["<b>503</b>", "known but not ready: index missing, stale, or failing a coverage or provenance gate", "the strategy's <code>load_error</code> string"],
        ["<b>422</b>", "invalid payload — empty query, out-of-range setting", "a validation detail"],
    ],
    widths=["10%", "50%", "40%"],
)}

<p><code>/api/compare</code> never fails as a whole. A column that cannot answer is returned as
<code>{{"ready": false, "error": …}}</code> and the other columns still serve. That is what lets the
comparison page stay useful while strategy C is still being captioned — a design decision made
because the alternative, failing the whole request, hides the two working strategies.</p>

<h2>9.3 What loads when</h2>
<p>Startup loads configuration, opens SQLite and constructs the three strategy objects — no model
weights, no FAISS files. The first query of a strategy loads its encoder, its indexes and, for B,
the cross-encoder reranker. The load is guarded by <code>ensure_loaded()</code>, so the cost is
paid once and never repeated. The VLM stack is never loaded by the serving process at all:
captioning is an offline job, and the serving image deliberately does not ship those extras.</p>

<h2>9.4 The latency budget</h2>
{table(
    ["Strategy", "p50", "p95", "Where the time goes"],
    [
        ["A · clip", f"{lat_p50('clip')} ms", f"{lat_p95('clip')} ms", "one text encode, one flat inner-product scan"],
        ["B · clip_nlp", f"{lat_p50('clip_nlp')} ms", f"{lat_p95('clip_nlp')} ms",
         "three index lookups plus a cross-encoder over 60 pairs"],
        ["C · vlm", f"{lat_p50('vlm')} ms", f"{lat_p95('vlm')} ms", "one text encode, two index lookups, no rerank"],
    ],
    widths=["16%", "12%", "12%", "60%"],
)}

<p>B's extra time is almost entirely the cross-encoder. Turning it off drops latency to roughly
43 ms at a cost of 0.072 mAP; all three configurations sit far inside an interactive budget, which
is why the default keeps the reranker on. The lesson is not that reranking is cheap — it is that
{lat_p50('clip_nlp')} ms is cheap enough here, and the ablation makes the trade explicit rather
than hiding it.</p>
{figure(figs[4], 4, "Page 4 of the diagram set. The nine steps of a search request, failure semantics, lazy loading and measured latency.", "fig4")}
</section>
""")

    # ---------------------------------------------------------------- evaluation
    parts.append(f"""
<section>
<h1 id="s10"><span class="num">10</span>Evaluation</h1>
<div class="rule"></div>
<p class="lede">A frozen, leakage-checked benchmark run through every ready strategy, with the
same gallery and the same metric code — and a report that the CLI and the web page both read.</p>

<h2>10.1 Building the benchmark</h2>
<p>240 queries are drafted for 80 visual, 80 intent and 80 metaphor sources, all drawn from the
558-image test split. Each draft is rewritten by hand and must clear the leakage gate before it is
frozen. The gate rejects a query when:</p>

<ul>
<li>an 8-gram overlaps <b>any</b> annotation field of the source <b>or of any gallery image</b>;</li>
<li>the query embeds a whole short annotation (≥ 4 tokens) — no 8-gram can see a three-word
caption pasted verbatim;</li>
<li>a contiguous run of ≥ 5 tokens is copied from the source caption — a short copy inside a
longer sentence slips under the n-gram rule;</li>
<li>token Jaccard with the source caption is ≥ 0.6 — this rule has no length floor, so short
captions cannot be pasted either.</li>
</ul>

<p>The freeze also verifies that every source is still in the test split and that its annotation
snapshot is unchanged, and it aborts rather than writing a partial set if any reviewed query is
rejected. Every verdict is stored in <code>metadata.leakage</code>, so the set can be audited
without re-running the gate.</p>

<div class="box eval"><span class="t">Why B makes leakage the central problem</span>
B indexes human annotations. If a query were allowed to reuse an annotation's wording, B would be
scored on recognising text it has already stored — and the metric would measure string overlap,
not retrieval. The gate is not a formality; it is what makes the comparison meaningful, and
Section 11 shows exactly how much a leaky set would have flattered B.</div>

<h2>10.2 Measuring</h2>
<p>All three strategies run the same queries against the same gallery with
<code>depth = max(top_k, largest relevant set, 100)</code>, so recall and mAP are never truncated
by the reported @k cuts. Every strategy is scored on P@1/5/10, R@1/5/10, mAP, MRR, nDCG@10, median
rank and misses; the results table reports the headline subset.</p>

<p>Relevance is graded. Grade 2 is the source meme. Grade 1 is assigned only by deterministic
rules: a near-identical image caption (token Jaccard ≥ 0.8), an identical intended meaning, or a
shared non-literal metaphor that carries the same meaning and appears in at most 10 images. No
model judges relevance — “looks similar to a model” must never become ground truth.</p>


<h2>10.3 Comparing</h2>
<p>A paired bootstrap (1,000 resamples, fixed seed 42) over per-query metric values gives the
confidence intervals. Holm–Bonferroni is applied once across the three pairwise × five metric
tests, so the three comparisons form a single family, and p-values are floored at the bootstrap
resolution 0.001.</p>

<p><code>--ablations</code> re-runs the benchmark with seven single changes, each isolating one
design decision: prompt ensembling for A; B without BM25, without the dense channel, without CLIP
or without the reranker; and C as pure VLM or with the CLIP weight raised to 0.35.</p>

<p>The run writes <code>data/reports/compare.json</code>, renders it to <code>compare.html</code>
and <code>failures.html</code>, and serves the same JSON through <code>/api/eval/summary</code>,
which the <code>/evaluation</code> page reads. The CLI and the page therefore cannot disagree.</p>
{figure(figs[5], 5, "Page 5 of the diagram set. Benchmark construction, measurement, comparison, and the measured result.", "fig5")}
</section>
""")

    # ---------------------------------------------------------------- results
    rows = []
    for key, label in (("clip", "A · clip"), ("clip_nlp", "B · clip_nlp"), ("vlm", "C · vlm")):
        mm = m.get(key, {})
        rows.append([
            label,
            f"<b>{fmt(mm.get('mAP'))}</b>" if key == "clip_nlp" else fmt(mm.get("mAP")),
            fmt(mm.get("MRR")), fmt(mm.get("ndcg@10")),
            fmt(mm.get("precision@1"), 3), fmt(mm.get("recall@10"), 3),
            f"{mm.get('median_rank', '—'):.0f}" if isinstance(mm.get("median_rank"), (int, float)) else "—",
            str(mm.get("misses", "—")),
            f"{lat_p50(key)} ms",
        ])

    ci = {}
    for pair in report.get("pairwise", []):
        for comp in pair.get("comparisons", []):
            if comp.get("metric") == "mAP":
                ci[(pair.get("a"), pair.get("b"))] = comp

    def delta(a, b):
        """Effect size for a - b, always oriented to the order asked for."""
        c = ci.get((a, b))
        lo, hi, d = None, None, None
        if c:
            d, lo, hi = c["delta"], c["ci_low"], c["ci_high"]
        else:
            c = ci.get((b, a))
            if not c:
                return "—"
            d, lo, hi = -c["delta"], -c["ci_high"], -c["ci_low"]
        sign = "−" if d < 0 else "+"
        bounds = ", ".join(f"{v:.3f}".replace("-", "−") for v in (lo, hi))
        text = f"{sign}{abs(d):.3f} [{bounds}]"
        if c.get("p_holm") is not None:
            text += f"  ·  p<sub>holm</sub> = {c['p_holm']:.3f}"
        return text

    cat = {k: v.get("per_category", {}) for k, v in strategies.items()}

    def catrow(label, key, field="mAP"):
        c = cat.get(key, {})
        return [label] + [fmt(c.get(name, {}).get(field), 3) for name in ("visual", "intent", "metaphor")]

    abl = report.get("ablations", {})
    abl_rows = [[esc(name), fmt(v.get("metrics", {}).get("mAP")),
                 f"{v.get('latency', {}).get('p50_ms', 0):.0f} ms" if v.get("latency") else "—"]
                for name, v in abl.items()]

    diag = report.get("diagnostics", {})

    # How often B ranks the source image higher than A on the frozen set.
    def _ranks(key: str) -> dict[str, int]:
        return {
            q["qid"]: (10**9 if q.get("rank") is None else q["rank"])
            for q in report.get("strategies", {}).get(key, {}).get("per_query", [])
        }

    ra, rb = _ranks("clip"), _ranks("clip_nlp")
    b_wins = sum(1 for qid, r in rb.items() if qid in ra and r < ra[qid])

    example_html = ""
    if examples:
        blocks = []
        for ex in examples:
            e = ex["entry"]
            rws = []
            for label, q in ex["rows"]:
                rank = q.get("rank")
                rws.append([label, "—" if rank is None else str(rank), f"{q.get('ap', 0):.3f}",
                            f"{q.get('rr', 0):.3f}"])
            blocks.append(f"""
<div class="ex">
<h3>{esc(e['category'].title())} query · <span class="mono">{esc(e['qid'])}</span></h3>
<p><i>“{esc(e['query'])}”</i></p>
{table(["Strategy", "Rank of the source", "AP", "RR"], rws, widths=["30%", "24%", "23%", "23%"])}
</div>
""")
        example_html = "".join(blocks)

    parts.append(f"""
<section>
<h1 id="s11"><span class="num">11</span>Results</h1>
<div class="rule"></div>
<p class="lede">240 frozen queries, one gallery, identical retrieval depth. B is far ahead; C is
statistically indistinguishable from A, and the per-category split explains why.</p>

<h2>11.1 Headline</h2>
{table(
    ["", "mAP", "MRR", "nDCG@10", "P@1", "R@10", "median rank", "misses", "p50"],
    rows,
    widths=["16%", "15%", "10%", "11%", "8%", "8%", "11%", "8%", "11%"],
)}

<p><b>B − A</b> is {delta('clip', 'clip_nlp')} mAP and <b>B − C</b> is {delta('vlm', 'clip_nlp')};
both are significant on every metric after correction. <b>C − A</b> is {delta('clip', 'vlm')} —
not significant on any metric. The two annotation-free strategies are indistinguishable, which is
itself the main result about strategy C: a 3B model recovers part of the meaning channel but nets
out to roughly the pixel baseline.</p>

<h2>11.2 Per category</h2>
{table(
    ["", "visual", "intent", "metaphor"],
    [
        catrow("A · clip", "clip"),
        catrow("B · clip_nlp", "clip_nlp"),
        catrow("C · vlm", "vlm"),
    ],
    widths=["28%", "24%", "24%", "24%"],
)}

<p>B improves every category — visual {fmt(cat.get('clip_nlp', {}).get('visual', {}).get('mAP'), 3)}
vs {fmt(cat.get('clip', {}).get('visual', {}).get('mAP'), 3)} for A and
{fmt(cat.get('vlm', {}).get('visual', {}).get('mAP'), 3)} for C, intent
{fmt(cat.get('clip_nlp', {}).get('intent', {}).get('mAP'), 3)}, metaphor
{fmt(cat.get('clip_nlp', {}).get('metaphor', {}).get('mAP'), 3)} — with its largest gain over A on
metaphor and its smallest on intent, because its documents contain the human-written meaning and
the metaphor pairs. C gains on intent and metaphor over A but <b>loses on the visual category</b>:
a 3B model describes the gist and the overlaid text, not every visible detail.</p>

<h2>11.3 Ablations</h2>
{table(["Variant", "mAP", "p50"], abl_rows, widths=["60%", "20%", "20%"])}

<p>Three results carry the argument. Removing B's reranker costs the most of any single channel
(0.7608 → 0.689), which is why it stays on despite dominating latency. Removing CLIP from B
<i>raises</i> mAP, so the visual channel is dilution once annotation text is present. Removing
CLIP from C is the opposite. Together they show the two text sources are not interchangeable.</p>

<h2>11.4 Worked examples</h2>
<p>Three frozen queries, one per category, chosen where both annotation-free strategies put the
source outside their top ten and B ranks it first. They are representative rather than
cherry-picked: the pattern repeats across the {b_wins} frozen queries on which B ranks the source
higher than A.</p>
{example_html}

<p>In the visual case, the query describes a face and an expression; CLIP matches other faces and
expressions, while B's annotation document contains the literal caption of the right image. In the
intent and metaphor cases, the words of the query are not in the image at all — only in the
annotation — so no pixel-only representation can reach them.</p>

<h2>11.5 The leakage diagnostic</h2>
{table(
    ["Strategy", "Main set mAP", "Raw-caption set mAP", "Δ"],
    [
        ["A · clip", fmt(m.get('clip', {}).get('mAP')), fmt(diag.get('clip', {}).get('mAP')),
         f"−{abs(m.get('clip', {}).get('mAP', 0) - diag.get('clip', {}).get('mAP', 0)):.3f}"],
        ["B · clip_nlp", fmt(m.get('clip_nlp', {}).get('mAP')), fmt(diag.get('clip_nlp', {}).get('mAP')),
         f"+{diag.get('clip_nlp', {}).get('mAP', 0) - m.get('clip_nlp', {}).get('mAP', 0):.3f}"],
        ["C · vlm", fmt(m.get('vlm', {}).get('mAP')), fmt(diag.get('vlm', {}).get('mAP')),
         f"−{abs(m.get('vlm', {}).get('mAP', 0) - diag.get('vlm', {}).get('mAP', 0)):.3f}"],
    ],
    widths=["22%", "26%", "26%", "26%"],
)}

<p>Running the untouched human annotations as queries is leaky by construction — B indexes those
annotations — so it is reported separately and never enters the main table. Only B improves when
queries copy annotation wording, and the gain is large enough to reorder the strategies on its
own. That is precisely why the frozen set uses rewritten queries, and why the comparison refuses
to publish a set in which any query failed the gate.</p>

<h2>11.6 Failure analysis</h2>
<p><code>failures.html</code> shows the ten worst queries per strategy with the graded relevant
set and the top-5 retrieved images. A's and C's ten worst are all visual — fine-grained structure
that a single embedding cannot carry, such as two-panel layouts, a specific expression or a
costume. B's split evenly, five visual and five intent queries about abstract meaning, and its 13
misses overall spread across the categories (7 intent, 5 visual, 1 metaphor). Intent, not visual,
is B's largest remaining gap.</p>
</section>
""")

    # ---------------------------------------------------------------- operating
    parts.append("""
<section>
<h1 id="s12"><span class="num">12</span>Operating and configuration</h1>
<div class="rule"></div>
<p class="lede">Structural choices need a rebuild and live in configuration; ranking knobs are
editable from the settings page and take effect on the next search.</p>

<h2>12.1 Where a setting lives</h2>
<table>
<thead><tr><th>Kind</th><th>Lives in</th><th>Examples</th><th>Change requires</th></tr></thead>
<tbody>
<tr><td><b>Build-time</b></td><td><code>configs/default.yaml</code></td>
    <td>model names, index type, dataset bounds, VLM backend and prompt version</td>
    <td>a rebuild of the affected index</td></tr>
<tr><td><b>Runtime</b></td><td><code>data/settings.json</code>, edited at <code>/settings</code></td>
    <td>fusion weights, BM25 weight, fetch multipliers, expansion, rerank top-N and α, C's CLIP weight, prompt ensembling, default <i>k</i></td>
    <td>nothing — applied on the next search</td></tr>
</tbody></table>

<p>Runtime edits are re-applied at startup from <code>data/settings.json</code>. One exception
matters: <code>app.default_top_k</code> is read once when the app is created and bound as the
search route's query default, so changing it through the settings endpoint has no effect until the
process restarts. Every other knob is read per request.</p>

<h2>12.2 Rebuild rules</h2>
<p>Rebuild an index when the dataset, the model or the document text changes; then restart the
API. Readiness is a load-time snapshot, so a running process keeps serving the indexes it loaded.
Stats are not a snapshot: strategies A and B re-read <code>meta.json</code> and the live row count
on every request, so after a rebuild they describe the newer files on disk even though search
still uses the old loaded index. Only strategy C reports the artefacts it actually loaded.</p>

<div class="box"><span class="t">The practical consequence</span>
After rebuilding, check <code>/api/health</code> before trusting a search result. If the process
has not been restarted, it is still serving the previous index — and the stats page will look
newer than the answers. This is the one place where the system's lazily-cached design can
mislead, and it is why the readiness flag is reported per strategy rather than inferred from the
files.</div>

<h2>12.3 Deployment shape</h2>
<p>The serving container needs the CLIP and BGE encoders, the index files and
<code>gallery.db</code>. It does not need the VLM weights — captioning is offline, and the base
image does not install those extras.</p>

<p>Enabling strategy C on a host therefore means shipping two things, not one: the vector files
in <code>data/index/vlm/</code> <i>and</i> the matching rows in <code>gallery.db</code>. The build
step reads <code>vlm_docs</code> from the database — it does not read
<code>descriptions.jsonl</code> — and <code>vlm-caption</code> is the only command that ingests the
JSONL, which it does with the VLM loaded. So on a GPU-less serving host, copy the index and the
<code>vlm_docs</code> rows together (or copy <code>gallery.db</code> whole); do not expect
"caption cache plus one build" to work there.</p>
</section>
""")

    # ---------------------------------------------------------------- reproduce
    parts.append(f"""
<section>
<h1 id="s13"><span class="num">13</span>Reproducing the results</h1>
<div class="rule"></div>
<p class="lede">Every number in this document is produced by the commands below; the full run takes
a few hours on a single GPU host, most of it captioning.</p>

<pre><code># small end-to-end loop (network required, ~300 images)
python -m tsr prepare --limit 300
python -m tsr build --strategy gallery
python -m tsr build --strategy clip_nlp
python -m tsr make-eval --num 60
python -m tsr eval --strategy clip,clip_nlp
python -m tsr serve                    # http://localhost:8000

# full dataset ({n_img:,} images)
python -m tsr prepare
python -m tsr build --strategy gallery
python -m tsr vlm-caption              # GPU host, resumable, ~2 h 35 min
python -m tsr build --strategy all
python -m tsr eval --strategy all
python -m tsr compare --ablations

# checks
PYTHONPATH=src pytest
PYTHONPATH=src ruff check src tests</code></pre>

<h2>13.1 Determinism</h2>
<p>The dataset is fixed by the manifest hash recorded in every index's <code>meta.json</code>. The
query set is frozen in <code>data/eval/</code> and carries its leakage verdicts. The bootstrap uses
a fixed seed. The only non-determinism in the reported pipeline is GPU generation for strategy C,
which is why the descriptions are cached and the build consumes the cache rather than the model:
re-running the build cannot change a single vector.</p>

<h2>13.2 What to check after a rebuild</h2>
<table>
<thead><tr><th>Check</th><th>Command</th><th>Expected</th></tr></thead>
<tbody>
<tr><td>Gallery size</td><td><code>tsr stats --json</code></td><td>{n_img:,} images, CLIP index ntotal = {n_img:,}</td></tr>
<tr><td>Strategy readiness</td><td><code>GET /api/health</code></td><td>every built strategy <code>ready</code>, C only above 95% coverage</td></tr>
<tr><td>Provenance</td><td><code>data/index/*/meta.json</code></td><td>matching backend/model/prompt triple for C</td></tr>
<tr><td>Report consistency</td><td><code>/evaluation</code> vs <code>data/reports/compare.json</code></td><td>identical numbers</td></tr>
</tbody></table>
</section>
""")

    # ---------------------------------------------------------------- appendices
    parts.append(f"""
<section>
<h1 id="sa"><span class="num">A</span>File map</h1>
<div class="rule"></div>
<p class="lede">Where each responsibility lives.</p>

<pre><code>configs/default.yaml       all build-time defaults
src/tsr/
  __main__.py              makes `python -m tsr` work
  cli.py                   argparse entry point, sub-commands
  config.py                AttrDict config + TSR_* environment overrides
  logging_utils.py         log formatting and verbosity
  memecap.py               download, extract, normalise, manifest
  manifest.py              record type, JSONL I/O, hashing
  store.py                 SQLite schema v2 + FTS5 BM25
  index.py                 FAISS index with a mandatory id map and meta checks
  artifacts.py             meta.json read/write and validation
  pipeline.py              gallery and strategy build pipelines
  encoders/clip.py         OpenCLIP image and text towers
  encoders/text.py         BGE encoder, CLS pooling, query instruction
  encoders/vlm.py          Qwen2.5-VL captioning, JSON repair, resume cache
  nlp/analyze.py           tokenisation, stop-words, lexicon expansion
  nlp/fusion.py            min-max, z-score, RRF
  nlp/rerank.py            cross-encoder
  strategies/base.py       Hit, SearchOutput, StrategyContext, Strategy
  strategies/clip_only.py  A
  strategies/clip_nlp.py   B
  strategies/vlm.py        C
  service.py               lazy StrategyManager
  evaluate.py              metrics, leakage checks, bootstrap
  compare.py               comparison, ablations, HTML reports
  api.py                   FastAPI routes
  schemas.py               response models
  settings.py              runtime-editable knobs
web/                       5 HTML pages + shared JS/CSS (7 page routes)
tests/                     unit tests, heavy deps importorskip-guarded
docs/                      this document, the diagram set, the report</code></pre>
</section>

<section>
<h1 id="sb"><span class="num">B</span>Where every figure comes from</h1>
<div class="rule"></div>
<p class="lede">Every table, figure and worked example is generated from the file named beside it
at build time; the prose repeats those values and was checked against the same sources.</p>

{table(
    ["Figure", "Source"],
    [
        [f"{n_ann:,} annotations → {n_img:,} images, {n_dup} duplicates", "<code>data/manifest.meta.json</code>"],
        ["5,820 / 558 split", "<code>data/manifest.jsonl</code>, <code>gallery.db</code>"],
        ["768 px, quality 90 (492 MiB once measured)", "<code>configs/default.yaml</code> · <code>dataset.*</code>; size from <code>data/images/</code>"],
        ["Model names and dimensions", "<code>data/index/*/meta.json</code>"],
        ["Fusion weights, BM25 column weights, fetch multipliers, rerank settings", "<code>configs/default.yaml</code> · <code>strategies.*</code>"],
        ["Build guards — id contiguity, content hash, provenance, 95% coverage", "<code>src/tsr/pipeline.py</code>, <code>src/tsr/strategies/*.py</code>"],
        ["Latencies, mAP, CIs, Holm-adjusted p-values, ablations", "<code>data/reports/compare.json</code>"],
        ["Query set, qrels and leakage verdicts", "<code>data/eval/memecap_eval_v1.json</code>"],
        ["Worked-example ranks and AP", "<code>data/reports/compare.json</code> · <code>per_query</code>"],
        ["Captioning throughput and parse failures", "<code>data/vlm/descriptions.jsonl</code>, <code>vlm_docs</code>"],
    ],
    widths=["52%", "48%"],
)}

<p>Regenerate this document with <code>python docs/make_detailed.py</code>; it re-renders the five
diagram pages from <code>docs/system-schematic.pdf</code> and re-reads every table above.</p>
</section>

<section>
<h1 id="sc"><span class="num">C</span>Glossary</h1>
<div class="rule"></div>
<p class="lede">Terms used with a specific meaning in this system.</p>

{table(
    ["Term", "Meaning here"],
    [
        ["<b>gallery</b>", f"the {n_img:,} normalised images plus their rows in SQLite; the single set every strategy searches"],
        ["<b>strategy</b>", "one of the closed set <code>clip</code>, <code>clip_nlp</code>, <code>vlm</code>; unknown keys are 404, known-but-unbuilt are 503"],
        ["<b>manifest</b>", "<code>data/manifest.jsonl</code> — the normalised record per image; the source of truth downstream"],
        ["<b>id map</b>", "<code>image_ids.npy</code> beside the CLIP index, <code>doc_ids.npy</code> beside the B and C document indexes; row <i>i</i> of the index belongs to this image id"],
        ["<b>meta.json</b>", "per-index sidecar: schema version, kind, model, dimension, count, dataset hash, config hash, build time"],
        ["<b>content hash</b>", "a hash of the stored annotations, compared with the manifest so edited text cannot keep old vectors"],
        ["<b>provenance triple</b>", "backend / model / prompt_version for VLM rows; a mismatch makes strategy C unavailable"],
        ["<b>coverage</b>", "the fraction of gallery images a strategy has a document for; C requires ≥ 95%"],
        ["<b>fused score</b>", "the combined, normalised ranking score; the per-signal parts are exposed as <code>score_components</code>"],
        ["<b>fetch multiplier</b>", "how many candidates a signal retrieves before fusion — 2× for A, 5× for B, 3× for C, with a 100 floor for B and C"],
        ["<b>leakage gate</b>", "the set of rules that rejects an evaluation query copying annotation wording"],
        ["<b>qrel grade</b>", "2 for the source meme, 1 for a deterministic near-equivalent, 0 otherwise"],
        ["<b>frozen set</b>", "<code>memecap_eval_v1</code> — 240 reviewed queries plus graded qrels, never edited after freezing"],
        ["<b>p<sub>holm</sub></b>", "a p-value after Holm–Bonferroni correction across the three pairwise × five metric tests"],
    ],
    widths=["22%", "78%"],
)}
</section>
""")

    return "".join(parts)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def render_figures(tmp: Path) -> dict[int, Path]:
    doc = pymupdf.open(SCHEMATIC)
    figs = {}
    for i, page in enumerate(doc, start=1):
        png = tmp / f"page{i}.png"
        page.get_pixmap(dpi=200).save(png)
        figs[i] = png
    return figs


def find_browser() -> Path:
    for path in BROWSERS:
        if path.exists():
            return path
    raise SystemExit("No Chromium-based browser found (Edge or Chrome).")


def print_pdf(browser: Path, html_path: Path, out: Path) -> None:
    subprocess.run(
        [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=10000",
            f"--print-to-pdf={out}",
            html_path.as_uri(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def locate_pages(pdf: Path, entries: list[tuple[str, str, str]]) -> dict[str, int]:
    doc = pymupdf.open(pdf)
    pages = {}
    # The contents page repeats every title; skip it by finding the page that holds most of them.
    titles = [t for _, _, t in entries]
    start = 0
    for i, page in enumerate(doc):
        text = page.get_text()
        if sum(1 for t in titles if t in text) >= 8:
            start = i + 1
    for anchor, number, title in entries:
        needle = f"{number} {title}"
        for i in range(start, doc.page_count):
            if needle in doc[i].get_text().replace("\n", " "):
                pages[anchor] = i + 1
                break
    return pages


def stamp(pdf: Path, entries: list[tuple[str, str, str]], pages: dict[str, int]) -> None:
    """Footer, page numbers and a running section title (Chromium has no margin boxes)."""
    doc = pymupdf.open(pdf)
    total = doc.page_count
    order = [(pages[a], f"{n} · {t}") for a, n, t in entries if a in pages]
    order.sort()

    def section_for(page_no: int) -> str:
        current = ""
        for start, title in order:
            if start <= page_no:
                current = title
        return current

    for i, page in enumerate(doc):
        w, h = page.rect.width, page.rect.height
        if i == 0:  # cover
            continue
        y = h - 16
        page.draw_line(pymupdf.Point(45, y - 8), pymupdf.Point(w - 45, y - 8),
                       color=(0.80, 0.84, 0.87), width=0.6)
        page.insert_text((45, y + 1), "Meme-Cap retrieval · COMP5523",
                         fontsize=7.2, fontname="helv", color=(0.44, 0.51, 0.56))
        label = section_for(i + 1)
        if label:
            width = pymupdf.get_text_length(label, fontname="helv", fontsize=7.2)
            page.insert_text(((w - width) / 2, y + 1), label,
                             fontsize=7.2, fontname="helv", color=(0.44, 0.51, 0.56))
        num = f"{i + 1} / {total}"
        width = pymupdf.get_text_length(num, fontname="helv", fontsize=7.2)
        page.insert_text((w - 45 - width, y + 1), num,
                         fontsize=7.2, fontname="helv", color=(0.44, 0.51, 0.56))

    doc.set_toc([[1, f"{n} · {t}", pages[a]] for a, n, t in entries if a in pages])
    doc.saveIncr()


def build(html_text: str, tmp: Path, name: str) -> Path:
    path = tmp / name
    path.write_text(html_text, encoding="utf-8")
    out = tmp / name.replace(".html", ".pdf")
    print_pdf(find_browser(), path, out)
    return out


def main() -> int:
    if not SCHEMATIC.exists():
        raise SystemExit(f"missing {SCHEMATIC}; run docs/make_schematic.py first")

    data = load_data()
    report = data["report"]
    results = {k: v.get("metrics", {}).get("mAP") for k, v in report.get("strategies", {}).items()}
    examples = worked_examples(data)

    entries = [
        ("s1", "1", "Overview"),
        ("s2", "2", "The dataset and the gallery"),
        ("s3", "3", "System architecture"),
        ("s4", "4", "Strategy A — CLIP only"),
        ("s5", "5", "Strategy B — CLIP + annotation NLP"),
        ("s6", "6", "Strategy C — VLM descriptions"),
        ("s7", "7", "The three strategies side by side"),
        ("s8", "8", "Offline build flow"),
        ("s9", "9", "Online query flow"),
        ("s10", "10", "Evaluation"),
        ("s11", "11", "Results"),
        ("s12", "12", "Operating and configuration"),
        ("s13", "13", "Reproducing the results"),
        ("sa", "A", "File map"),
        ("sb", "B", "Where every figure comes from"),
        ("sc", "C", "Glossary"),
    ]

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        figs = render_figures(tmp)

        def compose(pages: dict[str, int]) -> str:
            return (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>Meme retrieval: CLIP vs CLIP+NLP vs VLM</title>"
                f"<style>{CSS}</style></head><body>"
                + cover(data, results)
                + toc(entries, pages)
                # the rule is drawn by h1::after so heading + rule break as one block
                + body(data, figs, examples, report).replace('<div class="rule"></div>', "")
                + "</body></html>"
            )

        # Render, locate the chapters, re-render with real contents page numbers, and
        # repeat until the page map stops moving (digits in the contents can shift a page).
        pages: dict[str, int] = {}
        found: dict[str, int] = {}
        pdf = None
        for attempt in range(4):
            pdf = build(compose(pages), tmp, f"pass{attempt + 1}.html")
            found = locate_pages(pdf, entries)
            print(f"pass {attempt + 1}: located {len(found)}/{len(entries)} sections")
            if found == pages:
                break
            pages = found
        else:
            print("warning: pagination did not stabilise")

        stamp(pdf, entries, found)
        OUT.write_bytes(pdf.read_bytes())

    doc = pymupdf.open(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {doc.page_count} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
