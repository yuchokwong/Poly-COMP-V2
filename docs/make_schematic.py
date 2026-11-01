#!/usr/bin/env python3
"""Render docs/system-schematic.pdf.

Five landscape A4 pages: architecture, per-strategy pipelines and comparison, offline
build flow, online query flow and the evaluation pipeline. Every number and name is taken from the repository
(configs/default.yaml, src/tsr, data/index/*/meta.json); re-run after a rebuild
to refresh the measured figures.

    python docs/make_schematic.py
"""

from __future__ import annotations

import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

W, H = 11.69, 8.27  # A4 landscape, inches

INK = "#16202B"
BODY = "#4A5A68"
MUTED = "#71818F"
RULE = "#CBD5DE"
PAPER = "#FFFFFF"

C = {
    "data": {"fill": "#F2F5F8", "edge": "#66798A"},
    "clip": {"fill": "#E9F5F0", "edge": "#1F8A70"},
    "nlp": {"fill": "#EBEFF8", "edge": "#4A63B0"},
    "vlm": {"fill": "#FBF1E3", "edge": "#BF7A20"},
    "serve": {"fill": "#EDF3FA", "edge": "#2F6FB0"},
    "eval": {"fill": "#F8EEEF", "edge": "#A6474E"},
    "plain": {"fill": PAPER, "edge": "#9AAAB7"},
}

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0,
    }
)


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def wrap(text: str, width_pt: float, size: float) -> str:
    """Wrap to the character count that fits ``width_pt`` points at ``size``."""
    chars = max(8, int(width_pt / (size * 0.505)))
    out = []
    for line in text.split("\n"):
        out.extend(textwrap.wrap(line, chars) or [""])
    return "\n".join(out)


@dataclass
class Node:
    x: float
    y: float
    w: float
    h: float

    @property
    def left(self):
        return (self.x, self.y + self.h / 2)

    @property
    def right(self):
        return (self.x + self.w, self.y + self.h / 2)

    @property
    def top(self):
        return (self.x + self.w / 2, self.y + self.h)

    @property
    def bottom(self):
        return (self.x + self.w / 2, self.y)


def box(
    ax,
    x,
    y,
    w,
    h,
    title: str,
    body: str = "",
    *,
    tone: str = "plain",
    title_size: float = 7.4,
    body_size: float = 6.1,
    title_color: str | None = None,
    body_color: str = BODY,
    lw: float = 0.9,
    radius: float = 0.06,
    dashed: bool = False,
    align: str = "center",
    pad: float = 0.09,
    gap: float = 0.05,
) -> Node:
    style = C[tone]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=style["fill"],
        edgecolor=style["edge"],
        linewidth=lw,
        linestyle=(0, (3, 2)) if dashed else "solid",
        zorder=2,
    )
    ax.add_patch(patch)

    inner = (w - 2 * pad) * 72.0
    wrapped = wrap(body, inner, body_size) if body else ""
    lines = wrapped.split("\n") if wrapped else []
    line_h = body_size * 1.34 / 72.0
    title_h = title_size * 1.36 / 72.0
    total = title_h + (gap if lines else 0) + len(lines) * line_h

    if align == "center":
        top = y + h / 2 + total / 2
    else:  # top
        top = y + h - pad * 0.55
    cx = x + w / 2 if align == "center" else x + pad

    ax.text(
        cx,
        top,
        title,
        fontsize=title_size,
        fontweight="bold",
        color=title_color or style["edge"],
        ha="center" if align == "center" else "left",
        va="top",
        zorder=3,
    )
    if lines:
        ax.text(
            cx,
            top - title_h - gap,
            wrapped,
            fontsize=body_size,
            color=body_color,
            ha="center" if align == "center" else "left",
            va="top",
            linespacing=1.34,
            zorder=3,
        )
    return Node(x, y, w, h)


def arrow(
    ax,
    start,
    end,
    *,
    color: str = MUTED,
    label: str | None = None,
    style: str = "-|>",
    lw: float = 0.9,
    rad: float = 0.0,
    ls: str = "solid",
    label_size: float = 5.6,
    label_pos: float = 0.5,
    label_dy: float = 0.075,
    label_color: str | None = None,
    zorder: int = 1,
    connect: str | None = None,
):
    kw = {}
    if connect:
        kw["connectionstyle"] = connect
    elif rad:
        kw["connectionstyle"] = f"arc3,rad={rad}"
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=8.5,
        lw=lw,
        color=color,
        linestyle=ls,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
        **kw,
    )
    ax.add_patch(patch)
    if label:
        mx = start[0] + (end[0] - start[0]) * label_pos
        my = start[1] + (end[1] - start[1]) * label_pos
        ax.text(
            mx,
            my + label_dy,
            label,
            fontsize=label_size,
            color=label_color or color,
            ha="center",
            va="bottom",
            zorder=zorder + 1,
            bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 0.6, "alpha": 0.85},
        )


def band(ax, x, y, w, h, label: str, *, tone="data", pad_label=0.14):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor="#FAFCFD",
            edgecolor=RULE,
            linewidth=0.7,
            zorder=0,
        )
    )
    ax.text(
        x + pad_label,
        y + h - 0.13,
        label,
        fontsize=7.0,
        fontweight="bold",
        color=C[tone]["edge"],
        ha="left",
        va="top",
        zorder=1,
    )


def page_head(ax, title: str, subtitle: str, number: int, total: int = 5):
    ax.text(0.55, 7.98, title, fontsize=19, fontweight="bold", color=INK, va="top")
    ax.text(0.57, 7.47, subtitle, fontsize=8.6, color=BODY, va="top")
    ax.plot([0.55, W - 0.55], [7.30, 7.30], color=RULE, lw=0.8)
    ax.text(
        W - 0.55,
        7.98,
        f"{number} / {total}",
        fontsize=8.5,
        color=MUTED,
        ha="right",
        va="top",
    )


def page_foot(ax, left: str, right: str = "Meme-Cap retrieval  ·  COMP5523  ·  system schematic"):
    ax.plot([0.55, W - 0.55], [0.62, 0.62], color=RULE, lw=0.6)
    ax.text(0.55, 0.42, left, fontsize=5.9, color=MUTED, va="baseline")
    ax.text(W - 0.55, 0.42, right, fontsize=5.9, color=MUTED, ha="right", va="baseline")


def chip(ax, x, y, text, *, tone="data", size=5.9, mono=True, pad=0.05):
    """Small inline label; returns its width so callers can lay out a row."""
    width = len(text) * size * (0.60 if mono else 0.52) / 72.0 + 2 * pad
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            0.155,
            boxstyle="round,pad=0,rounding_size=0.03",
            facecolor=C[tone]["fill"],
            edgecolor=C[tone]["edge"],
            linewidth=0.5,
            alpha=0.9,
            zorder=2,
        )
    )
    ax.text(
        x + width / 2,
        y + 0.0775,
        text,
        fontsize=size,
        color=C[tone]["edge"],
        ha="center",
        va="center",
        family="DejaVu Sans Mono" if mono else "Arial",
        zorder=3,
    )
    return width


# --------------------------------------------------------------------------- #
# page 1 - architecture
# --------------------------------------------------------------------------- #
def page_architecture(fig, ax, number: int):
    page_head(
        ax,
        "System architecture",
        "One Meme-Cap gallery, three representations of the same 6,378 images, one "
        "benchmark and one FastAPI service",
        number,
    )

    # ------------------------------------------------------------- offline
    band(ax, 0.55, 4.95, W - 1.10, 2.20, "OFFLINE  ·  built once, stored in data/", tone="data")

    y1 = 6.02
    h1 = 0.74
    w1 = 2.38
    xs = [0.78, 3.42, 6.06, 8.70]
    n_src = box(
        ax, xs[0], y1, w1, h1,
        "Meme-Cap source",
        "6,382 annotated memes\nannotations: GitHub raw JSON\nimages: HF mirror zip (1.7 GB)",
        tone="data",
    )
    n_norm = box(
        ax, xs[1], y1, w1, h1,
        "Normalise",
        "transparent PNG on white\nlongest side 768 px, JPEG q=90\n492 MiB in data/images/",
        tone="data",
    )
    n_dedup = box(
        ax, xs[2], y1, w1, h1,
        "De-duplicate",
        "SHA-1 of the source PNG\n6,382 annotated → 6,378 unique\n5,820 trainval · 558 test",
        tone="data",
    )
    n_man = box(
        ax, xs[3], y1, w1, h1,
        "manifest.jsonl",
        "one record per image: image_id,\nsplit, captions, metaphors, sha1\n+ manifest.meta.json",
        tone="data",
    )
    for a, b in ((n_src, n_norm), (n_norm, n_dedup), (n_dedup, n_man)):
        arrow(ax, (a.right[0] + 0.02, a.right[1]), (b.left[0] - 0.02, b.left[1]))

    # artefacts panel
    ax.add_patch(
        FancyBboxPatch(
            (0.78, 5.10), 10.30, 0.68,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=PAPER, edgecolor=C["data"]["edge"], linewidth=0.8, zorder=2,
        )
    )
    ax.text(0.94, 5.60, "data/", fontsize=6.6, fontweight="bold", color=C["data"]["edge"],
            family="DejaVu Sans Mono", va="center", zorder=3)
    ax.text(0.94, 5.34, "what the server\nreads at query time", fontsize=5.3, color=MUTED,
            va="center", linespacing=1.3, zorder=3)
    art = [
        ("gallery.db", "images · annotations · metaphors\nvlm_docs · strategy_docs · FTS5"),
        ("index/clip/", "6,378 × 768-d CLIP image vectors\nimage.faiss · image_ids.npy · meta.json"),
        ("index/clip_nlp/", "6,378 × 384-d BGE annotation docs\ndoc.faiss · doc_ids.npy · meta.json"),
        ("index/vlm/", "6,378 × 384-d BGE VLM docs\nmeta.json records backend + prompt"),
    ]
    for i, (title, body) in enumerate(art):
        x = 2.62 + i * 2.10
        ax.text(x, 5.62, title, fontsize=6.0, fontweight="bold", color=INK,
                family="DejaVu Sans Mono", va="center", zorder=3)
        ax.text(x, 5.42, body, fontsize=5.2, color=BODY, va="top", linespacing=1.32, zorder=3)
        if i:
            ax.plot([x - 0.16, x - 0.16], [5.20, 5.68], color=RULE, lw=0.6, zorder=3)
    arrow(ax, n_man.bottom, (n_man.bottom[0], 5.80), lw=0.9)

    # -------------------------------------------------------------- online
    band(ax, 0.55, 1.12, W - 1.10, 3.62, "ONLINE  ·  tsr serve, FastAPI + uvicorn", tone="serve")

    nb = box(
        ax, 0.78, 2.02, 1.45, 1.94,
        "Browser",
        "static pages from web/\n\n/clip   /clip-nlp\n/vlm   /compare\n/evaluation\n/settings",
        tone="serve", title_size=7.0, body_size=5.8,
    )
    na = box(
        ax, 2.52, 2.02, 1.52, 1.94,
        "FastAPI",
        "api.py\n18 API routes\nPydantic response models\n\n404 unknown\n503 unbuilt\n422 invalid settings",
        tone="serve", title_size=7.0, body_size=5.8,
    )
    nm = box(
        ax, 4.34, 2.02, 1.70, 1.94,
        "StrategyManager",
        "service.py\n\nlazy ensure_loaded()\nRLock around every search\nreadiness reported, never\nraised to the page",
        tone="serve", title_size=7.0, body_size=5.8,
    )
    arrow(ax, nb.right, na.left)
    arrow(ax, na.right, nm.left)

    # artefacts feed the manager (which owns the store and builds the strategies)
    arrow(ax, (4.55, 5.10), (4.55, 3.98), ls=(0, (3, 2)), color=C["data"]["edge"], lw=0.9)
    ax.text(
        4.68,
        4.54,
        "each strategy verifies and loads\nits own artefacts on first use",
        fontsize=5.1,
        color=C["data"]["edge"],
        style="italic",
        ha="left",
        va="center",
        linespacing=1.3,
        zorder=4,
    )

    sx = 6.32
    sw = 3.32
    sh = 0.92
    specs = [
        ("clip", "A · clip",
         "OpenCLIP ViT-L-14 text→image cosine over the shared image index. "
         "No annotations, no OCR, no extra knowledge.",
         "index/clip/  ·  components {clip}"),
        ("nlp", "B · clip_nlp",
         "0.5·CLIP + 0.3·dense annotation docs + 0.2·field-weighted BM25, then a "
         "cross-encoder rerank of the top 60 (α = 0.3).",
         "index/clip + index/clip_nlp + FTS5  ·  {clip, doc, bm25, rerank}"),
        ("vlm", "C · vlm",
         "0.8·BGE over Qwen2.5-VL descriptions + 0.2·CLIP. The VLM itself never "
         "runs at query time.",
         "index/vlm + index/clip  ·  {doc, clip}"),
    ]
    strat = []
    for i, (tone, title, body, comp) in enumerate(specs):
        y = 3.52 - i * 1.02
        n = box(ax, sx, y, sw, sh, title, body, tone=tone, title_size=7.4, body_size=5.7, align="top")
        ax.text(sx + 0.10, y + 0.09, comp, fontsize=5.1, color=C[tone]["edge"], style="italic",
                ha="left", va="bottom", zorder=3)
        strat.append(n)
        arrow(ax, nm.right, (sx - 0.03, n.left[1]), rad=0.06 if i == 0 else (-0.05 if i == 2 else 0.0),
              lw=0.85)

    nr = box(
        ax, 9.80, 1.50, 1.34, 2.94,
        "SearchResponse",
        "rank · score\nscore_components\nimg / meme captions\nmetaphors\nVLM summary\n\n"
        "rendered by web/*.js as cards, score bars and a lightbox",
        tone="serve", title_size=6.6, body_size=5.3, align="top",
    )
    for n in strat:
        arrow(ax, n.right, (nr.x - 0.02, n.right[1]), lw=0.8, color=C["serve"]["edge"])

    ax.text(6.32, 1.30, "Build and query pipelines for each strategy, with the full comparison table, are on page 2.",
            fontsize=5.8, color=MUTED, style="italic")

    # ---------------------------------------------------------- invariants
    ax.add_patch(Rectangle((0.55, 0.72), W - 1.10, 0.30, facecolor="#F7F9FB", edgecolor=RULE, lw=0.5))
    ax.text(0.70, 0.87, "INVARIANTS", fontsize=6.0, fontweight="bold", color=C["data"]["edge"], va="center")
    ax.text(
        1.72,
        0.87,
        "images.id is the only key shared by SQLite, FAISS row ids and image_ids.npy — assigned once, never renumbered.   "
        "Every index carries meta.json (schema, model, dim, count, dataset hash); a mismatch is refused, never served.",
        fontsize=5.4,
        color=BODY,
        va="center",
    )

    page_foot(
        ax,
        "Measured on the frozen set (240 queries): mAP  A 0.4395  ·  B 0.7608  ·  C 0.4519.   "
        "Artifact counts from data/index/*/meta.json.",
    )
    return fig


def page_strategies(fig, ax, number: int):
    page_head(
        ax,
        "The three strategies, side by side",
        "Same gallery, same CLIP image vectors, same API — the only difference is which text, if any, "
        "represents an image at query time.",
        number,
    )

    cols = [
        {
            "tone": "clip", "name": "A · clip",
            "tag": "Pixels only — the zero-knowledge baseline",
            "build": [
                ("Encode the gallery", "CLIP ViT-L-14 image tower\n6,378 vectors, L2-normalised"),
                ("Store the index", "image.faiss + image_ids.npy\nmeta.json records model and dim"),
            ],
            "artefact": "index/clip/  ·  768-d",
            "query": [
                ("Encode the query", "CLIP text tower, 77 tokens\nno annotations are ever read"),
                ("Inner product", "against 6,378 image vectors\ninner product = cosine similarity"),
                ("Cut the list", "drop every score ≤ 0\nkeep the top k"),
            ],
            "verdict": "Sees the scene, not the joke: overlaid text is invisible to it.",
        },
        {
            "tone": "nlp", "name": "B · clip_nlp",
            "tag": "Human annotations, fused with the pixels",
            "build": [
                ("Four fields per image", "scene · meaning · metaphor · title\n1,968 literal metaphor rows dropped"),
                ("Write two stores", "dense doc → BGE-small, 384-d\nsame fields → FTS5 BM25 table"),
            ],
            "artefact": "index/clip_nlp/ + nlp_docs FTS5",
            "query": [
                ("Three candidate lists", "CLIP · BGE doc · BM25 (1.0/1.6/1.8/0.6)\nlexicon expansion enters BM25 only"),
                ("Min-max fuse", "0.5 CLIP + 0.3 doc + 0.2 BM25\nempty signals drop out and renormalise"),
                ("Rerank the top 60", "cross-encoder on (query, document)\nfused ← 0.7·fused + 0.3·rerank"),
            ],
            "verdict": "Wins wherever the intent lives — metaphor 0.912, intent 0.739.",
        },
        {
            "tone": "vlm", "name": "C · vlm",
            "tag": "A VLM writes the annotation, offline",
            "build": [
                ("Caption the gallery offline", "Qwen2.5-VL-3B → JSON per image\nscene · OCR · meaning · metaphors"),
                ("Embed the descriptions", "doc_text → BGE-small, 384-d\ndescriptions.jsonl is the resume cache"),
            ],
            "artefact": "index/vlm/  ·  384-d",
            "query": [
                ("Encode the query", "BGE with the query instruction\nthe VLM is never loaded here"),
                ("Search and fuse", "doc 0.8 + CLIP 0.2, min-max\nno CLIP → the doc weight becomes 1.0"),
                ("Return the ranking", "no rerank by default\nprovenance checked before serving"),
            ],
            "verdict": "Loses fine detail, recovers part of the intent — net ≈ A.",
        },
    ]

    xs = [0.55, 4.15, 7.75]
    cw = 3.36
    inner = (cw - 0.18) * 72.0

    for c, spec in enumerate(cols):
        x = xs[c]
        tone = spec["tone"]
        ax.add_patch(Rectangle((x, 2.80), cw, 4.42, facecolor="#FBFCFE", edgecolor=RULE, lw=0.6, zorder=0))
        ax.add_patch(Rectangle((x, 7.22), cw, 0.04, facecolor=C[tone]["edge"], edgecolor="none", zorder=1))

        ax.text(x + 0.14, 7.12, spec["name"], fontsize=9.2, fontweight="bold", color=C[tone]["edge"], va="top")
        ax.text(x + 0.14, 6.90, wrap(spec["tag"], inner, 5.7), fontsize=5.7, color=BODY, va="top",
                linespacing=1.35)

        ax.text(x + 0.14, 6.62, "BUILD  ·  once, offline", fontsize=5.6, fontweight="bold",
                color=MUTED, va="top")
        y = 6.40
        for i, (title, body) in enumerate(spec["build"]):
            box(ax, x + 0.14, y - 0.44, cw - 0.28, 0.44, title, body,
                tone=tone, title_size=6.2, body_size=5.1, align="top", pad=0.08)
            if i < len(spec["build"]) - 1:
                arrow(ax, (x + cw / 2, y - 0.44), (x + cw / 2, y - 0.54), lw=0.7)
            y -= 0.54
        chip(ax, x + 0.14, y - 0.16, spec["artefact"], tone=tone, size=5.2)

        ax.text(x + 0.14, 5.04, "QUERY  ·  every request", fontsize=5.6, fontweight="bold",
                color=MUTED, va="top")
        qy = 4.82
        for i, (title, body) in enumerate(spec["query"]):
            box(ax, x + 0.14, qy - 0.46, cw - 0.28, 0.46, title, body,
                tone="plain", title_size=6.2, body_size=5.1, align="top", pad=0.08,
                title_color=C[tone]["edge"])
            if i < len(spec["query"]) - 1:
                arrow(ax, (x + cw / 2, qy - 0.46), (x + cw / 2, qy - 0.56), lw=0.7)
            qy -= 0.56

        ax.add_patch(Rectangle((x + 0.14, 2.82), cw - 0.28, 0.34,
                               facecolor=C[tone]["fill"], edgecolor=C[tone]["edge"], lw=0.6))
        ax.text(x + 0.22, 2.99, wrap(spec["verdict"], (cw - 0.46) * 72.0, 5.2), fontsize=5.2,
                color=C[tone]["edge"], va="top", linespacing=1.35)

    # ------------------------------------------------------------- table
    hy = 2.62
    heads = ["Comparison", "A · clip", "B · clip_nlp", "C · vlm"]
    rows = [
        ("What it matches", "The pixels, nothing else.", "Human annotation text + pixels.",
         "A VLM's paraphrase + pixels."),
        ("Text per image", "none", "scene · meaning · metaphor · title",
         "scene · OCR · meaning · metaphors"),
        ("Indexes", "index/clip · 6,378 × 768-d", "+ index/clip_nlp · 384-d + FTS5",
         "+ index/vlm · 384-d"),
        ("Signals at query time", "1 — CLIP text tower", "3 — CLIP 0.5 · doc 0.3 · BM25 0.2",
         "2 — doc 0.8 · CLIP 0.2"),
        ("Post-processing", "drop score ≤ 0, keep top k", "cross-encoder on top 60, α = 0.3",
         "none — rerank disabled"),
        ("Candidates fetched", "max(2k, k)", "max(5k, 100) per signal", "max(3k, 100) per signal"),
        ("One-off cost · new memes", "minutes · generalises", "human annotation · no cold start",
         "2 h 35 min GPU · generalises"),
        ("Latency p50 / p95", "14.4 / 20.8 ms", "225.6 / 285.9 ms", "35.3 / 41.6 ms"),
        ("mAP on the frozen 240", "0.4395", "0.7608", "0.4519"),
    ]

    colx = [0.55, 2.52, 5.42, 8.32]
    row_h = 0.205
    ax.add_patch(Rectangle((0.55, hy - 0.02), W - 1.10, 0.20, facecolor="#E9EEF3", edgecolor="none"))
    for c, head in enumerate(heads):
        ax.text(colx[c], hy + 0.08, head, fontsize=5.9, fontweight="bold", color=INK, va="center")

    for r, row in enumerate(rows):
        y = hy - row_h * (r + 1) + 0.06
        if r % 2 == 1:
            ax.add_patch(Rectangle((0.55, y - 0.06), W - 1.10, row_h, facecolor="#F7F9FB", edgecolor="none"))
        last = r == len(rows) - 1
        ax.text(colx[0], y, row[0], fontsize=5.6, color=BODY, va="center")
        for c in (1, 2, 3):
            ax.text(colx[c], y, row[c], fontsize=5.6,
                    color=INK if (last or c == 2) else BODY,
                    fontweight="bold" if (last and c == 2) else "normal",
                    va="center")

    page_foot(
        ax,
        "Weights from configs/default.yaml; behaviour read from strategies/*.py; metrics from "
        "data/reports/compare.json.",
    )
    return fig


# --------------------------------------------------------------------------- #
# page 3 - offline build flow
# --------------------------------------------------------------------------- #
def page_build(fig, ax, number: int):
    page_head(
        ax,
        "Offline build flow",
        "Five commands turn the Meme-Cap download into the gallery and the three indexes. "
        "Every stage is keyed by a content hash, so a stale artefact is refused, never served.",
        number,
    )

    stages = [
        {
            "cmd": "tsr prepare",
            "when": "once  ·  ~15 min  ·  network",
            "tone": "data",
            "steps": [
                ("Download", "annotations (GitHub raw JSON) + memecap.zip\n(HF mirror, 1.7 GB, resumable)"),
                ("Normalise", "PNG → RGB on white · longest side 768\nJPEG q=90 · dHash stored per record"),
                ("De-duplicate", "SHA-1 of the source PNG\n6,382 annotated → 6,378 unique\n5,820 trainval · 558 test"),
            ],
            "out": ["manifest.jsonl", "manifest.meta.json", "images/ 492 MiB"],
            "guard": "An existing manifest is reused unless --force; extraction is skipped when the archive size matches.",
        },
        {
            "cmd": "tsr build --strategy gallery",
            "when": "once  ·  ~10 min on CPU",
            "tone": "clip",
            "steps": [
                ("Insert rows", "manifest order, one transaction\nimage_id order must match exactly"),
                ("Verify", "re-hash the stored annotations\nand compare with the manifest"),
                ("Encode", "CLIP ViT-L-14 image tower\nbatch 64 · L2-normalised"),
            ],
            "out": ["gallery.db", "index/clip/image.faiss", "index/clip/meta.json"],
            "guard": "Refuses a non-contiguous id range, a count mismatch, or annotations that drifted from the manifest.",
        },
        {
            "cmd": "tsr build --strategy clip_nlp",
            "when": "~5 min on CPU",
            "tone": "nlp",
            "steps": [
                ("Documents", "scene · meaning · metaphor · title\none dense doc, four BM25 columns"),
                ("BM25", "FTS5 external-content + triggers\ncolumns 1.0 / 1.6 / 1.8 / 0.6"),
                ("Encode", "BGE-small-en-v1.5, CLS pooling\n384-d, max 512 tokens"),
            ],
            "out": ["nlp_docs + FTS5", "index/clip_nlp/doc.faiss", "index/clip_nlp/meta.json"],
            "guard": "Metaphor rows meaning 'not related to the meme context' (1,968) are excluded from the document.",
        },
        {
            "cmd": "tsr vlm-caption",
            "when": "~2 h 35 min  ·  GPU host  ·  resumable",
            "tone": "vlm",
            "steps": [
                ("Generate", "Qwen2.5-VL-3B-Instruct · bf16\nbatch 8 · 448 new tokens max"),
                ("Parse", "JSON repair for stray quotes,\nmissing commas, truncated tails"),
                ("Store", "upsert vlm_docs + append to the\nJSONL cache, fsync per line"),
            ],
            "out": ["vlm/descriptions.jsonl", "vlm_docs rows", "0 errors · 6,378 / 6,378"],
            "guard": "Resume key is image_pk + sha1; backend, model and prompt_version are stored on every row.",
        },
        {
            "cmd": "tsr build --strategy vlm",
            "when": "~5 min on CPU",
            "tone": "vlm",
            "steps": [
                ("Filter", "status ok and the current\nprovenance triple only"),
                ("Encode", "VLM document text → BGE-small\n384-d"),
                ("Record", "meta.json stores backend,\nmodel and prompt_version"),
            ],
            "out": ["index/vlm/doc.faiss", "index/vlm/meta.json"],
            "guard": "A different generator, or coverage under 95% of the gallery, makes strategy C read as unavailable.",
        },
    ]

    top = 7.20
    row_h = 1.00
    gap = 0.10
    left = 0.55
    rail_w = (2.68 - 0.96) * 72.0

    for i, st in enumerate(stages):
        y = top - row_h - i * (row_h + gap)
        tone = st["tone"]
        ax.add_patch(
            Rectangle(
                (left, y), W - 2 * left, row_h,
                facecolor="#FCFDFE" if i % 2 == 0 else PAPER,
                edgecolor=RULE, lw=0.5, zorder=0,
            )
        )
        ax.add_patch(
            Circle((left + 0.23, y + row_h - 0.23), 0.105,
                   facecolor=C[tone]["edge"], edgecolor="none", zorder=2)
        )
        ax.text(left + 0.23, y + row_h - 0.23, str(i + 1), fontsize=7.0, fontweight="bold",
                color="white", ha="center", va="center", zorder=3)
        ax.text(left + 0.41, y + row_h - 0.13, st["cmd"], fontsize=6.4, fontweight="bold",
                color=INK, family="DejaVu Sans Mono", va="top")
        ax.text(left + 0.41, y + row_h - 0.33, st["when"], fontsize=5.3, color=MUTED, va="top")
        ax.text(left + 0.41, y + row_h - 0.50,
                wrap(st["guard"], rail_w, 4.7),
                fontsize=4.7, color=MUTED, style="italic", va="top", linespacing=1.36)

        sx, step_w, sg = 2.76, 1.86, 0.24
        for j, (title, body) in enumerate(st["steps"]):
            x = sx + j * (step_w + sg)
            box(ax, x, y + 0.19, step_w, 0.62, title, body,
                tone="plain", title_size=6.1, body_size=5.2, align="top", pad=0.07)
            if j:
                arrow(ax, (x - sg + 0.03, y + row_h / 2), (x - 0.03, y + row_h / 2), lw=0.7)

        ox = 9.14
        ax.text(ox, y + row_h - 0.15, "writes", fontsize=5.1, color=MUTED, va="top")
        for k, name in enumerate(st["out"]):
            chip(ax, ox, y + row_h - 0.42 - k * 0.20, name, tone=tone, size=4.9)

        if i < len(stages) - 1:
            arrow(ax, (left + 0.23, y), (left + 0.23, y - gap), lw=0.7, color=RULE)

    ax.text(
        left, 1.52,
        "Dependency order: gallery before clip_nlp (it reuses the CLIP index) and before vlm (it requires vlm_docs).  "
        "--strategy all runs clip → clip_nlp → vlm and skips C with a warning when models.vlm.backend is none.  "
        "B and C never touch images.id: they only rewrite their own rows and vector files.",
        fontsize=5.9, color=BODY, va="top", linespacing=1.45,
    )
    chip(ax, left, 1.12,
         "python -m tsr stats --json    →    readiness, counts and model strings for every stage",
         tone="data", size=5.7, mono=False)

    page_foot(
        ax,
        "Command order and guard rails read from cli.py and pipeline.py; timings measured on a 12 GB "
        "RTX 4070 SUPER (captioning) and a desktop CPU (index builds).",
    )
    return fig


# --------------------------------------------------------------------------- #
# page 4 - online query flow
# --------------------------------------------------------------------------- #
def page_query(fig, ax, number: int):
    page_head(
        ax,
        "Online query flow",
        "One search request, from the page to the ranked payload. Models and indexes load on the "
        "first query of each strategy, never at process start.",
        number,
    )

    actors = [
        ("page", "Page (web/*.js)", 1.34, "serve"),
        ("api", "FastAPI route", 3.14, "serve"),
        ("manager", "StrategyManager", 4.94, "data"),
        ("strategy", "Strategy A/B/C", 6.74, "clip"),
        ("encoders", "Encoders", 8.54, "nlp"),
        ("index", "FAISS + SQLite", 10.34, "data"),
    ]
    top_y, bot_y, half = 6.42, 2.30, 0.79

    for _key, label, x, tone in actors:
        ax.add_patch(
            FancyBboxPatch(
                (x - half, top_y), 2 * half, 0.34,
                boxstyle="round,pad=0,rounding_size=0.05",
                facecolor=C[tone]["fill"], edgecolor=C[tone]["edge"], lw=0.8, zorder=2,
            )
        )
        ax.text(x, top_y + 0.17, label, fontsize=6.3, fontweight="bold",
                color=C[tone]["edge"], ha="center", va="center", zorder=3)
        ax.plot([x, x], [bot_y, top_y], color=RULE, lw=0.7, ls=(0, (1.5, 2.2)), zorder=1)

    x = {key: pos for key, _label, pos, _tone in actors}
    mask = {"facecolor": PAPER, "edgecolor": "none", "pad": 0.8, "alpha": 0.92}

    msgs = [
        ("page", "api", "1  GET /api/strategies/{key}/search?q=…&k=", 5.90, "solid"),
        ("api", "manager", "2  search(key, query, k)", 5.50, "solid"),
        ("manager", "strategy", "3  get(key) → 404 if unknown", 5.10, "solid"),
        ("strategy", "encoders", "4  encode the query (CLIP text / BGE)", 4.40, "solid"),
        ("strategy", "index", "5  FAISS search, N = max(mult·k, 100)", 4.00, "solid"),
        ("index", "strategy", "6  ids + scores (cosine similarity)", 3.60, "dashed"),
        ("strategy", "manager", "7  hits + score_components + latency", 3.20, "dashed"),
        ("manager", "api", "8  enrich from gallery.db", 2.80, "dashed"),
        ("api", "page", "9  SearchResponse JSON", 2.40, "dashed"),
    ]
    for a, b, text, y, style in msgs:
        x0, x1 = x[a], x[b]
        arrow(ax, (x0, y), (x1, y), lw=0.85, ls=style, style="-|>",
              color=MUTED if style == "solid" else "#8A9AA8")
        if x1 > x0:
            ax.text(x0 + 0.07, y + 0.06, text, fontsize=5.6, color=INK,
                    ha="left", va="bottom", zorder=4, bbox=mask)
        else:
            ax.text(x0 - 0.07, y + 0.06, text, fontsize=5.6, color=INK,
                    ha="right", va="bottom", zorder=4, bbox=mask)

    # readiness note attached to the strategy lifeline, between steps 3 and 4
    ax.add_patch(
        FancyBboxPatch(
            (5.82, 4.60), 1.84, 0.34,
            boxstyle="round,pad=0,rounding_size=0.04",
            facecolor="#FFF8E6", edgecolor="#C9A227", lw=0.6, zorder=2,
        )
    )
    ax.text(5.90, 4.83, "ensure_loaded(): verify meta.json,", fontsize=5.0, color="#8A6D1B", va="top", zorder=3)
    ax.text(5.90, 4.70, "load the index once → 503 if not ready", fontsize=5.0, color="#8A6D1B", va="top", zorder=3)

    # -------------------------------------------------------------- panels
    panel_y, panel_h = 1.02, 0.98
    box(
        ax, 0.55, panel_y, 3.20, panel_h, "Failure semantics",
        "404  unknown strategy key\n503  known, but the index is missing or stale\n"
        "422  invalid settings payload\n/compare keeps a failing column as {ready: false}",
        tone="eval", title_size=6.3, body_size=5.1, align="top",
    )
    box(
        ax, 3.90, panel_y, 3.20, panel_h, "What loads when",
        "startup      config, SQLite, strategy objects\n"
        "first query  CLIP or BGE weights, FAISS files, reranker\n"
        "never        the VLM stack (captioning is offline)\n"
        "an RLock serialises searches across threads",
        tone="data", title_size=6.3, body_size=5.1, align="top",
    )

    bx, bw = 7.25, 3.90
    ax.add_patch(
        FancyBboxPatch(
            (bx, panel_y), bw, panel_h,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=C["clip"]["fill"], edgecolor=C["clip"]["edge"], lw=0.8, zorder=2,
        )
    )
    ax.text(bx + 0.10, panel_y + panel_h - 0.10,
            "Measured latency  ·  p50 / p95 over the 240 frozen queries",
            fontsize=6.3, fontweight="bold", color=C["clip"]["edge"], va="top", zorder=3)

    scale = 1.72 / 226.0
    for i, (name, p50, p95) in enumerate(
        [("A · clip", 14.4, 20.8), ("B · clip_nlp", 225.6, 285.9), ("C · vlm", 35.3, 41.6)]
    ):
        y = panel_y + panel_h - 0.36 - i * 0.20
        ax.text(bx + 0.10, y + 0.05, name, fontsize=5.2, color=BODY, va="center", zorder=3)
        ax.add_patch(Rectangle((bx + 0.86, y), p50 * scale, 0.10,
                               facecolor=C["clip"]["edge"], edgecolor="none", alpha=0.75, zorder=3))
        ax.add_patch(Rectangle((bx + 0.86, y), p95 * scale, 0.10,
                               facecolor="none", edgecolor=C["clip"]["edge"], lw=0.4, zorder=4))
        ax.text(bx + 0.92 + p95 * scale, y + 0.05, f"{p50:.0f} / {p95:.0f} ms",
                fontsize=5.0, color=BODY, va="center", zorder=3)
    ax.text(bx + 0.10, panel_y + 0.10,
            "bar = p50   ·   outline = p95   ·   reranking dominates B",
            fontsize=4.8, color=MUTED, va="bottom", zorder=3)

    page_foot(
        ax,
        "Route order from api.py, readiness checks from service.py and strategies/vlm.py; latencies from "
        "data/reports/compare.json.",
    )
    return fig


# --------------------------------------------------------------------------- #
# page 5 - evaluation flow
# --------------------------------------------------------------------------- #
def page_eval(fig, ax, number: int):
    page_head(
        ax,
        "Evaluation and reporting flow",
        "A frozen, leakage-checked benchmark is run through every ready strategy; the same report "
        "files drive the CLI and the /evaluation page.",
        number,
    )

    col_w, col_y, col_h = 3.32, 3.95, 3.20
    xs = [0.55, 4.18, 7.81]
    columns = [
        ("1  BUILD THE BENCHMARK", C["eval"]["edge"], [
            ("Draft", "from the 558 test images\n80 visual · 80 intent · 80 metaphor"),
            ("Rewrite", "one reviewed rewrite per draft\ndata/eval/rewrites/*.json"),
            ("Leakage gate", "8-gram · whole short annotation\n≥ 5-token copy · Jaccard ≥ 0.6"),
            ("Freeze", "memecap_eval_v1 · 240 queries\nsplit and annotation snapshot checked"),
        ]),
        ("2  MEASURE EVERY STRATEGY", C["clip"]["edge"], [
            ("Same gallery", "one image table, one CLIP index\nall three strategies search it"),
            ("Graded qrels", "grade 2 = the source meme\ngrade 1 = deterministic rules only"),
            ("Metrics", "P@1/5/10 · R@1/5/10 · mAP · MRR\nnDCG@10 · median rank · misses"),
            ("Depth", "max(top_k, largest relevant set, 100)\nlatency reported as p50 / p95"),
        ]),
        ("3  COMPARE AND PUBLISH", C["nlp"]["edge"], [
            ("Paired bootstrap", "1,000 resamples, seed 42\npaired per-query differences"),
            ("Correction", "Holm–Bonferroni over\n3 pairs × 5 metrics · p ≥ 0.001"),
            ("Ablations", "7 single-change variants\nprompt ensemble · B channels · C fusion"),
            ("Reports", "compare.json · compare.html · failures.html\n/api/eval/summary → /evaluation"),
        ]),
    ]

    for c, (title, colour, items) in enumerate(columns):
        x0 = xs[c]
        ax.add_patch(Rectangle((x0, col_y), col_w, col_h, facecolor="#FAFCFD", edgecolor=RULE, lw=0.6))
        ax.text(x0 + 0.16, col_y + col_h - 0.16, title, fontsize=7.0, fontweight="bold",
                color=colour, va="top")
        step = (col_h - 0.55 - 0.52 - 0.12) / 3
        for i, (name, body) in enumerate(items):
            ytop = col_y + col_h - 0.55 - i * step
            box(ax, x0 + 0.16, ytop - 0.52, col_w - 0.32, 0.52, name, body,
                tone="plain", title_size=6.2, body_size=5.1, align="top")

    for a, b in ((xs[0] + col_w, xs[1]), (xs[1] + col_w, xs[2])):
        arrow(ax, (a + 0.03, col_y + col_h / 2), (b - 0.03, col_y + col_h / 2), lw=1.0)

    dy = 3.32
    ax.add_patch(Rectangle((0.55, dy), W - 1.10, 0.46,
                           facecolor="#FDF7F7", edgecolor=C["eval"]["edge"], lw=0.6))
    ax.text(0.70, dy + 0.23, "LEAKAGE DIAGNOSTIC", fontsize=6.2, fontweight="bold",
            color=C["eval"]["edge"], va="center")
    ax.text(2.30, dy + 0.23,
            "The untouched human annotations are also run as queries and reported separately "
            "(B +0.139 mAP, A −0.085, C −0.193). Leaky by construction — never mixed into the main table.",
            fontsize=5.7, color=BODY, va="center")

    ax.text(0.55, 3.20, "MEASURED RESULT  ·  240 frozen queries, identical gallery and retrieval depth",
            fontsize=6.8, fontweight="bold", color=INK, va="top")

    rows = [
        ("Strategy", "mAP [95% CI]", "MRR", "nDCG@10", "P@1", "R@10", "p50"),
        ("A · clip", "0.4395  [0.385, 0.498]", "0.4461", "0.4744", "0.358", "0.607", "14 ms"),
        ("B · clip_nlp", "0.7608  [0.713, 0.808]", "0.7647", "0.7804", "0.700", "0.870", "226 ms"),
        ("C · vlm", "0.4519  [0.399, 0.507]", "0.4632", "0.4881", "0.383", "0.626", "35 ms"),
    ]
    colx = [0.55, 1.85, 4.05, 5.15, 6.20, 7.05, 7.95]
    row_h = 0.26
    for r, row in enumerate(rows):
        y = 2.86 - r * row_h
        if r == 0:
            ax.add_patch(Rectangle((0.55, y - 0.05), 7.95, row_h, facecolor="#E9EEF3", edgecolor="none"))
        elif r == 2:
            ax.add_patch(Rectangle((0.55, y - 0.05), 7.95, row_h, facecolor="#F2F6FB", edgecolor="none"))
        for c, value in enumerate(row):
            ax.text(colx[c], y + 0.08, value, fontsize=5.9,
                    color=INK if (r == 0 or c == 1) else BODY,
                    fontweight="bold" if (r == 0 or (r == 2 and c == 1)) else "normal",
                    va="center")

    ax.text(8.95, 2.94,
            "B − A   mAP +0.321  [0.270, 0.371]   p_holm = 0.015\n"
            "B − C   mAP +0.309  [0.255, 0.363]   p_holm = 0.015\n"
            "C − A   mAP +0.012  [−0.038, 0.064]   not significant",
            fontsize=5.7, color=BODY, va="top", linespacing=1.65)

    ax.text(0.55, 1.82,
            "Per-category mAP (visual / intent / metaphor)   ·   A  0.315 / 0.444 / 0.559      "
            "B  0.631 / 0.739 / 0.912      C  0.247 / 0.539 / 0.570",
            fontsize=5.8, color=BODY, va="top")
    ax.text(0.55, 1.58,
            "Ablations (mAP)   ·   B − BM25 0.704 · B − dense 0.704 · B − CLIP 0.780 · B − reranker 0.689   ·   "
            "C pure VLM 0.392 · C + CLIP 0.35 0.472",
            fontsize=5.8, color=BODY, va="top")

    ax.add_patch(Rectangle((0.55, 0.90), W - 1.10, 0.44, facecolor="#F7FAFC", edgecolor=RULE, lw=0.6))
    ax.text(0.70, 1.12, "Reading", fontsize=6.2, fontweight="bold", color=C["eval"]["edge"], va="center")
    ax.text(1.62, 1.12,
            "B wins by +0.32 mAP over both annotation-free strategies; C and A are statistically "
            "indistinguishable, so a 3B VLM is a cheap approximation of the annotation signal, not a substitute.",
            fontsize=5.7, color=BODY, va="center")

    page_foot(
        ax,
        "Sources: data/eval/memecap_eval_v1.{json,qrels.jsonl} and data/reports/compare.json "
        "(1,000 bootstrap resamples, seed 42).",
    )
    return fig


# --------------------------------------------------------------------------- #
def main() -> int:
    # An optional argument lets a locked output (open in a viewer) be bypassed.
    out = (Path(sys.argv[1]) if len(sys.argv) > 1
           else Path(__file__).with_name("system-schematic.pdf"))
    with PdfPages(out) as pdf:
        for i, draw in enumerate(
            (page_architecture, page_strategies, page_build, page_query, page_eval), start=1
        ):
            fig = plt.figure(figsize=(W, H))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, W)
            ax.set_ylim(0, H)
            ax.axis("off")
            fig.patch.set_facecolor(PAPER)
            draw(fig, ax, i)
            pdf.savefig(fig, facecolor=PAPER)
            plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
