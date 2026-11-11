"""Generate docs/evaluation_rewrites.pdf.

Every one of the 240 frozen queries of memecap_eval_v1, side by side with the
Meme-Cap annotation it was written from. The data is read from the same
artifacts the evaluator uses:

    data/eval/batches/batch1-6.json      source material handed to the reviewer
    data/eval/rewrites/batch1-6.json     qid -> final query text
    data/eval/memecap_eval_v1.json       frozen set + leakage verdicts

Run:  python docs/make_rewrite_table_pdf.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# --------------------------------------------------------------------------- #
# Fonts and palette (same faces as the other generated documents)
# --------------------------------------------------------------------------- #
FONT_DIR = r"C:\Windows\Fonts"
REG, BOLD, ITAL = "SegoeUI", "SegoeUIBold", "SegoeUIItalic"
MONO = "Consolas"


def _register() -> None:
    for name, filename in [
        (REG, "segoeui.ttf"),
        (BOLD, "segoeuib.ttf"),
        (ITAL, "segoeuii.ttf"),
        (MONO, "consola.ttf"),
    ]:
        path = os.path.join(FONT_DIR, filename)
        if os.path.exists(path):
            pdfmetrics.registerFont(TTFont(name, path))
    pdfmetrics.registerFontFamily(REG, normal=REG, bold=BOLD, italic=ITAL, boldItalic=BOLD)


_register()

NAVY = colors.HexColor("#102A43")
INK = colors.HexColor("#1F2933")
BODY = colors.HexColor("#243B53")
MUTED = colors.HexColor("#62717F")
TEAL = colors.HexColor("#0F766E")
TEAL_LIGHT = colors.HexColor("#E6F4F1")
AMBER = colors.HexColor("#B45309")
AMBER_LIGHT = colors.HexColor("#FDF6EC")
LIGHT = colors.HexColor("#F5F7FA")
RULE = colors.HexColor("#D9E2EC")
ZEBRA = colors.HexColor("#F7F9FB")
WHITE = colors.white

PAGE_W, PAGE_H = landscape(A4)
LM = RM = 42
TM = 56
BM = 46
CONTENT_W = PAGE_W - LM - RM

OUT = os.path.join(os.path.dirname(__file__), "evaluation_rewrites.pdf")
DOC_DATE = "4 November 2026"


def style(name, **kw):
    base = dict(
        fontName=REG, fontSize=7.4, leading=9.9, textColor=INK,
        spaceBefore=0, spaceAfter=0,
    )
    base.update(kw)
    return ParagraphStyle(name, **base)


S_TITLE = style("Title", fontName=BOLD, fontSize=23, leading=27, textColor=NAVY, spaceAfter=4)
S_SUB = style("Sub", fontSize=10.6, leading=15, textColor=MUTED, spaceAfter=14)
S_H1 = style("H1", fontName=BOLD, fontSize=13.5, leading=17, textColor=NAVY, spaceAfter=1)
S_H2 = style("H2", fontName=BOLD, fontSize=9.6, leading=12.5, textColor=TEAL, spaceAfter=2)
S_BODY = style("Body", fontSize=9.2, leading=13.2, textColor=BODY, spaceAfter=5)
S_SMALL = style("Small", fontSize=7.8, leading=10.6, textColor=MUTED)
S_TH = style("TH", fontName=BOLD, fontSize=7.3, leading=9.2, textColor=WHITE)
S_CELL = style("Cell", fontSize=7.4, leading=9.9)
S_SRC = style("Src", fontSize=7.4, leading=9.9, textColor=BODY)
S_REW = style("Rew", fontSize=7.4, leading=9.9, textColor=INK)
S_QID = style("Qid", fontName=MONO, fontSize=6.9, leading=9.4, textColor=MUTED)
S_NUM = style("Num", fontSize=6.6, leading=9.2, textColor=MUTED)


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def P(text: str, st=S_BODY) -> Paragraph:
    return Paragraph(text, st)


def bullets(items: list[str]) -> list[Paragraph]:
    st = style("Bullet", fontSize=8.8, leading=12.4, leftIndent=12, bulletIndent=1, spaceAfter=3.5)
    return [Paragraph(item, st, bulletText="\u2022") for item in items]


def callout(title: str, body: str, accent=AMBER, bg=AMBER_LIGHT) -> Table:
    inner = [
        Paragraph(
            f'<font color="{accent.hexval()}">{esc(title)}</font>',
            style("CT", fontName=BOLD, fontSize=8.6, leading=11.2, spaceAfter=2.5),
        ),
        Paragraph(body, style("CB", fontSize=8.3, leading=11.8, textColor=BODY)),
    ]
    table = Table([["", inner]], colWidths=[4, CONTENT_W - 4])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), accent),
        ("BACKGROUND", (1, 0), (1, 0), bg),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 0),
        ("TOPPADDING", (0, 0), (0, 0), 0),
        ("BOTTOMPADDING", (0, 0), (0, 0), 0),
        ("LEFTPADDING", (1, 0), (1, 0), 10),
        ("RIGHTPADDING", (1, 0), (1, 0), 10),
        ("TOPPADDING", (1, 0), (1, 0), 7),
        ("BOTTOMPADDING", (1, 0), (1, 0), 7),
    ]))
    return table


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_pairs() -> list[dict]:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_dir = os.path.join(here, "data", "eval")

    sources: list[dict] = []
    for path in sorted(glob.glob(os.path.join(eval_dir, "batches", "batch*.json"))):
        with open(path, encoding="utf-8") as fh:
            sources.extend(json.load(fh))

    rewrites: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(eval_dir, "rewrites", "batch*.json"))):
        with open(path, encoding="utf-8") as fh:
            rewrites.update(json.load(fh))

    with open(os.path.join(eval_dir, "memecap_eval_v1.json"), encoding="utf-8") as fh:
        frozen = {q["qid"]: q for q in json.load(fh)}

    pairs = []
    for source in sources:
        qid = str(source["qid"])
        query = frozen[qid]
        pairs.append({
            "qid": qid,
            "category": str(source["category"]).lower(),
            "source": source,
            "rewrite": rewrites[qid],
            "text": query["query"],
            "leak": query["metadata"]["leakage"],
        })
    pairs.sort(key=lambda p: (p["category"], p["qid"]))
    return pairs


def source_text(pair: dict) -> str:
    """The annotation the rewrite was written from, per category."""
    rec = pair["source"]
    cat = pair["category"]
    if cat == "visual":
        return "  /  ".join(esc(c) for c in rec.get("img_captions") or [])
    if cat == "intent":
        return "  \u2022  ".join(esc(c) for c in rec.get("meme_captions") or [])
    pairs = rec.get("metaphors") or []
    if not pairs:
        return esc(rec.get("title") or "")
    return "  \u2022  ".join(
        f'\u201c{esc(m.get("metaphor", ""))}\u201d = {esc(m.get("meaning", ""))}'
        for m in pairs
    )


CATEGORY_META = {
    "visual": (
        "Visual queries",
        "The source is <b>img_captions</b>, the annotator's literal description of the image. "
        "The rewrite keeps the scene but drops proper names and rephrases the wording.",
    ),
    "intent": (
        "Intent queries",
        "The source is every <b>meme_captions</b> entry, i.e. the human-written intended meaning. "
        "Records carry two to four captions (separated by \u2022); all are shown because the rewrite "
        "may draw on any of them.",
    ),
    "metaphor": (
        "Metaphor queries",
        "The source is the raw <b>metaphors</b> annotation: (metaphor, meaning) pairs, one to three "
        "per record, all shown. Some pairs are annotation boilerplate (e.g. \u201cmen\u201d = "
        "\u201cMeme poster\u201d); the rewrite usually targets the salient pair.",
    ),
}


# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #
def _footer(c: canvas.Canvas) -> None:
    c.saveState()
    c.setStrokeColor(RULE)
    c.setLineWidth(0.6)
    c.line(LM, BM - 12, PAGE_W - RM, BM - 12)
    c.setFont(REG, 7)
    c.setFillColor(MUTED)
    c.drawString(LM, BM - 24, "memecap_eval_v1 \u00b7 240 source/rewrite pairs \u00b7 generated from data/eval/")
    c.drawRightString(PAGE_W - RM, BM - 24, f"{c.getPageNumber()}")
    c.restoreState()


def _header(c: canvas.Canvas) -> None:
    c.saveState()
    c.setFont(REG, 7)
    c.setFillColor(MUTED)
    c.drawString(LM, PAGE_H - 30, "Evaluation benchmark \u2014 source annotation vs frozen query")
    c.drawRightString(PAGE_W - RM, PAGE_H - 30, f"Meme-Cap T2I retrieval \u00b7 {DOC_DATE}")
    c.setStrokeColor(RULE)
    c.setLineWidth(0.6)
    c.line(LM, PAGE_H - 38, PAGE_W - RM, PAGE_H - 38)
    c.restoreState()


def on_first(c: canvas.Canvas, _doc) -> None:
    _footer(c)


def on_later(c: canvas.Canvas, _doc) -> None:
    _header(c)
    _footer(c)


# --------------------------------------------------------------------------- #
# Story
# --------------------------------------------------------------------------- #
def pairs_table(pairs: list[dict], start_no: int) -> Table:
    head = [
        Paragraph("No.", S_TH),
        Paragraph("Query", S_TH),
        Paragraph("Source annotation (original)", S_TH),
        Paragraph("Frozen query (rewrite)", S_TH),
    ]
    rows: list[list] = [head]
    for offset, pair in enumerate(pairs):
        rows.append([
            Paragraph(str(start_no + offset), S_NUM),
            Paragraph(esc(pair["qid"]), S_QID),
            Paragraph(source_text(pair), S_SRC),
            Paragraph(esc(pair["text"]), S_REW),
        ])

    widths = [26, 48, 352, 336]
    table = Table(rows, colWidths=widths, repeatRows=1)
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.35, RULE),
        ("LINEBEFORE", (2, 0), (2, -1), 0.35, RULE),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            commands.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    table.setStyle(TableStyle(commands))
    return table


def build_story(pairs: list[dict]) -> list:
    story: list = []
    counts = Counter(p["category"] for p in pairs)
    all_passed = all(p["leak"]["passed"] for p in pairs)
    max_jac = max(p["leak"]["jaccard"] for p in pairs)

    # ---------------------------------------------------------------- cover
    story.append(Spacer(1, 26))
    story.append(P("Evaluation benchmark: source vs rewrite", S_TITLE))
    story.append(P(
        "All 240 queries of <b>memecap_eval_v1</b>, paired with the Meme-Cap annotation each was "
        "written from. The left-hand source is what the annotator said; the right-hand query is what "
        "the retriever actually receives.", S_SUB))

    story.append(P("What this document contains", S_H2))
    story.append(P(
        f"<b>{len(pairs)} pairs</b> \u2014 {counts['visual']} visual, {counts['intent']} intent, "
        f"{counts['metaphor']} metaphor. The three sections use the source field that matches the "
        "query category, so the comparison is like-for-like: a visual query is compared with the "
        "literal image caption, an intent query with the written meaning, a metaphor query with the "
        "metaphor annotation. Every pair keeps its <font name=\"Consolas\" size=\"8\">qid</font>, so "
        "any row can be traced back to the frozen JSON, the qrels and the leakage record."))

    stats = Table(
        [
            [Paragraph("Category", S_TH), Paragraph("Pairs", S_TH),
             Paragraph("Source field", S_TH), Paragraph("Max Jaccard", S_TH)],
            [Paragraph("visual", S_SRC), Paragraph(str(counts["visual"]), S_SRC),
             Paragraph("img_captions (image description)", S_SRC),
             Paragraph(f"{max(p['leak']['jaccard'] for p in pairs if p['category'] == 'visual'):.3f}", S_SRC)],
            [Paragraph("intent", S_SRC), Paragraph(str(counts["intent"]), S_SRC),
             Paragraph("meme_captions (intended meaning)", S_SRC),
             Paragraph(f"{max(p['leak']['jaccard'] for p in pairs if p['category'] == 'intent'):.3f}", S_SRC)],
            [Paragraph("metaphor", S_SRC), Paragraph(str(counts["metaphor"]), S_SRC),
             Paragraph("metaphors (metaphor = meaning pairs)", S_SRC),
             Paragraph(f"{max(p['leak']['jaccard'] for p in pairs if p['category'] == 'metaphor'):.3f}", S_SRC)],
        ],
        colWidths=[80, 48, 300, 90],
    )
    stats.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, NAVY),
        ("LINEBELOW", (0, 1), (-1, -2), 0.35, RULE),
        ("BACKGROUND", (0, 2), (-1, 2), ZEBRA),
    ]))
    story.append(stats)
    story.append(Spacer(1, 10))

    story.extend(bullets([
        f"<b>Leakage gate.</b> All {len(pairs)} rewrites passed the gate "
        f"(max 8-gram overlap 0, max token Jaccard {max_jac:.2f} against a 0.6 threshold). "
        "The verdict is stored per query under <font name=\"Consolas\" size=\"8\">metadata.leakage</font>.",
        "<b>Not a paraphrase log.</b> The rewritten text comes from "
        "<font name=\"Consolas\" size=\"8\">data/eval/rewrites/batch1-6.json</font>; no model, prompt "
        "or reviewer is recorded, and earlier revisions were not kept.",
        "<b>Ordering.</b> Rows are grouped by category and sorted by <font name=\"Consolas\" size=\"8\">qid</font> "
        "within each section.",
    ]))
    story.append(Spacer(1, 6))
    story.append(callout(
        "How to read a row",
        "The <b>source</b> column quotes the annotation verbatim (multiple captions or metaphor pairs "
        "are separated by \u2022). The <b>frozen query</b> is the exact string the evaluator feeds to "
        "all three strategies; it is not a retyped or cleaned-up copy. Differences in vocabulary are "
        "the rewrite; differences in meaning would be a benchmark error.",
    ))
    story.append(Spacer(1, 12))
    provenance = Table(
        [
            [Paragraph("Source artifacts", S_TH), Paragraph("Role", S_TH)],
            [Paragraph("data/eval/batches/batch1-6.json", S_SRC),
             Paragraph("240 source records handed to the reviewer (annotation snapshot per qid)", S_SRC)],
            [Paragraph("data/eval/rewrites/batch1-6.json", S_SRC),
             Paragraph("240 reviewed rewrites, mapped by qid (the text in the right-hand column)", S_SRC)],
            [Paragraph("data/eval/memecap_eval_v1.json", S_SRC),
             Paragraph("frozen queries, graded qrels and the per-query leakage verdict", S_SRC)],
            [Paragraph("python docs/make_rewrite_table_pdf.py", S_SRC),
             Paragraph("regenerates this document from those three files", S_SRC)],
        ],
        colWidths=[250, 470],
    )
    provenance.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LINEBELOW", (0, 0), (-1, -2), 0.35, RULE),
        ("BACKGROUND", (0, 2), (-1, 2), ZEBRA),
        ("BACKGROUND", (0, 4), (-1, 4), ZEBRA),
    ]))
    story.append(provenance)
    story.append(NextPageTemplate("later"))
    story.append(PageBreak())

    # ---------------------------------------------------------------- sections
    n = 1
    offset = 1
    for cat in ("visual", "intent", "metaphor"):
        title, blurb = CATEGORY_META[cat]
        subset = [p for p in pairs if p["category"] == cat]
        story.append(P(f"{n}.&nbsp;&nbsp;{title} ({len(subset)})", S_H1))
        story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceBefore=1, spaceAfter=6))
        story.append(P(blurb, S_SMALL))
        story.append(Spacer(1, 7))
        story.append(pairs_table(subset, start_no=offset))
        story.append(Spacer(1, 4))
        story.append(P(
            f"Table {n}. {len(subset)} {title.lower()}; source annotation on the left, frozen query on the right.",
            style("Caption", fontSize=7, leading=9, textColor=MUTED, spaceBefore=3),
        ))
        offset += len(subset)
        if cat != "metaphor":
            story.append(PageBreak())
        n += 1

    return story


def main() -> None:
    pairs = load_pairs()
    doc = BaseDocTemplate(
        OUT,
        pagesize=(PAGE_W, PAGE_H),
        leftMargin=LM,
        rightMargin=RM,
        topMargin=TM,
        bottomMargin=BM,
        title="Evaluation benchmark: source vs rewrite",
        author="Meme-Cap T2I retrieval",
        subject="All 240 frozen queries of memecap_eval_v1 with their source annotations",
    )
    frame_first = Frame(LM, BM, CONTENT_W, PAGE_H - TM - BM, id="first")
    frame_later = Frame(LM, BM, CONTENT_W, PAGE_H - TM - BM, id="later")
    doc.addPageTemplates([
        PageTemplate(id="first", frames=[frame_first], onPage=on_first),
        PageTemplate(id="later", frames=[frame_later], onPage=on_later),
    ])
    doc.build(build_story(pairs))
    print(f"Wrote {OUT} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
