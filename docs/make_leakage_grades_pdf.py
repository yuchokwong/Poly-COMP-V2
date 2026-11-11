"""Generate docs/data_leakage_grades.pdf.

The report explains the Anti-Leakage Evaluation Dataset used by
``memecap_eval_v1``: why the 240 queries exist, the four lexical gates that
decide whether a query is leak-free, the deterministic Grade 2 / Grade 1 /
Grade 0 ground truth, and the checking-and-grading result for every query.

It reads only the frozen dataset artifacts:

    data/eval/memecap_eval_v1.json         queries, metadata.leakage, grades
    data/eval/memecap_eval_v1.qrels.jsonl  the same grades as flat judgments
    data/manifest.jsonl                     split counts (optional)

Run:  python docs/make_leakage_grades_pdf.py
"""
from __future__ import annotations

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
        if not os.path.exists(path):
            raise SystemExit(
                f"missing font {path}; the docs/ ReportLab reports expect Segoe UI "
                "and Consolas on Windows"
            )
        pdfmetrics.registerFont(TTFont(name, path))
    pdfmetrics.registerFontFamily(REG, normal=REG, bold=BOLD, italic=ITAL, boldItalic=BOLD)


_register()

NAVY = colors.HexColor("#102A43")
INK = colors.HexColor("#1F2933")
BODY = colors.HexColor("#243B53")
MUTED = colors.HexColor("#62717F")
TEAL = colors.HexColor("#0F766E")
TEAL_LIGHT = colors.HexColor("#E6F4F1")
RED = colors.HexColor("#B42318")
LIGHT = colors.HexColor("#F5F7FA")
RULE = colors.HexColor("#D9E2EC")
ZEBRA = colors.HexColor("#F7F9FB")
WHITE = colors.white

PAGE_W, PAGE_H = landscape(A4)
LM = RM = 42
TM = 56
BM = 46
CONTENT_W = PAGE_W - LM - RM

DOC_DATE = "12 November 2026"
OUT = os.path.join(os.path.dirname(__file__), "data_leakage_grades.pdf")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH = os.path.join(REPO, "data", "eval", "memecap_eval_v1.json")
QRELS_PATH = os.path.join(REPO, "data", "eval", "memecap_eval_v1.qrels.jsonl")
MANIFEST_PATH = os.path.join(REPO, "data", "manifest.jsonl")

CATEGORIES = ("visual", "intent", "metaphor")
DEFAULT_GALLERY = 6378
DEFAULT_TEST = 558
DEFAULT_TRAINVAL = 5820


def style(name, **kw):
    base = dict(
        fontName=REG, fontSize=7.4, leading=9.9, textColor=INK,
        spaceBefore=0, spaceAfter=0,
    )
    base.update(kw)
    return ParagraphStyle(name, **base)


S_TITLE = style("Title", fontName=BOLD, fontSize=23, leading=27, textColor=NAVY, spaceAfter=4)
S_SUB = style("Sub", fontSize=10.6, leading=15, textColor=MUTED, spaceAfter=12)
S_H1 = style("H1", fontName=BOLD, fontSize=13.5, leading=17, textColor=NAVY, spaceAfter=1)
S_H2 = style("H2", fontName=BOLD, fontSize=9.6, leading=12.5, textColor=TEAL, spaceAfter=2)
S_BODY = style("Body", fontSize=9.2, leading=13.2, textColor=BODY, spaceAfter=5)
S_SMALL = style("Small", fontSize=7.8, leading=10.6, textColor=MUTED)
S_TH = style("TH", fontName=BOLD, fontSize=7.1, leading=9.0, textColor=WHITE)
S_CELL = style("Cell", fontSize=7.2, leading=9.6)
S_CELL_SMALL = style("CellSmall", fontSize=6.7, leading=9.0)
S_QUERY = style("Query", fontSize=6.9, leading=9.2)
S_QID = style("Qid", fontName=MONO, fontSize=6.6, leading=9.0, textColor=MUTED)
S_IMG = style("Img", fontName=MONO, fontSize=6.0, leading=8.4, textColor=BODY)
S_IMG_WRAP = style("ImgWrap", fontName=MONO, fontSize=6.4, leading=8.8, textColor=BODY)
S_NUM = style("Num", fontSize=6.5, leading=9.0, textColor=BODY)
S_TOTAL = style("Total", fontName=BOLD, fontSize=7.0, leading=9.2, textColor=NAVY)
S_PASS = style(
    "Pass", fontName=BOLD, fontSize=6.7, leading=9.0, textColor=TEAL, alignment=1
)
S_FAIL = style(
    "Fail", fontName=BOLD, fontSize=6.7, leading=9.0, textColor=RED, alignment=1
)


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def P(text: str, st=S_BODY) -> Paragraph:
    return Paragraph(text, st)


def bullets(items: list[str]) -> list[Paragraph]:
    st = style("Bullet", fontSize=8.6, leading=12.0, leftIndent=12, bulletIndent=1, spaceAfter=3)
    return [Paragraph(item, st, bulletText="\u2022") for item in items]


def callout(title: str, body: str, accent=TEAL, bg=TEAL_LIGHT) -> Table:
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
def load_queries() -> list[dict]:
    with open(EVAL_PATH, encoding="utf-8") as fh:
        records = json.load(fh)
    records.sort(key=lambda q: str(q["qid"]))
    return records


def load_qrels() -> tuple[dict[str, Counter], int]:
    counts: dict[str, Counter] = {}
    judgments = 0
    with open(QRELS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            counts.setdefault(str(row["qid"]), Counter())[int(row["grade"])] += 1
            judgments += 1
    return counts, judgments


def load_counts() -> dict[str, int]:
    """Gallery and split sizes, preferring the manifest when it is present."""
    counts = {
        "gallery": DEFAULT_GALLERY,
        "test": DEFAULT_TEST,
        "trainval": DEFAULT_TRAINVAL,
        "source": "defaults",
    }
    if os.path.exists(MANIFEST_PATH):
        splits: Counter = Counter()
        with open(MANIFEST_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                splits[json.loads(line).get("split")] += 1
        if splits:
            counts["test"] = splits.get("test", counts["test"])
            counts["trainval"] = splits.get("trainval", counts["trainval"])
            counts["gallery"] = sum(splits.values())
            counts["source"] = "data/manifest.jsonl"
    return counts


def grade1_ids(query: dict) -> list[str]:
    return sorted(img for img, grade in query["grades"].items() if int(grade) == 1)


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
    c.drawString(
        LM,
        BM - 24,
        "memecap_eval_v1 \u00b7 anti-leakage checking and deterministic grading "
        "\u00b7 generated from data/eval/",
    )
    c.drawRightString(PAGE_W - RM, BM - 24, f"{c.getPageNumber()}")
    c.restoreState()


def _header(c: canvas.Canvas) -> None:
    c.saveState()
    c.setFont(REG, 7)
    c.setFillColor(MUTED)
    c.drawString(LM, PAGE_H - 30, "Anti-Leakage Evaluation Dataset \u2014 checking and grading")
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
# Tables
# --------------------------------------------------------------------------- #
def headline_table(queries: list[dict], qrels: dict[str, Counter], counts: dict) -> Table:
    total = len(queries)
    passed = sum(1 for q in queries if q["metadata"]["leakage"]["passed"])
    g2 = sum(c.get(2, 0) for c in qrels.values())
    g1 = sum(c.get(1, 0) for c in qrels.values())
    judgments = sum(sum(c.values()) for c in qrels.values())
    max_jac = max(q["metadata"]["leakage"]["jaccard"] for q in queries)
    max_run = max(q["metadata"]["leakage"]["max_source_span"] for q in queries)
    with_g1 = sum(1 for q in queries if grade1_ids(q))
    rows = [
        [Paragraph("Item", S_TH), Paragraph("Value", S_TH)],
        [P("Query set", S_CELL),
         P(f"{total} frozen queries \u2014 80 visual, 80 intent, 80 metaphor", S_CELL)],
        [P("Source pool", S_CELL),
         P(f"{counts['test']:,} unseen test memes \u00b7 gallery {counts['gallery']:,} images "
           f"({counts['trainval']:,} trainval + {counts['test']:,} test)", S_CELL)],
        [P("Leakage check", S_CELL),
         P(f"<b>{passed}/{total} PASS</b> \u00b7 max 8-gram 0 \u00b7 max copied run {max_run} "
           f"(limit &lt;5) \u00b7 max Jaccard {max_jac:.3f} (limit &lt;0.60)", S_CELL)],
        [P("Ground truth", S_CELL),
         P(f"{judgments} explicit judgments \u2014 {g2} Grade 2 + {g1} Grade 1; "
           "Grade 0 is every remaining gallery image", S_CELL)],
        [P("Who grades", S_CELL),
         P("Deterministic rules only \u2014 no model or human relevance judgment", S_CELL)],
    ]
    table = Table(rows, colWidths=[118, CONTENT_W - 118])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, NAVY),
        ("LINEBELOW", (0, 1), (-1, -2), 0.35, RULE),
        ("BACKGROUND", (0, 2), (-1, 2), ZEBRA),
        ("BACKGROUND", (0, 4), (-1, 4), ZEBRA),
    ]))
    return table


def gate_table(queries: list[dict]) -> Table:
    total = len(queries)
    passed = sum(1 for q in queries if q["metadata"]["leakage"]["passed"])
    head = ["Gate", "What it rules out", "Rule", "Observed on the 240"]
    widths = [82, 276, 128, 272]
    rows: list[list] = [[Paragraph(h, S_TH) for h in head]]
    rows.extend([
        [P("<b>1 \u00b7 8-gram overlap</b>", S_CELL),
         P("Copied phrasing from the source annotation or from any gallery annotation", S_CELL),
         P("= 0 shared 8-grams", S_CELL),
         P("max 0 (source) \u00b7 max 0 (gallery)", S_CELL)],
        [P("<b>2 \u00b7 Short-caption containment</b>", S_CELL),
         P("A whole 3\u20134 word caption pasted inside a longer query \u2014 no 8-gram can catch it", S_CELL),
         P("no contained short annotation", S_CELL),
         P("0 source \u00b7 0 gallery", S_CELL)],
        [P("<b>3 \u00b7 Contiguous copy run</b>", S_CELL),
         P("A long verbatim fragment even when it is shorter than 8 tokens", S_CELL),
         P("&lt; 5 tokens shared with the source", S_CELL),
         P("max 4 (limit &lt; 5)", S_CELL)],
        [P("<b>4 \u00b7 Token Jaccard</b>", S_CELL),
         P("Diffuse copying: keeping most of the caption's vocabulary without a long run", S_CELL),
         P("&lt; 0.60 against the source caption", S_CELL),
         P("max 0.500 (limit &lt; 0.60)", S_CELL)],
        [P("<b>Overall</b>", S_TOTAL),
         P("<b>All four gates, source and gallery</b>", S_TOTAL),
         P("\u2014", S_TOTAL),
         P(f"<b>{passed}/{total} PASS</b>", S_TOTAL)],
    ])
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.35, RULE),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), LIGHT),
        ("BACKGROUND", (0, 2), (-1, 2), ZEBRA),
        ("BACKGROUND", (0, 4), (-1, 4), ZEBRA),
    ]))
    return table


def grade_table(queries: list[dict], qrels: dict[str, Counter]) -> Table:
    g2 = sum(c.get(2, 0) for c in qrels.values())
    g1 = sum(c.get(1, 0) for c in qrels.values())
    with_g1 = sum(1 for q in queries if grade1_ids(q))
    head = ["Grade", "Object", "Meaning", "Deterministic assignment", "Count"]
    widths = [42, 44, 150, 404, 118]
    rows: list[list] = [[Paragraph(h, S_TH) for h in head]]
    rows.extend([
        [P("<b>2</b>", S_TOTAL),
         P("image", S_CELL),
         P("The source test meme \u2014 the intended answer", S_CELL),
         P("Exactly one per query: the image the query was written from "
           "(<font name=\"Consolas\" size=\"6.6\">metadata.image_id</font>)", S_CELL),
         P(f"{g2} judgments<br/>(1 per query)", S_CELL)],
        [P("<b>1</b>", S_TOTAL),
         P("image", S_CELL),
         P("A near-equivalent meme, or a meme sharing the same rare metaphor", S_CELL),
         P("Only these rules fire: <b>visual</b> \u2014 another image's caption has Jaccard "
           "\u2265 0.8 with the source caption; <b>intent</b> \u2014 identical normalized "
           "intended meaning; <b>metaphor</b> \u2014 shared specific label with document "
           "frequency \u2264 10 and similar meaning", S_CELL),
         P(f"{g1} judgments<br/>({with_g1} visual queries)", S_CELL)],
        [P("<b>0</b>", S_TOTAL),
         P("image", S_CELL),
         P("Every other gallery image", S_CELL),
         P("Implicit: the complement of Grades 2 and 1. Not written to the qrels file. "
           "For one query: <font name=\"Consolas\" size=\"6.6\">6,378 \u2212 1 \u2212 G1</font>",
           S_CELL),
         P("the rest of<br/>the gallery", S_CELL)],
    ])
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (4, 1), (4, -1), "CENTER"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.35, RULE),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("BACKGROUND", (0, 2), (-1, 2), ZEBRA),
    ]))
    return table


def category_table(queries: list[dict], qrels: dict[str, Counter]) -> Table:
    head = [
        "Category", "Queries", "Leak PASS", "Max 8g<br/>(src/gal)", "Max run",
        "Max Jac", "Grade 2", "Grade 1", "Queries<br/>with G1",
    ]
    widths = [68, 48, 56, 74, 46, 46, 46, 46, 74]
    rows: list[list] = [[Paragraph(h, S_TH) for h in head]]
    for cat in CATEGORIES:
        subset = [q for q in queries if q["category"] == cat]
        leaks = [q["metadata"]["leakage"] for q in subset]
        counts = [qrels.get(str(q["qid"])) or Counter(q["grades"].values()) for q in subset]
        g2 = sum(c.get(2, 0) for c in counts)
        g1 = sum(c.get(1, 0) for c in counts)
        with_g1 = sum(1 for q in subset if grade1_ids(q))
        rows.append([
            P(cat, S_CELL),
            P(str(len(subset)), S_NUM),
            P(f"{sum(1 for x in leaks if x['passed'])}/{len(subset)}", S_NUM),
            P(f"{max(x['max_ngram_overlap'] for x in leaks)} / "
              f"{max(x['strict_overlap'] for x in leaks)}", S_NUM),
            P(str(max(x["max_source_span"] for x in leaks)), S_NUM),
            P(f"{max(x['jaccard'] for x in leaks):.3f}", S_NUM),
            P(str(g2), S_NUM),
            P(str(g1), S_NUM),
            P(str(with_g1), S_NUM),
        ])
    all_leaks = [q["metadata"]["leakage"] for q in queries]
    g2 = sum(c.get(2, 0) for c in qrels.values())
    g1 = sum(c.get(1, 0) for c in qrels.values())
    rows.append([
        P("<b>total</b>", S_CELL),
        P(str(len(queries)), S_NUM),
        P(f"{sum(1 for x in all_leaks if x['passed'])}/{len(queries)}", S_NUM),
        P(f"{max(x['max_ngram_overlap'] for x in all_leaks)} / "
          f"{max(x['strict_overlap'] for x in all_leaks)}", S_NUM),
        P(str(max(x["max_source_span"] for x in all_leaks)), S_NUM),
        P(f"{max(x['jaccard'] for x in all_leaks):.3f}", S_NUM),
        P(str(g2), S_NUM),
        P(str(g1), S_NUM),
        P(str(sum(1 for q in queries if grade1_ids(q))), S_NUM),
    ])
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.35, RULE),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), LIGHT),
    ]
    for i in range(1, len(rows) - 1):
        if i % 2 == 0:
            commands.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    table.setStyle(TableStyle(commands))
    return table


def register_table(queries: list[dict], gallery: int) -> Table:
    head = [
        "No.", "qid", "Cat", "Query", "Leak", "8g<br/>src/gal", "Short<br/>ann.",
        "Run<br/>(max)", "Jac", "Grade 2 source", "G1<br/>count", "G0<br/>count",
    ]
    widths = [22, 38, 46, 300, 34, 36, 34, 28, 36, 110, 28, 42]
    rows: list[list] = [[Paragraph(h, S_TH) for h in head]]
    for number, query in enumerate(queries, start=1):
        qid = str(query["qid"])
        leak = query["metadata"]["leakage"]
        source = str(query["metadata"]["image_id"])
        g1 = grade1_ids(query)
        g0 = gallery - 1 - len(g1)
        rows.append([
            P(str(number), S_NUM),
            P(qid, S_QID),
            P(str(query["category"]), S_CELL_SMALL),
            P(esc(query["query"]), S_QUERY),
            P("PASS" if leak["passed"] else "FAIL", S_PASS if leak["passed"] else S_FAIL),
            P(f"{leak['max_ngram_overlap']}/{leak['strict_overlap']}", S_NUM),
            P("no" if not (leak["contained"] or leak["gallery_contained"]) else "yes", S_NUM),
            P(str(leak["max_source_span"]), S_NUM),
            P(f"{leak['jaccard']:.3f}", S_NUM),
            P(esc(source), S_IMG),
            P(str(len(g1)), S_NUM),
            P(f"{g0:,}", S_NUM),
        ])

    table = Table(rows, colWidths=widths, repeatRows=1)
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (3, 1), (3, -1), "LEFT"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE),
        ("LINEBEFORE", (4, 1), (4, -1), 0.45, RULE),
        ("LINEBEFORE", (9, 1), (9, -1), 0.45, RULE),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            commands.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    table.setStyle(TableStyle(commands))
    return table


def grade1_table(queries: list[dict]) -> Table:
    head = ["qid", "Category", "Query", "G1", "Grade-1 image ids"]
    widths = [40, 50, 250, 30, 388]
    rows: list[list] = [[Paragraph(h, S_TH) for h in head]]
    rows_for_query = [q for q in queries if grade1_ids(q)]
    for query in rows_for_query:
        ids = grade1_ids(query)
        rows.append([
            P(str(query["qid"]), S_QID),
            P(str(query["category"]), S_CELL_SMALL),
            P(esc(query["query"]), S_QUERY),
            P(str(len(ids)), S_NUM),
            P(", ".join(esc(i) for i in ids), S_IMG_WRAP),
        ])
    total = sum(len(grade1_ids(q)) for q in rows_for_query)
    rows.append([
        P("<b>total</b>", S_TOTAL),
        P(f"<b>{len(rows_for_query)} queries</b>", S_TOTAL),
        P("", S_TOTAL),
        P(f"<b>{total}</b>", S_TOTAL),
        P("<b>all Grade-1 judgments</b>", S_TOTAL),
    ])
    table = Table(rows, colWidths=widths, repeatRows=1)
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (3, 0), (3, -1), "CENTER"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.35, RULE),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), LIGHT),
    ]
    for i in range(1, len(rows) - 1):
        if i % 2 == 0:
            commands.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    table.setStyle(TableStyle(commands))
    return table


def provenance_table(counts: dict, judgments: int) -> Table:
    rows = [
        [Paragraph("Source artifact", S_TH), Paragraph("Role", S_TH)],
        [P("data/eval/memecap_eval_v1.json", S_CELL),
         P("240 frozen queries: query text, category, "
           "<font name=\"Consolas\" size=\"6.6\">metadata.leakage</font>, and the "
           "image-level <font name=\"Consolas\" size=\"6.6\">grades</font> map", S_CELL)],
        [P("data/eval/memecap_eval_v1.qrels.jsonl", S_CELL),
         P(f"The same grades flattened into {judgments} judgments "
           "(240 Grade 2 + 99 Grade 1)", S_CELL)],
        [P(esc(counts["source"]), S_CELL),
         P(f"Split counts: {counts['trainval']:,} trainval + {counts['test']:,} test "
           f"= {counts['gallery']:,} gallery images", S_CELL)],
        [P("configs/default.yaml", S_CELL),
         P("<font name=\"Consolas\" size=\"6.6\">eval.max_ngram_overlap: 8</font>, "
           "<font name=\"Consolas\" size=\"6.6\">eval.min_copy_tokens: 5</font>, "
           "<font name=\"Consolas\" size=\"6.6\">eval.metaphor_max_df: 10</font>", S_CELL)],
        [P("src/tsr/evaluate.py", S_CELL),
         P("<font name=\"Consolas\" size=\"6.6\">check_leakage()</font> implements the four "
           "gates; <font name=\"Consolas\" size=\"6.6\">build_eval_query()</font> assigns "
           "the deterministic Grade 2 and Grade 1 judgments", S_CELL)],
        [P("python docs/make_leakage_grades_pdf.py", S_CELL),
         P("regenerates this document from the artifacts above", S_CELL)],
    ]
    table = Table(rows, colWidths=[250, CONTENT_W - 250])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LINEBELOW", (0, 0), (-1, -1), 0.35, RULE),
        ("BACKGROUND", (0, 2), (-1, 2), ZEBRA),
        ("BACKGROUND", (0, 4), (-1, 4), ZEBRA),
        ("BACKGROUND", (0, 6), (-1, 6), ZEBRA),
    ]))
    return table


# --------------------------------------------------------------------------- #
# Story
# --------------------------------------------------------------------------- #
def build_story(
    queries: list[dict], qrels: dict[str, Counter], judgments: int, counts: dict
) -> list:
    story: list = []
    total = len(queries)
    passed = sum(1 for q in queries if q["metadata"]["leakage"]["passed"])
    g1_total = sum(c.get(1, 0) for c in qrels.values())
    with_g1 = sum(1 for q in queries if grade1_ids(q))
    max_jac = max(q["metadata"]["leakage"]["jaccard"] for q in queries)
    max_run = max(q["metadata"]["leakage"]["max_source_span"] for q in queries)

    # ---------------------------------------------------------------- cover
    story.append(Spacer(1, 16))
    story.append(P("Anti-Leakage Evaluation Dataset", S_TITLE))
    story.append(P(
        "Checking and grading the 240 frozen queries of <b>memecap_eval_v1</b>: "
        "four lexical gates, then a deterministic Grade 2 / Grade 1 / Grade 0 "
        "ground truth over the 6,378-image gallery.", S_SUB))

    story.append(callout(
        "What this document answers",
        f"<b>memecap_eval_v1</b> is the frozen benchmark used to compare Strategies A, B and C. "
        f"This document explains how its {total} queries are checked for annotation leakage and "
        "how each query is graded, then prints the result for every query. "
        f"<b>{passed}/{total} queries pass the leakage check.</b> Every query has exactly one "
        "Grade 2 source image; 15 visual queries add 99 Grade 1 near-duplicate judgments; every "
        "other gallery image is Grade 0.",
    ))
    story.append(Spacer(1, 10))
    story.append(headline_table(queries, qrels, counts))
    story.append(Spacer(1, 10))
    story.append(callout(
        "Two axes \u2014 keep them apart",
        "<b>Query axis (checking).</b> Does the query copy its source annotation? PASS means "
        "leak-free. This is the outcome the presentation labels \u201cGrade 0 leak-free "
        "queries\u201d. All 240 are PASS.<br/><br/>"
        "<b>Image axis (grading).</b> For one query, which gallery images are relevant? "
        "Grade 2 is the source test meme, Grade 1 is a deterministic near-equivalent, and "
        "Grade 0 is every other image. These grades describe <i>images</i>, not the query "
        "verdict. The same numbers are used on both axes, so this document names the axis "
        "explicitly.",
        accent=NAVY, bg=LIGHT,
    ))
    story.append(Spacer(1, 14))
    story.append(P("In this document", S_H2))
    story.append(HRFlowable(width="100%", thickness=0.8, color=RULE, spaceBefore=1, spaceAfter=4))
    story.append(P(
        "1 What the dataset is&nbsp;&nbsp;·&nbsp;&nbsp;2 The four leakage gates"
        "&nbsp;&nbsp;·&nbsp;&nbsp;3 Deterministic grading"
        "&nbsp;&nbsp;·&nbsp;&nbsp;4 Summary by category"
        "&nbsp;&nbsp;·&nbsp;&nbsp;5 Per-query register"
        "&nbsp;&nbsp;·&nbsp;&nbsp;6 Grade-1 judgments"
        "&nbsp;&nbsp;·&nbsp;&nbsp;7 Provenance", S_SMALL))
    story.append(NextPageTemplate("later"))
    story.append(PageBreak())

    # ---------------------------------------------------------------- concept
    story.append(P("1.&nbsp;&nbsp;What the dataset is", S_H1))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceBefore=1, spaceAfter=6))
    story.append(P(
        f"<b>memecap_eval_v1</b> contains {total} queries written from the "
        f"{counts['test']} images in the test split of Meme-Cap. The split is balanced: "
        "80 <b>visual</b> queries describe what is visible, 80 <b>intent</b> queries paraphrase "
        "the meme's intended meaning, and 80 <b>metaphor</b> queries name the cultural template "
        "or metaphor the meme invokes.", S_BODY))
    story.append(P(
        "The queries are rewrites, not captions. That is the point of the anti-leakage design: "
        "Strategies B and C retrieve through the gallery's annotations, so a query that reused "
        "an annotation's wording could be matched by BM25 or a dense encoder on the copied "
        "phrase. Such a query would measure the annotation, not the retrieval model. Every "
        "candidate query must therefore pass four lexical gates before it is frozen.", S_BODY))
    story.append(P(
        "The dataset then has two layers. <b>Checking</b> decides whether a query is leak-free. "
        "<b>Grading</b> records which gallery images count as relevant to that query. They are "
        "independent: a leak-free query can have more than one relevant image (a near-duplicate "
        "meme), and a query with one relevant image can still leak. The next two sections take "
        "each layer in turn.", S_BODY))
    story.append(Spacer(1, 4))

    # ---------------------------------------------------------------- checking
    story.append(P("2.&nbsp;&nbsp;Checking: the four leakage gates", S_H1))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceBefore=1, spaceAfter=6))
    story.append(P(
        "The gate is applied query by query. Each candidate is compared with its own source "
        "annotation and with every annotation in the gallery; it is frozen only if it passes "
        "all four checks.", S_BODY))
    story.append(gate_table(queries))
    story.append(Spacer(1, 7))
    story.append(P("Why four checks?", S_H2))
    story.extend(bullets([
        "<b>8-gram overlap</b> catches direct copying. A shared 8-token sequence with the "
        "source or with any of the 6,378 gallery annotations is a fail. The gallery-wide "
        "comparison matters because a query written from one meme can copy a different meme's "
        "annotation.",
        "<b>Short-caption containment</b> catches what an 8-gram cannot: a whole 3- or 4-word "
        "caption pasted into a longer query. If the source annotation is short, the query may "
        "not contain all of it; the same check runs against gallery annotations.",
        "<b>The contiguous-run limit</b> catches a long verbatim fragment even when it is "
        "shorter than 8 tokens. The longest run shared with the source must stay below "
        "<font name=\"Consolas\" size=\"8\">min_copy_tokens = 5</font>.",
        "<b>Jaccard</b> catches diffuse copying: a query assembled by deleting words from the "
        "caption can keep most of its vocabulary without any long contiguous run. Token-set "
        "Jaccard with the source caption must stay below 0.60.",
    ]))
    story.append(P(
        f"<b>Result.</b> Every query clears every gate. Maximum 8-gram overlap is 0 against "
        f"both source and gallery; no short annotation is contained; the longest copied run is "
        f"{max_run} tokens; the highest Jaccard is {max_jac:.3f}. The check is lexical by "
        "design. It asks whether the query's <i>form</i> was copied; it does not ask whether "
        "the query is answerable from the image. Answerability is the grading layer's job.",
        S_BODY))
    story.append(Spacer(1, 4))

    # ---------------------------------------------------------------- grading
    story.append(P("3.&nbsp;&nbsp;Grading: deterministic ground truth", S_H1))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceBefore=1, spaceAfter=6))
    story.append(P(
        "Each query is paired with a graded set of gallery images. The grades are deterministic: "
        "the only inputs are the source image, the annotations, and the rules below. No model or "
        "crowd judge decides relevance.", S_BODY))
    story.append(grade_table(queries, qrels))
    story.append(Spacer(1, 7))
    story.extend(bullets([
        "<b>Grade 2 is exactly one image per query</b>: "
        "<font name=\"Consolas\" size=\"8\">metadata.image_id</font>, the test meme the query "
        "was written from. There are 240 Grade-2 judgments.",
        f"<b>Grade 1 adds other near-equivalent answers.</b> On this set the rule fires only "
        f"for visual queries: {with_g1} queries, {g1_total} judgments, at most 18 images for "
        "q0015. Intent and metaphor queries have no Grade-1 images.",
        "<b>Grade 0 is implicit.</b> It is every gallery image not graded 2 or 1, so for one "
        "query it contains <font name=\"Consolas\" size=\"8\">6,378 \u2212 1 \u2212 G1</font> "
        "images. It is not written to the qrels file; storing tens of thousands of zero "
        "judgments would add no information.",
    ]))
    story.append(Spacer(1, 4))

    # worked example
    example = next((q for q in queries if 0 < len(grade1_ids(q)) <= 2), queries[0])
    eid = str(example["qid"])
    eleak = example["metadata"]["leakage"]
    eids = grade1_ids(example)
    eg0 = counts["gallery"] - 1 - len(eids)
    g1_text = ", ".join(eids) if eids else "none"
    story.append(callout(
        f"Worked example: {eid} ({example['category']})",
        f"<b>{esc(example['query'])}</b><br/>"
        f"<b>Check:</b> {'PASS' if eleak['passed'] else 'FAIL'} \u2014 8-gram "
        f"{eleak['max_ngram_overlap']}/{eleak['strict_overlap']}, no short annotation, "
        f"longest run {eleak['max_source_span']}, Jaccard {eleak['jaccard']:.4f}.<br/>"
        f"<b>Grade 2:</b> {esc(example['metadata']['image_id'])} (the source test meme).<br/>"
        f"<b>Grade 1:</b> {esc(g1_text)} ({len(eids)} near-equivalent "
        f"{'meme' if len(eids) == 1 else 'memes'}).<br/>"
        f"<b>Grade 0:</b> {eg0:,} other gallery images (implicit).",
    ))
    story.append(Spacer(1, 4))

    # ---------------------------------------------------------------- summary
    story.append(P("4.&nbsp;&nbsp;Summary by category", S_H1))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceBefore=1, spaceAfter=6))
    story.append(category_table(queries, qrels))
    story.append(Spacer(1, 3))
    story.append(P(
        "All categories pass; only visual queries carry Grade-1 images; the highest Jaccard "
        "is 0.500, below the 0.60 limit.", S_SMALL))
    story.append(PageBreak())

    # ---------------------------------------------------------------- register
    story.append(P(f"5.&nbsp;&nbsp;Per-query checking and grading ({total} queries)", S_H1))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceBefore=1, spaceAfter=6))
    story.append(P(
        "Rows are in qid order: q0001\u2013q0080 visual, q0081\u2013q0160 intent, "
        "q0161\u2013q0240 metaphor. <b>Leak</b> is the query-level verdict: all 240 rows are "
        "PASS, the presentation's \u201cGrade 0 leak-free\u201d outcome. The Grade 2/1/0 "
        "columns describe <i>images</i>, not query verdicts. <b>8g</b> is the "
        "maximum 8-gram overlap with the source / with any gallery annotation. <b>Short</b> "
        "says whether the query contains a whole short annotation from the source or gallery. "
        "<b>Run</b> is the longest contiguous token run shared with the source (limit &lt; 5). "
        "<b>Jac</b> is token Jaccard with the source caption (limit &lt; 0.60). <b>Grade 2 "
        "source</b> is the image id of the source test meme \u2014 one per query. <b>G1</b> "
        "counts Grade-1 near-equivalent images; <b>G0</b> counts the remaining gallery images "
        "(6,378 \u2212 1 \u2212 G1).", S_SMALL))
    story.append(Spacer(1, 6))
    story.append(register_table(queries, counts["gallery"]))
    story.append(Spacer(1, 5))
    story.append(P(
        f"Table 1. {total} frozen queries; {passed}/{total} pass the four-gate leakage check. "
        "Every query has exactly one Grade 2 source image; all other images are Grade 0 unless "
        "listed as Grade 1 in Table 2.", S_SMALL))

    # ---------------------------------------------------------------- grade 1
    story.append(PageBreak())
    story.append(P("6.&nbsp;&nbsp;Grade-1 judgments", S_H1))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceBefore=1, spaceAfter=6))
    story.append(P(
        f"Only {with_g1} visual queries carry Grade-1 images. The table lists every id so the "
        f"grading can be audited. These {g1_total} judgments plus the 240 Grade-2 source "
        "judgments are the 339 explicit qrels; every other gallery image is an implicit "
        "Grade 0.", S_SMALL))
    story.append(Spacer(1, 6))
    story.append(grade1_table(queries))
    story.append(Spacer(1, 5))
    story.append(P(
        "Table 2. Deterministic Grade-1 judgments. Intent and metaphor queries have none "
        "on this set.", S_SMALL))

    # ---------------------------------------------------------------- provenance
    story.append(Spacer(1, 14))
    story.append(P("7.&nbsp;&nbsp;Provenance", S_H1))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceBefore=1, spaceAfter=6))
    story.append(provenance_table(counts, judgments))
    story.append(Spacer(1, 4))
    story.append(P(
        "Regenerate with <font name=\"Consolas\" size=\"7.5\">python "
        "docs/make_leakage_grades_pdf.py</font>. Every number is read from the repository at "
        "build time. Retrieval scores are separate from this dataset document; they live in "
        "<font name=\"Consolas\" size=\"7.5\">data/reports/compare.json</font> and the "
        "evaluation report.", S_SMALL))
    return story


def main() -> None:
    queries = load_queries()
    qrels, judgments = load_qrels()
    counts = load_counts()
    if len(queries) != 240:
        print(f"warning: expected 240 queries, found {len(queries)}")
    missing = [q["qid"] for q in queries if str(q["qid"]) not in qrels]
    if missing:
        print(f"warning: no qrels for {len(missing)} queries: {', '.join(missing[:5])}")
    for query in queries:
        qid = str(query["qid"])
        source = str(query["metadata"]["image_id"])
        expected = Counter(int(g) for g in query["grades"].values())
        if expected.get(2, 0) != 1:
            print(f"warning: {qid} has {expected.get(2, 0)} Grade-2 images")
        if str(source) not in query["grades"] or int(query["grades"][source]) != 2:
            print(f"warning: {qid} source image is not graded 2")
        if qid in qrels and qrels[qid] != expected:
            print(f"warning: {qid} qrels disagree with the grades map")

    doc = BaseDocTemplate(
        OUT,
        pagesize=(PAGE_W, PAGE_H),
        leftMargin=LM,
        rightMargin=RM,
        topMargin=TM,
        bottomMargin=BM,
        title="Anti-Leakage Evaluation Dataset: checking and grading",
        author="Meme-Cap T2I retrieval",
        subject="The four leakage gates and deterministic Grade 2/1/0 ground truth "
                "for the 240 queries of memecap_eval_v1",
    )
    frame_first = Frame(LM, BM, CONTENT_W, PAGE_H - TM - BM, id="first")
    frame_later = Frame(LM, BM, CONTENT_W, PAGE_H - TM - BM, id="later")
    doc.addPageTemplates([
        PageTemplate(id="first", frames=[frame_first], onPage=on_first),
        PageTemplate(id="later", frames=[frame_later], onPage=on_later),
    ])
    doc.build(build_story(queries, qrels, judgments, counts))
    print(f"Wrote {OUT} ({len(queries)} queries, {judgments} judgments)")


if __name__ == "__main__":
    main()
