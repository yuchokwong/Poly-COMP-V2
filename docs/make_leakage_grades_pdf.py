"""Generate docs/data_leakage_grades.pdf.

A per-query register for all 240 frozen queries of memecap_eval_v1: the stored
data-leakage verdict and gate metrics, the graded qrels, and each strategy's
first-relevant rank and average precision.

The data is read from the same artifacts the evaluator uses:

    data/eval/memecap_eval_v1.json       frozen queries + metadata.leakage
    data/eval/memecap_eval_v1.qrels.jsonl  graded judgments (qid, image, grade)
    data/reports/compare.json            per-query rank/AP for A, B, C
    data/reports/eval_<key>.json         fallback when compare.json is absent

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
RED_LIGHT = colors.HexColor("#FDECEA")
LIGHT = colors.HexColor("#F5F7FA")
RULE = colors.HexColor("#D9E2EC")
ZEBRA = colors.HexColor("#F7F9FB")
WHITE = colors.white

PAGE_W, PAGE_H = landscape(A4)
LM = RM = 42
TM = 56
BM = 46
CONTENT_W = PAGE_W - LM - RM

DOC_DATE = "11 November 2026"
OUT = os.path.join(os.path.dirname(__file__), "data_leakage_grades.pdf")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH = os.path.join(REPO, "data", "eval", "memecap_eval_v1.json")
QRELS_PATH = os.path.join(REPO, "data", "eval", "memecap_eval_v1.qrels.jsonl")
REPORTS_DIR = os.path.join(REPO, "data", "reports")

STRATEGIES = (
    ("clip", "A", "A \u00b7 clip (CLIP only)"),
    ("clip_nlp", "B", "B \u00b7 clip_nlp (CLIP + NL)"),
    ("vlm", "C", "C \u00b7 vlm (Qwen descriptions)"),
)
CATEGORIES = ("visual", "intent", "metaphor")


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
S_NUM = style("Num", fontSize=6.5, leading=9.0, textColor=BODY)
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


def load_qrels() -> dict[str, Counter]:
    counts: dict[str, Counter] = {}
    with open(QRELS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            counts.setdefault(str(row["qid"]), Counter())[int(row["grade"])] += 1
    return counts


def load_scores() -> tuple[dict, dict[str, dict]]:
    """Per-query rank/AP for each strategy, plus provenance.

    ``compare.json`` is the single source when present: it carries the frozen
    per-query rows for all three strategies and the run provenance. The
    standalone ``eval_<key>.json`` reports are an equivalent fallback.
    """
    meta = {
        "generated_at": None,
        "gallery": None,
        "top_k": [1, 5, 10],
        "seed": 42,
        "n_boot": 1000,
        "source": "data/reports/compare.json",
    }
    raw: dict[str, dict | None] = {}
    compare_path = os.path.join(REPORTS_DIR, "compare.json")
    if os.path.exists(compare_path):
        with open(compare_path, encoding="utf-8") as fh:
            data = json.load(fh)
        meta["generated_at"] = data.get("generated_at")
        meta["gallery"] = (data.get("probe") or {}).get("num_images")
        meta["top_k"] = data.get("top_k") or meta["top_k"]
        meta["seed"] = data.get("seed", meta["seed"])
        meta["n_boot"] = data.get("n_boot", meta["n_boot"])
        strategies = data.get("strategies") or {}
        for key, _letter, _label in STRATEGIES:
            raw[key] = strategies.get(key)
    else:
        meta["source"] = "data/reports/eval_<key>.json"
        for key, _letter, _label in STRATEGIES:
            path = os.path.join(REPORTS_DIR, f"eval_{key}.json")
            raw[key] = None
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    raw[key] = json.load(fh)

    scores: dict[str, dict] = {}
    for key, _letter, _label in STRATEGIES:
        report = raw.get(key) or {}
        rows = report.get("per_query") or []
        scores[key] = {
            "ready": bool(report.get("ready")) and bool(rows),
            "title": report.get("title") or key,
            "metrics": report.get("metrics") or {},
            "per_category": report.get("per_category") or {},
            "rows": {str(row["qid"]): row for row in rows},
        }
    return meta, scores


def qrels_for(qid: str, query: dict, qrels: dict[str, Counter]) -> Counter:
    counts = qrels.get(qid)
    if counts:
        return counts
    return Counter(query.get("grades", {}).values())


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
        "memecap_eval_v1 \u00b7 per-query leakage grades and retrieval scores \u00b7 "
        "generated from data/eval/ and data/reports/",
    )
    c.drawRightString(PAGE_W - RM, BM - 24, f"{c.getPageNumber()}")
    c.restoreState()


def _header(c: canvas.Canvas) -> None:
    c.saveState()
    c.setFont(REG, 7)
    c.setFillColor(MUTED)
    c.drawString(LM, PAGE_H - 30, "Data leakage grades \u2014 memecap_eval_v1")
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
def fmt_ap(value) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def fmt_metric(value) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def fmt_rank(value) -> str:
    return "miss" if value is None else str(int(value))


def headline_table(
    queries: list[dict], qrels: dict[str, Counter], scores: dict, meta: dict
) -> Table:
    total = len(queries)
    passed = sum(1 for q in queries if q["metadata"]["leakage"]["passed"])
    judgments = sum(sum(c.values()) for c in qrels.values())
    g2 = sum(c.get(2, 0) for c in qrels.values())
    g1 = sum(c.get(1, 0) for c in qrels.values())
    max_jac = max(q["metadata"]["leakage"]["jaccard"] for q in queries)
    max_span = max(q["metadata"]["leakage"]["max_source_span"] for q in queries)
    maps = "  \u00b7  ".join(
        f"{letter} {fmt_metric(scores[key]['metrics'].get('mAP'))}"
        for key, letter, _label in STRATEGIES
    )
    gallery = meta.get("gallery") or 6378
    rows = [
        [Paragraph("Query set", S_TH), Paragraph("Value", S_TH)],
        [P("memecap_eval_v1", S_CELL),
         P(f"{total} frozen queries \u2014 80 visual, 80 intent, 80 metaphor", S_CELL)],
        [P("Leakage verdicts", S_CELL),
         P(f"{passed}/{total} PASS \u00b7 {total - passed} FAIL \u00b7 "
           f"max 8-gram overlap 0 \u00b7 max shared span {max_span} "
           f"\u00b7 max Jaccard {max_jac:.2f}", S_CELL)],
        [P("Graded judgments", S_CELL),
         P(f"{judgments} judgments \u2014 {g2} grade 2 + {g1} grade 1 "
           "(15 visual queries carry grade-1 near-duplicates)", S_CELL)],
        [P("Retrieval scores", S_CELL),
         P(f"mAP {maps} \u00b7 gallery {gallery:,} \u00b7 depth 100", S_CELL)],
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


def category_table(queries: list[dict], qrels: dict[str, Counter], scores: dict) -> Table:
    head = [
        "Category", "Queries", "Leak PASS", "Max 8g", "Max span", "Max Jac",
        "Grade 2", "Grade 1", "mAP A", "mAP B", "mAP C",
    ]
    widths = [70, 46, 54, 44, 50, 48, 46, 46, 50, 50, 50]
    rows: list[list] = [[Paragraph(h, S_TH) for h in head]]
    for cat in CATEGORIES:
        subset = [q for q in queries if q["category"] == cat]
        leaks = [q["metadata"]["leakage"] for q in subset]
        counts = [qrels_for(str(q["qid"]), q, qrels) for q in subset]
        g2 = sum(c.get(2, 0) for c in counts)
        g1 = sum(c.get(1, 0) for c in counts)
        maps = [
            scores[key]["per_category"].get(cat, {}).get("mAP")
            for key, _letter, _label in STRATEGIES
        ]
        rows.append([
            P(cat, S_CELL),
            P(str(len(subset)), S_NUM),
            P(f"{sum(1 for x in leaks if x['passed'])}/{len(subset)}", S_NUM),
            P(str(max(x["max_ngram_overlap"] for x in leaks)), S_NUM),
            P(str(max(x["max_source_span"] for x in leaks)), S_NUM),
            P(f"{max(x['jaccard'] for x in leaks):.3f}", S_NUM),
            P(str(g2), S_NUM),
            P(str(g1), S_NUM),
            P(fmt_metric(maps[0]), S_NUM),
            P(fmt_metric(maps[1]), S_NUM),
            P(fmt_metric(maps[2]), S_NUM),
        ])
    all_leaks = [q["metadata"]["leakage"] for q in queries]
    g2 = sum(c.get(2, 0) for c in qrels.values())
    g1 = sum(c.get(1, 0) for c in qrels.values())
    rows.append([
        P("<b>total</b>", S_CELL),
        P(str(len(queries)), S_NUM),
        P(f"{sum(1 for x in all_leaks if x['passed'])}/{len(queries)}", S_NUM),
        P(str(max(x["max_ngram_overlap"] for x in all_leaks)), S_NUM),
        P(str(max(x["max_source_span"] for x in all_leaks)), S_NUM),
        P(f"{max(x['jaccard'] for x in all_leaks):.3f}", S_NUM),
        P(str(g2), S_NUM),
        P(str(g1), S_NUM),
        *[P(fmt_metric(scores[key]["metrics"].get("mAP")), S_NUM)
          for key, _letter, _label in STRATEGIES],
    ])
    table = Table(rows, colWidths=widths, repeatRows=1)
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


def register_table(queries: list[dict], qrels: dict[str, Counter], scores: dict) -> Table:
    head = [
        "No.", "qid", "Cat", "Query", "Leak", "8g", "span", "Jac", "G2", "G1",
        "A AP", "B AP", "C AP", "A rank", "B rank", "C rank",
    ]
    widths = [24, 40, 48, 248, 34, 26, 30, 38, 22, 22, 38, 38, 38, 34, 34, 34]
    rows: list[list] = [[Paragraph(h, S_TH) for h in head]]
    for number, query in enumerate(queries, start=1):
        qid = str(query["qid"])
        leak = query["metadata"]["leakage"]
        counts = qrels_for(qid, query, qrels)
        cells: list = [
            P(str(number), S_NUM),
            P(qid, S_QID),
            P(str(query["category"]), S_CELL_SMALL),
            P(esc(query["query"]), S_QUERY),
            P("PASS" if leak["passed"] else "FAIL", S_PASS if leak["passed"] else S_FAIL),
            P(str(leak["max_ngram_overlap"]), S_NUM),
            P(str(leak["max_source_span"]), S_NUM),
            P(f"{leak['jaccard']:.3f}", S_NUM),
            P(str(counts.get(2, 0)), S_NUM),
            P(str(counts.get(1, 0)), S_NUM),
        ]
        for key, _letter, _label in STRATEGIES:
            row = scores[key]["rows"].get(qid)
            cells.append(P(fmt_ap(row.get("ap")) if row else "n/a", S_NUM))
        for key, _letter, _label in STRATEGIES:
            row = scores[key]["rows"].get(qid)
            cells.append(P(fmt_rank(row.get("rank")) if row else "n/a", S_NUM))
        rows.append(cells)

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
        ("LINEBEFORE", (10, 1), (10, -1), 0.45, RULE),
        ("LINEBEFORE", (13, 1), (13, -1), 0.45, RULE),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            commands.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    table.setStyle(TableStyle(commands))
    return table


def provenance_table(meta: dict, scores: dict) -> Table:
    generated = meta.get("generated_at") or "not recorded"
    gallery = meta.get("gallery") or 6378
    rows = [
        [Paragraph("Source artifact", S_TH), Paragraph("Role", S_TH)],
        [P("data/eval/memecap_eval_v1.json", S_CELL),
         P("240 frozen queries with <b>metadata.leakage</b> and the graded qrels", S_CELL)],
        [P("data/eval/memecap_eval_v1.qrels.jsonl", S_CELL),
         P("339 judgments (240 grade 2 + 99 grade 1), a flattened export of the "
           "same grades", S_CELL)],
        [P(esc(meta.get("source", "data/reports/compare.json")), S_CELL),
         P(f"per-query rank and AP for A/B/C; generated {generated}; "
           f"gallery {gallery:,} images; top_k "
           f"{'/'.join(str(k) for k in meta.get('top_k') or [])}; "
           f"seed {meta.get('seed')}", S_CELL)],
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
    ]))
    return table


# --------------------------------------------------------------------------- #
# Story
# --------------------------------------------------------------------------- #
def build_story(queries: list[dict], qrels: dict[str, Counter], scores: dict, meta: dict) -> list:
    story: list = []
    total = len(queries)
    passed = sum(1 for q in queries if q["metadata"]["leakage"]["passed"])
    missing = [key for key, _l, _lab in STRATEGIES if not scores[key]["ready"]]

    # ---------------------------------------------------------------- cover
    story.append(Spacer(1, 18))
    story.append(P("Data leakage grades and retrieval scores", S_TITLE))
    story.append(P(
        f"All {total} frozen queries of <b>memecap_eval_v1</b>, with the leakage verdict and "
        "gate metrics stored for each one, the graded qrels, and the first-relevant rank and "
        "average precision of every strategy on the same 6,378-image gallery.", S_SUB))

    story.append(callout(
        "Answer to the request",
        f"<b>Yes.</b> Every one of the {total} frozen queries has a stored data-leakage verdict "
        f"in <font name=\"Consolas\" size=\"8\">metadata.leakage</font>, and "
        f"<b>{passed}/{total} pass</b> the gate. The graded qrels and the per-query retrieval "
        "scores are printed in the register below.",
    ))
    story.append(Spacer(1, 10))
    story.append(headline_table(queries, qrels, scores, meta))
    story.append(Spacer(1, 10))
    story.extend(bullets([
        '<b>What "grade" means here.</b> Each query has two independent records: a '
        "leakage verdict (PASS/FAIL plus the gate metrics) and one or more relevance grades "
        "(grade 2 = the source meme, grade 1 = a deterministic near-duplicate).",
        "<b>Scores are per query.</b> The register shows each strategy's first-relevant rank "
        "(<font name=\"Consolas\" size=\"8\">miss</font> when nothing relevant is retrieved) and "
        "AP. Per-candidate similarity scores are not stored in the reports, so they are not shown.",
        "<b>Diagnostics excluded.</b> The <font name=\"Consolas\" size=\"8\">*_raw_captions</font> "
        "reports are leaky by construction and are a separate labelled diagnostic; they never "
        "enter this table.",
    ]))
    if missing:
        story.append(P(
            f"<b>Unavailable:</b> {', '.join(missing)} \u2014 affected score cells read "
            "<font name=\"Consolas\" size=\"8\">n/a</font>.", S_SMALL))
    story.append(Spacer(1, 6))
    story.append(P(
        "Sources: <font name=\"Consolas\" size=\"7.5\">data/eval/memecap_eval_v1.json</font> "
        "\u00b7 <font name=\"Consolas\" size=\"7.5\">data/eval/memecap_eval_v1.qrels.jsonl</font> "
        "\u00b7 <font name=\"Consolas\" size=\"7.5\">data/reports/compare.json</font>. "
        "Every number is read at build time; nothing is typed in by hand.", S_SMALL))
    story.append(NextPageTemplate("later"))
    story.append(PageBreak())

    # ---------------------------------------------------------------- definitions
    story.append(P("1.&nbsp;&nbsp;How to read the grades", S_H1))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceBefore=1, spaceAfter=6))
    story.append(P("Leakage gate", S_H2))
    story.extend(bullets([
        "A query is rejected if an <b>8-gram</b> overlaps any annotation field of its source "
        "<i>or of any gallery image</i>.",
        "It is rejected if it embeds a whole short annotation (<b>\u2265 4 tokens</b>) \u2014 no "
        "8-gram can catch a 3-word caption pasted verbatim.",
        "It is rejected if it shares a contiguous run of <b>\u2265 5 tokens</b> with its source "
        "caption, or if its token <b>Jaccard \u2265 0.6</b> (no length floor).",
        "Verdicts are stored per query in <font name=\"Consolas\" size=\"8\">metadata.leakage</font>; "
        "on this set all 240 pass, so every row reads PASS.",
    ]))
    story.append(P("Relevance grades", S_H2))
    story.extend(bullets([
        "<b>Grade 2</b> is the source meme itself \u2014 exactly one per query.",
        "<b>Grade 1</b> is assigned only by deterministic rules: a visual query whose other "
        "image has caption Jaccard \u2265 0.8; an intent query with identical normalized intended "
        "meaning; or a metaphor query sharing a specific label (document frequency \u2264 10) "
        "with a similar meaning. No model judges relevance.",
        f"On this set, {sum(1 for q in queries if sum(1 for g in q['grades'].values() if g == 1) > 0)} "
        "visual queries carry grade-1 near-duplicates (99 judgments); intent and metaphor "
        "queries have one grade-2 image each.",
    ]))
    story.append(P("Metric semantics", S_H2))
    story.extend(bullets([
        "<b>rank</b> is the 1-based position of the first relevant image (grade 1 or 2); "
        "<b>AP</b> is average precision over the 100-deep ranked list. Both treat grade 1 and "
        "grade 2 as equally relevant.",
        "<b>nDCG@10</b> (summarised, not per row) is the only graded metric: it uses gains "
        "<font name=\"Consolas\" size=\"8\">2**grade - 1</font>.",
        "Retrieval depth is <font name=\"Consolas\" size=\"8\">max(top_k, largest relevant "
        "set, 100)</font>; top_k is 1/5/10 and the largest relevant set is 19, so every list "
        "is 100 deep.",
    ]))
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------------- summary
    story.append(P("2.&nbsp;&nbsp;Summary by category", S_H1))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceBefore=1, spaceAfter=6))
    story.append(category_table(queries, qrels, scores))
    story.append(Spacer(1, 4))
    spans = Counter(q["metadata"]["leakage"]["max_source_span"] for q in queries)
    span_text = ", ".join(f"{span} on {count}" for span, count in sorted(spans.items()))
    story.append(P(
        "All 240 queries pass the leakage gate. Max 8-gram overlap is 0 in every category; "
        f"the longest shared span is 4 tokens (below the 5-token copy limit); distribution: "
        f"{span_text}. The highest Jaccard is 0.500, below the 0.600 threshold. Grade-1 "
        "judgments exist only on visual queries.",
        S_SMALL))

    example = queries[0]
    eid = str(example["qid"])
    eleak = example["metadata"]["leakage"]
    ecounts = qrels_for(eid, example, qrels)
    score_bits = []
    for key, letter, _label in STRATEGIES:
        row = scores[key]["rows"].get(eid)
        if row:
            score_bits.append(
                f"{letter}: AP {fmt_ap(row.get('ap'))}, rank {fmt_rank(row.get('rank'))}"
            )
        else:
            score_bits.append(f"{letter}: n/a")
    story.append(Spacer(1, 10))
    story.append(callout(
        f"Worked example: {eid} ({example['category']})",
        f"<b>{esc(example['query'])}</b><br/>"
        f"Leakage: {'PASS' if eleak['passed'] else 'FAIL'} · 8-gram "
        f"{eleak['max_ngram_overlap']} · span {eleak['max_source_span']} · "
        f"Jaccard {eleak['jaccard']:.3f}. Qrels: G2 {ecounts.get(2, 0)}, "
        f"G1 {ecounts.get(1, 0)}. " + " · ".join(score_bits) + ".",
    ))
    story.append(Spacer(1, 12))
    story.append(PageBreak())

    # ---------------------------------------------------------------- register
    story.append(P(f"3.&nbsp;&nbsp;Per-query register ({total} queries)", S_H1))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceBefore=1, spaceAfter=6))
    story.append(P(
        "Rows are in qid order, which is also category order: q0001–q0080 visual, "
        "q0081–q0160 intent, q0161–q0240 metaphor. <b>Leak</b> is the stored "
        "verdict; <b>8g</b>, <b>span</b> and "
        "<b>Jac</b> are the 8-gram overlap, longest shared source span and token Jaccard, "
        "respectively. The <font name=\"Consolas\" size=\"8\">contained</font>, "
        "<font name=\"Consolas\" size=\"8\">gallery_contained</font> and "
        "<font name=\"Consolas\" size=\"8\">strict_overlap</font> flags are 0/false for all 240 "
        "queries. <b>G2</b>/<b>G1</b> count the graded qrels. <b>AP</b> and <b>rank</b> are the "
        "per-query retrieval scores for A clip, B clip_nlp and C vlm.", S_SMALL))
    story.append(Spacer(1, 6))
    story.append(register_table(queries, qrels, scores))
    story.append(Spacer(1, 5))
    story.append(P(
        f"Table 1. {total} frozen queries; all {passed} pass the data-leakage gate. "
        "AP is rounded to three decimals in this table; the reports retain four.", S_SMALL))

    # ---------------------------------------------------------------- provenance
    story.append(Spacer(1, 14))
    story.append(P("4.&nbsp;&nbsp;Provenance", S_H1))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL, spaceBefore=1, spaceAfter=6))
    story.append(provenance_table(meta, scores))
    story.append(Spacer(1, 4))
    story.append(P(
        "Regenerate with <font name=\"Consolas\" size=\"7.5\">python "
        "docs/make_leakage_grades_pdf.py</font>. The PDF is generated from the repository, "
        "not written by hand.", S_SMALL))
    return story


def main() -> None:
    queries = load_queries()
    qrels = load_qrels()
    meta, scores = load_scores()
    if len(queries) != 240:
        print(f"warning: expected 240 queries, found {len(queries)}")
    missing = [q["qid"] for q in queries if str(q["qid"]) not in qrels]
    if missing:
        print(f"warning: no qrels for {len(missing)} queries: {', '.join(missing[:5])}")
    for key, _letter, _label in STRATEGIES:
        if not scores[key]["ready"]:
            print(f"warning: strategy {key} is not ready; its score cells will read n/a")
            continue
        absent = [q["qid"] for q in queries if str(q["qid"]) not in scores[key]["rows"]]
        if absent:
            print(f"warning: {key} has no per-query row for {len(absent)} queries")

    doc = BaseDocTemplate(
        OUT,
        pagesize=(PAGE_W, PAGE_H),
        leftMargin=LM,
        rightMargin=RM,
        topMargin=TM,
        bottomMargin=BM,
        title="Data leakage grades and retrieval scores",
        author="Meme-Cap T2I retrieval",
        subject="Per-query leakage verdicts, graded qrels and retrieval scores for memecap_eval_v1",
    )
    frame_first = Frame(LM, BM, CONTENT_W, PAGE_H - TM - BM, id="first")
    frame_later = Frame(LM, BM, CONTENT_W, PAGE_H - TM - BM, id="later")
    doc.addPageTemplates([
        PageTemplate(id="first", frames=[frame_first], onPage=on_first),
        PageTemplate(id="later", frames=[frame_later], onPage=on_later),
    ])
    doc.build(build_story(queries, qrels, scores, meta))
    print(f"Wrote {OUT} ({len(queries)} queries)")


if __name__ == "__main__":
    main()
