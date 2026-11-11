"""Generate docs/evaluation_pipeline.pdf.

A self-contained technical walkthrough of the evaluation architecture: how the
240-query benchmark is drafted, rewritten, gated and frozen, how strategies are
run and scored, how significance is computed, and how the report reaches the
/evaluation page.

Run:  python docs/make_evaluation_pdf.py
"""
from __future__ import annotations

import os
from datetime import date

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    HRFlowable,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

# --------------------------------------------------------------------------- #
# Fonts and palette
# --------------------------------------------------------------------------- #
FONT_DIR = r"C:\Windows\Fonts"
REG = "SegoeUI"
BOLD = "SegoeUIBold"
ITAL = "SegoeUIItalic"
MONO = "Consolas"
MONO_B = "ConsolasBold"


def _register() -> None:
    faces = [
        (REG, "segoeui.ttf"),
        (BOLD, "segoeuib.ttf"),
        (ITAL, "segoeuii.ttf"),
        (MONO, "consola.ttf"),
        (MONO_B, "consolab.ttf"),
    ]
    for name, filename in faces:
        path = os.path.join(FONT_DIR, filename)
        if os.path.exists(path):
            pdfmetrics.registerFont(TTFont(name, path))
    pdfmetrics.registerFontFamily(REG, normal=REG, bold=BOLD, italic=ITAL, boldItalic=BOLD)
    pdfmetrics.registerFontFamily(MONO, normal=MONO, bold=MONO_B, italic=MONO, boldItalic=MONO_B)


_register()

NAVY = colors.HexColor("#102A43")
INK = colors.HexColor("#1F2933")
BODY = colors.HexColor("#243B53")
MUTED = colors.HexColor("#62717F")
TEAL = colors.HexColor("#0F766E")
TEAL_LIGHT = colors.HexColor("#E6F4F1")
BLUE = colors.HexColor("#1D4ED8")
BLUE_LIGHT = colors.HexColor("#EEF3FF")
AMBER = colors.HexColor("#B45309")
AMBER_LIGHT = colors.HexColor("#FDF6EC")
RED = colors.HexColor("#B42318")
RED_LIGHT = colors.HexColor("#FEF3F2")
GREEN = colors.HexColor("#047857")
LIGHT = colors.HexColor("#F5F7FA")
LIGHTER = colors.HexColor("#FAFBFC")
RULE = colors.HexColor("#D9E2EC")
RULE_DARK = colors.HexColor("#BCCCDC")
ZEBRA = colors.HexColor("#F7F9FB")

PAGE_W, PAGE_H = A4
LM = RM = 54
TM = 62
BM = 54
CONTENT_W = PAGE_W - LM - RM

COMMIT = "d852b7c"
DOC_DATE = "4 November 2026"
ARTIFACT_DATE = "1 November 2026"
PDF_PATH = os.path.join(os.path.dirname(__file__), "evaluation_pipeline.pdf")


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
def style(name, **kw):
    base = dict(
        fontName=REG,
        fontSize=9.4,
        leading=13.4,
        textColor=BODY,
        spaceBefore=0,
        spaceAfter=5,
    )
    base.update(kw)
    return ParagraphStyle(name, **base)


S_BODY = style("Body")
S_SMALL = style("Small", fontSize=8.2, leading=11.4, textColor=MUTED)
S_H1 = style(
    "H1", fontName=BOLD, fontSize=16, leading=20, textColor=NAVY,
    spaceBefore=6, spaceAfter=2,
)
S_H2 = style(
    "H2", fontName=BOLD, fontSize=11.2, leading=14.5, textColor=TEAL,
    spaceBefore=11, spaceAfter=2,
)
S_H3 = style(
    "H3", fontName=BOLD, fontSize=9.6, leading=12.8, textColor=INK,
    spaceBefore=7, spaceAfter=1,
)
S_BULLET = style("Bullet", leftIndent=13, bulletIndent=2, spaceAfter=3.5)
S_CODE = style(
    "Code", fontName=MONO, fontSize=7.5, leading=10.2, textColor=INK,
    spaceAfter=0, spaceBefore=0,
)
S_CAPTION = style("Caption", fontSize=7.7, leading=10.4, textColor=MUTED, spaceBefore=3, spaceAfter=9)
S_TH = style("TH", fontName=BOLD, fontSize=7.7, leading=10, textColor=colors.white, spaceAfter=0)
S_TD = style("TD", fontSize=7.8, leading=10.4, textColor=INK, spaceAfter=0)
S_TD_SMALL = style("TDSmall", fontSize=7.2, leading=9.5, textColor=INK, spaceAfter=0)
S_TOC_H = style("TOCH", fontName=BOLD, fontSize=16, leading=20, textColor=NAVY, spaceAfter=12)
S_QUOTE = style("Quote", fontName=ITAL, fontSize=8.6, leading=12.4, textColor=MUTED, leftIndent=10)


def esc(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def mono(text: str, size: float = 7.9) -> str:
    return f'<font name="{MONO}" size="{size}" color="#334E68">{esc(text)}</font>'


def P(text: str, st=S_BODY):
    return Paragraph(text, st)


def H1(number: str, title: str):
    return [
        CondPageBreak(90),
        Paragraph(f"{number}&nbsp;&nbsp;{esc(title)}", S_H1),
        HRFlowable(width="100%", thickness=1.1, color=TEAL, spaceBefore=1, spaceAfter=8),
    ]


def H2(number: str, title: str):
    return [CondPageBreak(170), Paragraph(f"{number}&nbsp;&nbsp;{esc(title)}", S_H2)]


def H3(title: str):
    return [Paragraph(esc(title), S_H3)]


def bullets(items: list[str], st=S_BULLET):
    return [Paragraph(item, st, bulletText="•") for item in items]


def callout(title: str, body: str, kind: str = "teal"):
    accents = {
        "teal": (TEAL, TEAL_LIGHT),
        "amber": (AMBER, AMBER_LIGHT),
        "red": (RED, RED_LIGHT),
        "blue": (BLUE, BLUE_LIGHT),
        "gray": (MUTED, LIGHT),
    }
    accent, bg = accents[kind]
    inner = [
        Paragraph(
            f'<font color="{accent.hexval()}">{esc(title)}</font>',
            style("CT", fontName=BOLD, fontSize=8.5, leading=11, spaceAfter=2.5),
        ),
        Paragraph(body, style("CB", fontSize=8.2, leading=11.8, textColor=BODY, spaceAfter=0)),
    ]
    table = Table([["", inner]], colWidths=[4, CONTENT_W - 4])
    table.setStyle(
        TableStyle(
            [
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
            ]
        )
    )
    return KeepTogether(table)


def code_block(text: str, size: float = 7.4):
    text = text.strip("\n")
    flow = Preformatted(text, S_CODE if size == 7.5 else style(f"Code{size}", fontName=MONO, fontSize=size, leading=size * 1.35))
    table = Table([[flow]], colWidths=[CONTENT_W])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def caption(text: str):
    return Paragraph(text, S_CAPTION)


def make_table(
    header: list[str],
    rows: list[list[str]],
    widths: list[float],
    *,
    align: list[str] | None = None,
    font: ParagraphStyle = S_TD,
    header_font: ParagraphStyle = S_TH,
    repeat: int = 1,
    pad: float = 4.2,
):
    data = [[Paragraph(h, header_font) for h in header]]
    for row in rows:
        data.append([cell if isinstance(cell, Flowable) else Paragraph(str(cell), font) for cell in row])
    table = Table(data, colWidths=widths, repeatRows=repeat, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), pad),
        ("RIGHTPADDING", (0, 0), (-1, -1), pad),
        ("TOPPADDING", (0, 0), (-1, -1), 4.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.6),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            commands.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    if align:
        for j, kind in enumerate(align):
            if kind == "r":
                commands.append(("ALIGN", (j, 1), (j, -1), "RIGHT"))
            elif kind == "c":
                commands.append(("ALIGN", (j, 1), (j, -1), "CENTER"))
    table.setStyle(TableStyle(commands))
    if len(data) <= 9:
        return KeepTogether(table)
    return table


# --------------------------------------------------------------------------- #
# Diagrams
# --------------------------------------------------------------------------- #
def _wrap_lines(text: str, font: str, size: float, width: float) -> list[str]:
    return simpleSplit(text, font, size, width)


def _box(c, x, y, w, h, title, lines, *, fill=colors.white, stroke=RULE_DARK,
         title_color=NAVY, size=6.6, title_size=7.4, radius=4):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(0.6)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1)
    inner_w = w - 8
    if not title:
        wrapped: list[str] = []
        for line in lines:
            wrapped.extend(_wrap_lines(line, REG, size, inner_w))
        total_h = len(wrapped) * size + max(0, len(wrapped) - 1) * 1.8
        ty = y + (h + total_h) / 2 - size
        c.setFillColor(BODY)
        c.setFont(REG, size)
        for line in wrapped:
            c.drawString(x + 4, ty, line)
            ty -= size + 1.8
        return
    c.setFillColor(title_color)
    c.setFont(BOLD, title_size)
    ty = y + h - title_size - 5
    for line in _wrap_lines(title, BOLD, title_size, inner_w):
        c.drawString(x + 4, ty, line)
        ty -= title_size + 1.5
    c.setFont(REG, size)
    c.setFillColor(BODY)
    ty -= 1.5
    for line in lines:
        for part in _wrap_lines(line, REG, size, inner_w):
            if ty < y + 3:
                break
            c.drawString(x + 4, ty, part)
            ty -= size + 1.8


def _arrow_h(c, x1, y, x2, color=RULE_DARK):
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(0.9)
    c.line(x1, y, x2 - 3, y)
    c.line(x2 - 3, y, x2 - 7, y + 2.6)
    c.line(x2 - 3, y, x2 - 7, y - 2.6)


def _arrow_v(c, x, y1, y2, color=RULE_DARK):
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(0.9)
    c.line(x, y1, x, y2 + 3)
    c.line(x, y2 + 3, x - 2.6, y2 + 7)
    c.line(x, y2 + 3, x + 2.6, y2 + 7)


class ArchitectureDiagram(Flowable):
    def __init__(self, width=CONTENT_W, height=286):
        super().__init__()
        self.width = width
        self.height = height

    def wrap(self, aw, ah):
        return (self.width, self.height)

    def draw(self):
        c = self.canv
        w = self.width
        labels = [
            ("01", "SOURCE", "Meme-Cap annotations and images; SHA-1 dedupe"),
            ("02", "STORE", "Manifest, SQLite and image files; shared image ids"),
            ("03", "INDEXES", "One CLIP index, two text indexes, one VLM index"),
            ("04", "BENCHMARK", "Draft, rewrite, leakage gate, frozen qrels"),
            ("05", "EVAL & SERVE", "Same queries through A/B/C; static reports"),
        ]
        pills = [
            ["Meme-Cap 6,382 entries", "prepare: PNG → JPEG", "6,378 unique images"],
            ["manifest.jsonl", "SQLite images", "strategy_docs", "data/images/"],
            ["A  CLIP → FAISS", "B  BGE + BM25", "C  VLM → BGE"],
            ["draft_sources.json", "rewrites/", "leakage gate", "frozen set + qrels"],
            ["tsr eval / compare", "reports/*.json", "compare.html", "/evaluation"],
        ]
        band_h = 44
        gap = 7
        top = self.height - 4
        spine_x = 21
        label_w = 92

        c.setStrokeColor(RULE)
        c.setLineWidth(1)
        c.line(spine_x, top - len(labels) * (band_h + gap) + gap + 6, spine_x, top - 4)

        for i, ((num, name, desc), row) in enumerate(zip(labels, pills)):
            y = top - (i + 1) * band_h - i * gap
            c.setFillColor(TEAL if i in (3, 4) else NAVY)
            c.circle(spine_x, y + band_h / 2, 8.5, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.setFont(BOLD, 6.6)
            c.drawCentredString(spine_x, y + band_h / 2 - 2.2, num)

            c.setFillColor(NAVY)
            c.setFont(BOLD, 7.8)
            c.drawString(label_w - 52, y + band_h - 13, name)
            c.setFillColor(MUTED)
            c.setFont(REG, 5.9)
            ty = y + band_h - 23
            for line in _wrap_lines(desc, REG, 5.9, 60):
                c.drawString(label_w - 52, ty, line)
                ty -= 7.2

            x = label_w + 44
            available = w - x - 4
            n = len(row)
            gap_pill = 6
            pill_w = (available - gap_pill * (n - 1)) / n
            for text in row:
                _box(c, x, y + 5, pill_w, band_h - 10, "", [text], size=6.3, fill=LIGHTER,
                     stroke=RULE_DARK, radius=3)
                x += pill_w + gap_pill

            if i < len(labels) - 1:
                _arrow_v(c, spine_x, y - 1, y - gap + 1, color=RULE_DARK)


class PipelineDiagram(Flowable):
    def __init__(self, width=CONTENT_W, height=128):
        super().__init__()
        self.width = width
        self.height = height

    def wrap(self, aw, ah):
        return (self.width, self.height)

    def draw(self):
        c = self.canv
        stages = [
            ("1  DRAFT", ["seed 42", "per-category pool", "template + snapshot"]),
            ("2  REWRITE", ["one text per qid", "same intent", "new wording"]),
            ("3  GATE", ["5 lexical checks", "source + gallery", "all must pass"]),
            ("4  FREEZE", ["rule-based qrels", "test split only", "no partial write"]),
            ("5  RUN", ["A / B / C", "cached retrieval", "per-query metrics"]),
            ("6  COMPARE", ["bootstrap CIs", "paired tests", "Holm, one family"]),
        ]
        n = len(stages)
        gap = 10
        bw = (self.width - gap * (n - 1)) / n
        y = 34
        h = 78
        for i, (title, lines) in enumerate(stages):
            x = i * (bw + gap)
            fill = TEAL_LIGHT if i in (2, 3) else colors.white
            stroke = TEAL if i in (2, 3) else RULE_DARK
            _box(c, x, y, bw, h, title, lines, fill=fill, stroke=stroke, size=6.0, title_size=6.8)
            if i < n - 1:
                _arrow_h(c, x + bw + 1, y + h / 2, x + bw + gap - 1)
        c.setFillColor(MUTED)
        c.setFont(REG, 6.4)
        c.drawString(0, 14, "Any rejected rewrite aborts the freeze: the set is never silently shrunk.")
        c.setFillColor(RED)
        c.drawString(0, 6, "The gate is lexical, not semantic: a paraphrase with no shared n-gram passes by design.")


class StatsDiagram(Flowable):
    def __init__(self, width=CONTENT_W, height=102):
        super().__init__()
        self.width = width
        self.height = height

    def wrap(self, aw, ah):
        return (self.width, self.height)

    def draw(self):
        c = self.canv
        boxes = [
            ("SAME QUERIES", ["frozen q0001–q0240", "same gallery", "one strategy at a time"]),
            ("PER-QUERY VALUES", ["AP, RR, recall@R", "P/R/nDCG at 1,5,10", "latency per search"]),
            ("RESAMPLING", ["1,000 bootstrap draws", "percentile 95% CI", "paired B−A differences"]),
            ("ONE REPORT", ["Holm across 15 tests", "win / tie / loss", "JSON + HTML"]),
        ]
        n = len(boxes)
        gap = 16
        bw = (self.width - gap * (n - 1)) / n
        y = 8
        h = 84
        for i, (title, lines) in enumerate(boxes):
            x = i * (bw + gap)
            _box(c, x, y, bw, h, title, lines, size=6.1, title_size=6.9,
                 fill=BLUE_LIGHT if i in (1, 2) else colors.white,
                 stroke=BLUE if i in (1, 2) else RULE_DARK)
            if i < n - 1:
                _arrow_h(c, x + bw + 1, y + h / 2, x + bw + gap - 1)


# --------------------------------------------------------------------------- #
# Cover and document template
# --------------------------------------------------------------------------- #
def draw_cover(c: canvas.Canvas, doc):
    w, h = PAGE_W, PAGE_H
    band_top = h * 0.40

    c.setFillColor(NAVY)
    c.rect(0, band_top, w, h - band_top, fill=1, stroke=0)
    c.setFillColor(TEAL)
    c.rect(0, band_top - 4, w, 4, fill=1, stroke=0)

    x = 54
    y = h - 78
    c.setFillColor(colors.HexColor("#9FB3C8"))
    c.setFont(BOLD, 8.2)
    c.drawString(x, y, "SYSTEM DOCUMENTATION  ·  EVALUATION")

    c.setFillColor(colors.white)
    c.setFont(BOLD, 29)
    c.drawString(x, y - 46, "Evaluation Architecture")
    c.drawString(x, y - 80, "and Benchmark Pipeline")

    c.setFillColor(colors.HexColor("#D9E2EC"))
    c.setFont(REG, 11.2)
    subtitle = (
        "How the 240-query Meme-Cap benchmark is drafted, rewritten, leakage-gated, "
        "frozen, executed and reported for three retrieval strategies."
    )
    ty = y - 112
    for line in _wrap_lines(subtitle, REG, 11.2, w - 160):
        c.drawString(x, ty, line)
        ty -= 16

    # strategy chips
    chip_y = band_top + 26
    labels = ["A  clip", "B  clip_nlp", "C  vlm"]
    cx = x
    for label in labels:
        cw = 84
        c.setFillColor(colors.HexColor("#1B3A57"))
        c.setStrokeColor(colors.HexColor("#3E5C76"))
        c.roundRect(cx, chip_y, cw, 24, 4, fill=1, stroke=1)
        c.setFillColor(colors.HexColor("#D9E2EC"))
        c.setFont(BOLD, 7.8)
        c.drawCentredString(cx + cw / 2, chip_y + 8, label)
        cx += cw + 8

    # metadata below the band
    meta_y = band_top - 46
    c.setFillColor(NAVY)
    c.setFont(BOLD, 8.2)
    c.drawString(x, meta_y, "PROJECT")
    c.setFont(REG, 9.2)
    c.setFillColor(INK)
    c.drawString(x + 66, meta_y, "Meme-Cap text-to-image retrieval  ·  COMP5523  ·  repository V2")

    c.setFillColor(NAVY)
    c.setFont(BOLD, 8.2)
    c.drawString(x, meta_y - 20, "SOURCE")
    c.setFont(REG, 9.2)
    c.setFillColor(INK)
    c.drawString(x + 66, meta_y - 20, f"Code at commit {COMMIT}  ·  local artifacts generated {ARTIFACT_DATE}")

    c.setFillColor(NAVY)
    c.setFont(BOLD, 8.2)
    c.drawString(x, meta_y - 40, "DOCUMENT")
    c.setFont(REG, 9.2)
    c.setFillColor(INK)
    c.drawString(x + 66, meta_y - 40, f"Version 1.0  ·  {DOC_DATE}")

    # pipeline strip
    strip_y = 76
    c.setStrokeColor(RULE_DARK)
    c.setLineWidth(0.8)
    c.line(x, strip_y + 12, w - x, strip_y + 12)
    stages = ["Draft", "Review", "Gate", "Freeze", "Evaluate", "Compare", "Serve"]
    step = (w - 2 * x) / (len(stages) - 1)
    for i, label in enumerate(stages):
        sx = x + i * step
        c.setFillColor(TEAL if i in (2, 3) else NAVY)
        c.circle(sx, strip_y + 12, 2.6, fill=1, stroke=0)
        c.setFillColor(MUTED)
        c.setFont(BOLD, 6.6)
        c.drawCentredString(sx, strip_y - 2, label.upper())

    c.setFillColor(MUTED)
    c.setFont(REG, 7)
    c.drawString(x, 34, "Every statement in this document traces to the code paths and artifacts named in its text.")
    c.drawRightString(w - x, 34, "Contents on the following page")


class Doc(BaseDocTemplate):
    def __init__(self, filename, **kw):
        super().__init__(filename, pagesize=A4, **kw)
        self._section = ""

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            name = flowable.style.name
            if name in ("H1", "H2"):
                text = flowable.getPlainText().replace("\u00a0", " ")
                level = 0 if name == "H1" else 1
                self.notify("TOCEntry", (level, text, self.page))
                if level == 0:
                    self._section = text


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            if self._pageNumber > 1:
                self._running(total)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def _running(self, total):
        w, h = PAGE_W, PAGE_H
        y = h - 32
        self.setStrokeColor(RULE)
        self.setLineWidth(0.5)
        self.line(LM, y - 10, w - RM, y - 10)
        self.setFillColor(MUTED)
        self.setFont(REG, 7)
        self.drawString(LM, y, "Evaluation Architecture and Benchmark Pipeline")
        self.drawRightString(w - RM, y, f"Meme-Cap T2I Retrieval  ·  COMP5523")

        self.setStrokeColor(RULE)
        self.line(LM, 36, w - RM, 36)
        self.setFillColor(MUTED)
        self.setFont(REG, 6.8)
        self.drawString(LM, 24, f"Source commit {COMMIT}  ·  artifacts {ARTIFACT_DATE}  ·  document {DOC_DATE}")
        self.drawRightString(w - RM, 24, f"{self._pageNumber} / {total}")


# --------------------------------------------------------------------------- #
# Story
# --------------------------------------------------------------------------- #
def build_story() -> list:
    story: list = []
    add = story.append
    extend = story.extend

    # ---------------- Cover ----------------
    add(NextPageTemplate("Body"))
    add(PageBreak())

    # ---------------- Contents ----------------
    add(Paragraph("Contents", S_TOC_H))
    toc = TableOfContents()
    toc.levelStyles = [
        style("TOC0", fontName=BOLD, fontSize=9.2, leading=12.4, textColor=NAVY, leftIndent=0,
              firstLineIndent=0, spaceBefore=4, spaceAfter=0),
        style("TOC1", fontSize=8.2, leading=10.8, textColor=BODY, leftIndent=16,
              firstLineIndent=0, spaceBefore=0, spaceAfter=0),
    ]
    toc.tableStyle = TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ])
    toc.dotsMinLevel = 0
    add(toc)
    add(Spacer(1, 18))
    add(callout(
        "Scope of this document",
        "The reader is assumed to know information retrieval and basic statistics, but not this "
        "codebase. Sections 1–3 give the map; Sections 4–8 follow one query from raw annotation to "
        "a table cell on the /evaluation page; Sections 9–10 cover reproducibility and the points an "
        "auditor should check. Appendix A is the file/function index, Appendix B a worked example, "
        "Appendix C the current local snapshot, and Appendix D the command reference.",
        "blue",
    ))
    add(PageBreak())

    # ======================================================================
    # 1 Executive summary
    # ======================================================================
    extend(H1("1", "Executive summary"))

    add(P(
        "The system retrieves memes from one gallery of 6,378 images using three strategies that differ "
        "only in which text they match against the image: <b>A</b> <font name=\"Consolas\" size=\"8\">clip</font> uses "
        "the OpenCLIP image embedding alone; <b>B</b> <font name=\"Consolas\" size=\"8\">clip_nlp</font> fuses CLIP with a dense "
        "index over human annotations and BM25, with an optional cross-encoder rerank; <b>C</b> "
        "<font name=\"Consolas\" size=\"8\">vlm</font> uses offline Qwen2.5-VL descriptions so that no human annotation is needed "
        "at serving time. Evaluation exists to answer two separate questions, and the code keeps them "
        "separate: <i>is the benchmark trustworthy?</i> and <i>is the difference between strategies real?</i>"
    ))
    add(P(
        "The first question dominates the design. Strategy B indexes the same human annotations that a "
        "query could accidentally copy, so a benchmark built naively from captions would reward string "
        "overlap rather than retrieval. The pipeline therefore drafts queries, rewrites them away from "
        "the annotation wording, runs a lexical leakage gate over the source image and the entire gallery, "
        "and freezes the survivors with rule-derived graded relevance. The second question is handled with "
        "paired bootstrap tests over per-query metric values and a single Holm–Bonferroni correction across "
        "every pairwise comparison."
    ))

    extend(H2("1.1", "At a glance"))
    add(make_table(
        ["Item", "Value", "Where it is defined or stored"],
        [
            ["Query set", "240 queries: 80 visual, 80 intent, 80 metaphor", "configs/default.yaml · eval.per_category"],
            ["Source split", "558 test-split images (out of 6,378 gallery images)", "SQLite images.split · local store"],
            ["Distinct query sources", "240 (one per query)", "data/eval/memecap_eval_v1.json"],
            ["Graded judgments", "339: 240 grade 2 + 99 grade 1", "data/eval/memecap_eval_v1.qrels.jsonl"],
            ["Frozen set", "memecap_eval_v1 · never edited after freeze", "data/eval/memecap_eval_v1.json"],
            ["Metrics", "P@1/5/10, R@1/5/10, mAP, MRR, nDCG@10, recall@R, median rank, misses", "evaluate.py · evaluate_queries"],
            ["Significance", "1,000-sample paired bootstrap, seed 42; Holm across 15 tests", "configs/default.yaml · eval.bootstrap/seed"],
            ["Served report", compare_placeholder, "data/reports/compare.json"],
        ],
        [92, 232, 163],
        font=S_TD_SMALL,
    ))
    add(caption(
        "Values measured from the local store and the frozen artifacts on " + ARTIFACT_DATE + ". "
        "“Where it is defined” names the authoritative source, not a copy."
    ))

    extend(H2("1.2", "The five properties that make the comparison defensible"))
    extend(bullets([
        "<b>Test-only sources.</b> Every query is written for an image in the 558-image test split, and the "
        "freeze verifies that the source is still in that split. The gallery itself remains the full 6,378 images; "
        "the split constrains where queries come from, not where retrieval searches.",
        "<b>A frozen, auditable set.</b> Queries and qrels are written once. Each record carries the leakage verdict "
        "that admitted it, so the set can be audited without rerunning the gate. A rejected rewrite aborts the "
        "build instead of shrinking the set silently.",
        "<b>Deterministic relevance.</b> The source image is grade 2. Grade 1 exists only where a fixed rule "
        "fires: a near-identical scene caption, an identical intent sentence, or a shared non-literal metaphor "
        "with the same meaning and a document frequency below 10. No model judges relevance.",
        "<b>A fair, paired comparison.</b> All strategies see the same frozen queries against the same gallery. "
        "The same cached retrieval run supplies both the headline metrics and the per-query vectors used for "
        "paired tests, so a strategy cannot benefit from a lucky second run.",
        "<b>Static, traceable reporting.</b> <font name=\"Consolas\" size=\"8\">tsr compare</font> writes JSON and HTML once; the "
        "web page reads those files. The two header values “Evaluation queries 240” and “Eval set "
        "memecap_eval_v1” are fields of <font name=\"Consolas\" size=\"8\">compare.json</font>, not a live query.",
    ]))

    add(callout(
        "Where the numbers on the /evaluation page come from",
        "The page calls <b>GET /api/eval/summary</b>, which returns <font name=\"Consolas\" size=\"7.8\">data/reports/compare.json</font> "
        "through the <font name=\"Consolas\" size=\"7.8\">EvalSummaryResponse</font> schema. The header row is rendered by "
        "<font name=\"Consolas\" size=\"7.8\">web/eval.js</font>: <font name=\"Consolas\" size=\"7.8\">report.num_queries</font> → "
        "“Evaluation queries”, and <font name=\"Consolas\" size=\"7.8\">report.eval_set</font> → “Eval set”. "
        "<font name=\"Consolas\" size=\"7.8\">num_queries</font> is <font name=\"Consolas\" size=\"7.8\">len(queries)</font> at compare "
        "time; <font name=\"Consolas\" size=\"7.8\">eval_set</font> is the query file stem "
        "(<font name=\"Consolas\" size=\"7.8\">memecap_eval_v1</font>), defaulting to "
        "<font name=\"Consolas\" size=\"7.8\">eval.set_name</font>. If the page and the artifacts disagree, the page is stale: "
        "rerun <font name=\"Consolas\" size=\"7.8\">tsr compare</font>.",
        "teal",
    ))

    # ======================================================================
    # 2 Architecture
    # ======================================================================
    extend(H1("2", "System architecture and boundaries"))

    add(P(
        "Evaluation is a layer on top of the same gallery and the same strategy objects that serve the API. "
        "Nothing in the evaluation path re-implements retrieval: a strategy is asked for a ranked list, and "
        "the evaluator compares that list to qrels. This keeps the measured system and the served system "
        "identical, which is the point of evaluating in the first place."
    ))
    add(ArchitectureDiagram())
    add(caption(
        "Figure 1. End-to-end architecture. The benchmark layer (04) sits beside the index layer (03): it "
        "consumes the same gallery records to build queries and qrels, but never changes an index. The "
        "evaluation layer (05) only reads frozen queries and strategy output."
    ))

    extend(H2("2.1", "Three evaluation modules, three responsibilities"))
    add(make_table(
        ["Module", "Owns", "Does not own"],
        [
            [mono("evaluate.py"), "Query/qrels data model, leakage checks, drafting and freezing, metric primitives, bootstrap and Holm helpers.",
             "Strategy loading, report layout, HTTP."],
            [mono("compare.py"), "Running strategies over the frozen set, caching retrieval, per-query values, pairwise tests, ablations, report writing.",
             "Building the query set; serving the report."],
            [mono("cli.py"), "The <font name=\"Consolas\" size=\"7.4\">make-eval</font>, <font name=\"Consolas\" size=\"7.4\">eval</font> and <font name=\"Consolas\" size=\"7.4\">compare</font> entry points, label rules for diagnostic sets.",
             "Metric definitions or statistical tests."],
        ],
        [78, 255, 154],
        font=S_TD_SMALL,
    ))

    extend(H2("2.2", "The identity contract between retrieval and evaluation"))
    add(P(
        "Qrels are keyed by <font name=\"Consolas\" size=\"8\">image_id</font>, the stable string from the manifest. "
        "Strategy hits carry the SQLite integer primary key. The evaluation runner bridges the two through "
        "<font name=\"Consolas\" size=\"8\">store.get(pk).image_id</font> and caches the mapping, so a qrel can never be "
        "matched against the wrong row even if SQLite row order changes. The same <font name=\"Consolas\" size=\"8\">image_id</font> "
        "is the shared key across SQLite, the FAISS id map and <font name=\"Consolas\" size=\"8\">image_ids.npy</font>; strategy "
        "builds may rewrite their own document rows but must never renumber <font name=\"Consolas\" size=\"8\">images</font>."
    ))

    extend(H2("2.3", "Readiness is reported, not assumed"))
    add(P(
        "A strategy that is known but unbuilt returns 503 from the API and is reported as "
        "<font name=\"Consolas\" size=\"8\">ready: false</font> in the evaluation report rather than crashing the run. "
        "The evaluator records the load error and continues with the strategies that are available. Strategy C "
        "adds a stricter rule: if its description index covers less than 95% of the gallery, it is reported "
        "unavailable, so an interrupted captioning run cannot masquerade as a built strategy. This is why a "
        "report can legitimately contain two strategies instead of three; the pairwise family is then 1 pair × "
        "5 metrics, not 3 pairs × 5."
    ))

    add(callout(
        "The gallery is not the test split",
        "Retrieval always searches all 6,378 images. The test split (558 images) is the pool from which query "
        "sources are drawn, not a filtered retrieval corpus. A common misreading is that evaluation searches only "
        "the test split; it does not, and the report's <font name=\"Consolas\" size=\"7.8\">probe.num_images</font> confirms "
        "the full gallery size.",
        "amber",
    ))

    # ======================================================================
    # 3 Corpus and splits
    # ======================================================================
    extend(H1("3", "Corpus, splits and category eligibility"))

    add(P(
        "Images and annotations come from Meme-Cap (arXiv:2305.13703). The dataset provides 6,382 annotated "
        "entries. <font name=\"Consolas\" size=\"8\">tsr prepare</font> downloads and normalises them, converts PNG to JPEG, "
        "and drops duplicate images by SHA-1. The local manifest records 6,378 records, 6,382 annotations and "
        "4 duplicates, so the gallery is 6,378 unique images. The store partitions those images into "
        "<font name=\"Consolas\" size=\"8\">test</font> (558) and <font name=\"Consolas\" size=\"8\">trainval</font> (5,820)."
    ))
    add(callout(
        "Two documentation numbers are slightly off",
        "README.md says “6,382 images” and a “559-image test split”; the generated walkthrough says 558. The "
        "local manifest and SQLite store agree on 6,378 unique images and 558 test images. The 6,382 figure is "
        "the annotation count before deduplication, and 559 appears to be an off-by-one transcription. Cite "
        "6,378 and 558 when describing the actual gallery, or fix the prose before publication.",
        "amber",
    ))

    extend(H2("3.1", "Category eligibility"))
    add(P(
        "Each category draws from a different annotation field, so a query is only drafted where the source "
        "image can support it:"
    ))
    add(make_table(
        ["Category", "Eligible when the image has", "Draft template", "What the query tests"],
        [
            ["visual", "at least one <font name=\"Consolas\" size=\"7.4\">img_captions</font> entry",
             "“i am looking for an image that depicts …”", "Literal scene description, no intended meaning."],
            ["intent", "at least one <font name=\"Consolas\" size=\"7.4\">meme_captions</font> entry",
             "“find a meme whose message is that …”", "The joke or argument the poster conveys."],
            ["metaphor", "a non-literal metaphor row (label + meaning)",
             "“a meme using … as a symbol for …”", "Symbol → intended meaning, not the literal scene."],
        ],
        [58, 150, 152, 127],
        font=S_TD_SMALL,
    ))
    add(P(
        "A metaphor row is non-literal when both the label and the meaning are non-empty and the meaning is "
        "not the dataset's literal marker <font name=\"Consolas\" size=\"8\">“not related to the meme context”</font>, nor a "
        "bare description of the annotation such as “Meme poster”. Those rows are still indexed by Strategy B, "
        "so they participate in the leakage gate, but they never seed a metaphor query or earn grade-1 "
        "relevance. The exclusion is deliberate: “a man” is a literal depiction, not shared symbolism."
    ))

    extend(H2("3.2", "Why the source split matters"))
    add(P(
        "The benchmark is documented as test-only, and the freeze rejects any source whose live split is not "
        "<font name=\"Consolas\" size=\"8\">test</font>. This makes the query sample a fixed, documented subset of the corpus "
        "rather than an arbitrary draw from images that also appear in training material. It does <i>not</i> "
        "partition retrieval: the gallery still contains trainval and test images, and Strategy B indexes "
        "annotations for both. Leakage control therefore has to be lexical and per-query (Section 4.4), not a "
        "split barrier."
    ))

    # ======================================================================
    # 4 Benchmark construction
    # ======================================================================
    extend(H1("4", "Benchmark construction: draft → rewrite → gate → freeze"))

    add(P(
        "The benchmark is not generated in one shot. It moves through four stages, each with a different "
        "failure mode and a different guarantee. The first stage is deterministic and cheap; the second is the "
        "only stage that adds new language; the third is the gate; the fourth fixes the qrels and writes the "
        "artifacts."
    ))
    add(PipelineDiagram())
    add(caption(
        "Figure 2. The benchmark pipeline. The gate (stage 3) and freeze (stage 4) are the two points where a "
        "bad query can stop the build. Everything downstream reads the frozen file and never re-drafts."
    ))

    # 4.1
    extend(H2("4.1", "Stage 1 — drafting the review sources"))
    add(P(
        "<font name=\"Consolas\" size=\"8\">tsr make-eval</font> without <font name=\"Consolas\" size=\"8\">--rewrites</font> calls "
        "<font name=\"Consolas\" size=\"8\">draft_eval_set()</font>. It takes the test split (or the whole gallery if the "
        "split has fewer images than requested), builds one candidate pool per category, and shuffles each pool "
        "with a local <font name=\"Consolas\" size=\"8\">random.Random(seed)</font>. Separate pools are important: a category "
        "with many rejected drafts must not consume candidates needed by the next category. A global "
        "<font name=\"Consolas\" size=\"8\">used</font> set prevents the same image from seeding two queries."
    ))
    add(P(
        "The draft query is a template around a lightly mangled annotation. "
        "<font name=\"Consolas\" size=\"8\">_paraphrase()</font> applies a small synonym table and swaps common function "
        "words (“the” → “this”, “with” → “alongside”), then strips punctuation. It is deliberately crude. The "
        "draft is a <i>starting point for review</i>, not a benchmark query, and the code says so in the docstring."
    ))
    add(P(
        "The output is <font name=\"Consolas\" size=\"8\">data/eval/draft_sources.json</font>: one record per qid with "
        "<font name=\"Consolas\" size=\"8\">image_id</font>, <font name=\"Consolas\" size=\"8\">category</font>, the annotation snapshot "
        "(captions, metaphors, title), but <b>no final query text</b>. qids are assigned in category order: "
        "q0001–q0080 visual, q0081–q0160 intent, q0161–q0240 metaphor (given the YAML order). The snapshot is "
        "what makes the later freeze able to detect stale sources."
    ))

    # 4.2
    extend(H2("4.2", "Stage 2 — the reviewed rewrite"))
    add(P(
        "For every draft qid, a new query text is written and stored as a <font name=\"Consolas\" size=\"8\">qid → text</font> "
        "mapping in <font name=\"Consolas\" size=\"8\">data/eval/rewrites/batch1.json</font> … "
        "<font name=\"Consolas\" size=\"8\">batch6.json</font> (40 qids each, 240 total). The rewrite is the actual query "
        "used in evaluation; the draft text never reaches a strategy."
    ))
    add(P(
        "The rewrite must do four things at once:"
    ))
    extend(bullets([
        "<b>Preserve the target.</b> The text must still identify the same source meme and the same category of "
        "intent (scene, message, or metaphor). Otherwise the grade-2 qrel would point at a different image than "
        "the query describes.",
        "<b>Read like a query.</b> A natural search phrase, not the dataset's annotation register "
        "(“Meme poster is conveying that …”).",
        "<b>Not borrow annotation wording.</b> This is the leakage constraint, and it shapes the vocabulary of "
        "every rewrite.",
        "<b>Stay within the category.</b> A visual query describes the scene without giving away the intended "
        "meaning; an intent query states the message without reciting the scene; a metaphor query names the "
        "symbol and its meaning.",
    ]))
    add(make_table(
        ["Category / qid", "Source annotation (abridged)", "Frozen rewrite"],
        [
            ["visual · q0001", "“God is holding up a wine cup.”",
             "“a robed ancient man with a halo clutching a small cup of wine”"],
            ["intent · q0081", "“Meme poster is annoyed with anti-vaxxer logic.”",
             "“a joke about how refusing a vaccine because it is not flawless is like refusing to cross a strong "
             "bridge and choosing to swim through rough water instead”"],
            ["metaphor · q0161", "metaphor “a cartoon figure”; meaning “Meme poster trying to explain the male psyche”",
             "“using a cartoon character sprinting into a tunnel to represent men chasing any scrap of affection”"],
        ],
        [78, 190, 219],
        font=S_TD_SMALL,
    ))
    add(caption("Table 4.1. Three real rewrites from the frozen set, one per category."))

    add(callout(
        "Who actually wrote the rewrites? The artifacts do not say.",
        "The code labels every record <font name=\"Consolas\" size=\"7.8\">query_origin: \"llm_rewrite\"</font> — that is the "
        "default of <font name=\"Consolas\" size=\"7.8\">reviewed_eval_set()</font> and all 240 records carry it. The CLI help "
        "calls the directory “reviewed query rewrites” and “written for human review”, while the generated "
        "walkthrough says each draft is “rewritten by hand”. There is no prompt log, model version, or reviewer "
        "sign-off in the repository. The defensible description is “reviewed rewrites”: text written per draft and "
        "accepted through the same gate as any other query. If a report needs a claim about authorship, record "
        "it; do not infer it from the label.",
        "amber",
    ))

    add(callout(
        "Two things the rewrite is not",
        "It is not <font name=\"Consolas\" size=\"7.8\">_paraphrase()</font>, the automatic synonym/stopword swap used to "
        "produce drafts. And it is not the runtime query expansion in "
        "<font name=\"Consolas\" size=\"7.8\">nlp/analyze.py</font>: that expansion runs at search time, feeds BM25 only, and "
        "never touches the frozen benchmark. Confusing the three is the most common source of “but the query was "
        "rewritten” misunderstandings.",
        "gray",
    ))

    # 4.3
    extend(H2("4.3", "Stage 3 — freezing the benchmark"))
    add(P(
        "<font name=\"Consolas\" size=\"8\">tsr make-eval --sources data/eval/draft_sources.json --rewrites "
        "data/eval/rewrites --force</font> calls <font name=\"Consolas\" size=\"8\">reviewed_eval_set()</font>. It merges the "
        "batch files, then processes sources in order. For each source it requires a non-empty rewrite, a known "
        "image, a live <font name=\"Consolas\" size=\"8\">test</font> split, an unchanged annotation snapshot, and a query that "
        "passes the leakage gate. If any source fails, the command prints every rejection reason and exits "
        "without writing anything."
    ))
    add(P(
        "The snapshot check is a prefix check: the draft stored a prefix of each caption list and the non-literal "
        "metaphor rows; the current record must still start with exactly those values (only fields present in "
        "the draft are compared). Appending a caption therefore keeps a reviewed query valid, while editing or "
        "removing one invalidates it, because the query would now describe text the gallery no longer says. This "
        "is what makes “the qid still points at the same evidence” checkable months later."
    ))
    add(P("The rejection vocabulary is small and explicit:"))
    add(code_block(
        "missing_rewrite   no text for this qid in any batch file\n"
        "unknown_image     image_id no longer present in the gallery\n"
        "not_test_split    the source's live split is trainval\n"
        "stale_source      the annotation snapshot no longer matches\n"
        "                  (build_eval_query returns None when leakage fails)"
    ))
    add(P(
        "On success it writes three files: <font name=\"Consolas\" size=\"8\">memecap_eval_v1.json</font> (queries, qrels and "
        "metadata), <font name=\"Consolas\" size=\"8\">memecap_eval_v1.qrels.jsonl</font> (one judgment per line), and "
        "<font name=\"Consolas\" size=\"8\">memecap_eval_v1.raw_captions.json</font> (the leaky diagnostic, Section 4.7). Each "
        "query record carries <font name=\"Consolas\" size=\"8\">metadata.split</font>, "
        "<font name=\"Consolas\" size=\"8\">metadata.image_id</font>, <font name=\"Consolas\" size=\"8\">metadata.query_origin</font> and the full "
        "<font name=\"Consolas\" size=\"8\">metadata.leakage</font> verdict."
    ))

    # 4.4
    extend(H2("4.4", "The leakage gate in detail"))
    add(P(
        "The gate is the core of the benchmark's credibility. It answers one question: does the query reuse "
        "text that a strategy could have indexed? It compares the query against two sets of documents: the "
        "source image's own annotation fields, and every annotation in the gallery. Both matter. The source check "
        "stops a query from copying the evidence for its own qrel; the gallery check stops a query from copying "
        "any other image's annotation, which would create an accidental lexical match elsewhere."
    ))
    add(P("The source documents are exactly what a strategy could index for that image:"))
    add(code_block(
        "img_captions[*]  +  meme_captions[*]  +  \"<metaphor> means <meaning>\"[*]  +  title"
    ))
    add(P(
        "A query passes only if <b>all five</b> checks pass. Each check targets a blind spot in the others."
    ))
    add(make_table(
        ["Check", "Rule", "Reject when", "Why it exists"],
        [
            ["Source n-gram", "8-gram token overlap with the source documents", "any shared 8-gram",
             "Catches a long verbatim phrase."],
            ["Gallery n-gram", "8-gram overlap with all gallery annotations", "any shared 8-gram",
             "Catches copying another image's annotation."],
            ["Short-doc containment", "whole short annotation (4–7 tokens) inside the query", "contained verbatim",
             "The 8-gram rule cannot see a 3-word caption pasted whole."],
            ["Longest source span", "longest contiguous token run shared with the source", "≥ 5 tokens",
             "The 8-gram rule misses a 5–7 token copy inside a longer sentence."],
            ["Token Jaccard", "set overlap with any source document", "≥ 0.60",
             "Length-independent; stops short captions being pasted with stopword edits."],
        ],
        [82, 160, 80, 165],
        font=S_TD_SMALL,
    ))
    add(caption(
        "Table 4.2. The five leakage checks. Thresholds come from configs/default.yaml: "
        "max_ngram_overlap 8, min_copy_tokens 5, Jaccard 0.6 (the Jaccard constant is in check_leakage)."
    ))
    add(P(
        "Every verdict is stored, not just the boolean: <font name=\"Consolas\" size=\"8\">max_ngram_overlap</font>, "
        "<font name=\"Consolas\" size=\"8\">strict_overlap</font>, <font name=\"Consolas\" size=\"8\">contained</font>, "
        "<font name=\"Consolas\" size=\"8\">gallery_contained</font>, <font name=\"Consolas\" size=\"8\">max_source_span</font>, "
        "<font name=\"Consolas\" size=\"8\">min_copy_tokens</font>, <font name=\"Consolas\" size=\"8\">jaccard</font>, "
        "<font name=\"Consolas\" size=\"8\">num_tokens</font>, <font name=\"Consolas\" size=\"8\">passed</font>, "
        "<font name=\"Consolas\" size=\"8\">checked_n</font>. An auditor can recompute any of them from the frozen query and "
        "the manifest without trusting the build."
    ))
    add(callout(
        "What the gate proves, and what it does not",
        "It proves the query does not lexically copy an indexed annotation. It does <b>not</b> prove the query is "
        "semantically original: a paraphrase with no shared 8-gram, no 5-token run and Jaccard below 0.6 can "
        "still restate the annotation's meaning. That is accepted deliberately — meaning must be preserved for "
        "the qrel to make sense — and the strict thresholds are chosen so the query cannot be answered by string "
        "matching alone. The gate is a lexical floor, not a plagiarism detector.",
        "blue",
    ))
    add(P(
        "The current frozen set is comfortable inside every threshold: all 240 queries pass; the largest Jaccard "
        "is 0.50 (limit 0.60); the longest shared source span is 4 tokens (limit 5); query length ranges from 10 "
        "to 35 tokens, median 17."
    ))

    # 4.5
    extend(H2("4.5", "Stage 4 — graded relevance (qrels)"))
    add(P(
        "Relevance is built by fixed rules, never by a model's judgement. The source image is grade 2. Grade 1 is "
        "assigned to another image only when one of three category-specific rules fires:"
    ))
    add(make_table(
        ["Category", "Grade-1 rule", "Implementation detail"],
        [
            ["visual", "Near-identical image caption.",
             "Token Jaccard between the two <font name=\"Consolas\" size=\"7.4\">img_captions[0]</font> values ≥ 0.8."],
            ["intent", "Identical intended meaning.",
             "Exact equality of normalised <font name=\"Consolas\" size=\"7.4\">meme_captions[0]</font> (strip + lower)."],
            ["metaphor", "Shared specific metaphor with the same meaning.",
             "Same lower-cased label, meaning sets equal or Jaccard ≥ 0.6, and label document frequency ≤ 10 "
             "(<font name=\"Consolas\" size=\"7.4\">eval.metaphor_max_df</font>). Literal meanings excluded."],
        ],
        [62, 150, 275],
        font=S_TD_SMALL,
    ))
    add(P(
        "Only the first caption is used for the visual and intent rules; metaphors use every non-literal row. "
        "The metaphor rule is deliberately the most restrictive: sharing a label is not enough, because labels "
        "such as “a man” or “having” are literal depictions rather than shared symbolism. The label must also be "
        "rare (document frequency ≤ 10) and carry the same meaning in both images."
    ))
    add(KeepTogether([
        P("The qrels file is one judgment per line:"),
        code_block('{"qid": "q0001", "image_id": "memecap_memes_yuirix", "grade": 2}'),
    ]))
    add(P(
        "Across the current 240 queries there are 339 judgments: 240 grade 2 and 99 grade 1. 225 queries have "
        "only their source image; the other 15 have between 2 and 19 relevant images (three metaphor queries "
        "reach 19). This distribution is a property of the corpus, not a bug, but it matters for metric "
        "interpretation and is discussed in Section 6.3."
    ))

    # 4.6
    extend(H2("4.6", "Artifacts produced by the construction stages"))
    add(make_table(
        ["Artifact", "Written by", "Contents"],
        [
            [mono("data/eval/draft_sources.json"), "make-eval (draft pass)",
             "240 qids, source image ids, categories, annotation snapshots; no query text."],
            [mono("data/eval/batches/batch1–6.json"), "review packaging",
             "The same sources split into six 40-item review batches."],
            [mono("data/eval/rewrites/batch1–6.json"), "review",
             "qid → final query text; merged by <font name=\"Consolas\" size=\"7.4\">_load_rewrites()</font>."],
            [mono("data/eval/memecap_eval_v1.json"), "make-eval (freeze pass)",
             "Frozen queries with relevant ids, grades, split, origin and leakage verdict."],
            [mono("data/eval/memecap_eval_v1.qrels.jsonl"), "make-eval",
             "One <font name=\"Consolas\" size=\"7.4\">{qid, image_id, grade}</font> per line."],
            [mono("data/eval/memecap_eval_v1.raw_captions.json"), "make-eval",
             "Diagnostic set: the same qrels with raw image captions as queries."],
        ],
        [208, 90, 189],
        font=S_TD_SMALL,
    ))
    add(caption(
        "All evaluation data lives under data/, which is git-ignored. The artifacts are local; the code that "
        "rebuilds them deterministically is versioned."
    ))

    # 4.7
    extend(H2("4.7", "The raw-caption diagnostic set"))
    add(P(
        "After freezing, <font name=\"Consolas\" size=\"8\">make-eval</font> builds a second query set by replacing each "
        "query with the source image's first raw caption (<font name=\"Consolas\" size=\"8\">img_captions[0]</font>). It keeps "
        "the frozen qrels, marks <font name=\"Consolas\" size=\"8\">query_origin: \"raw_caption_diagnostic\"</font>, and "
        "recomputes the leakage verdict. This set is <b>leaky by construction</b>: Strategy B indexes those exact "
        "captions, so its score is an upper bound on annotation-overlap exploitation, not a retrieval result."
    ))
    add(P(
        "The separation is enforced at several points. <font name=\"Consolas\" size=\"8\">tsr eval --queries "
        "…raw_captions.json</font> writes <font name=\"Consolas\" size=\"8\">reports/eval_&lt;key&gt;_raw_captions.json</font> instead of "
        "overwriting the main report. <font name=\"Consolas\" size=\"8\">tsr compare</font> refuses a diagnostic query file "
        "outright. <font name=\"Consolas\" size=\"8\">load_diagnostics()</font> only folds a diagnostic file into the report if "
        "its <font name=\"Consolas\" size=\"8\">eval_set</font> is the expected "
        "<font name=\"Consolas\" size=\"8\">&lt;main&gt;.raw_captions</font> and its query count matches the main table, so a stale "
        "file cannot be compared against a different benchmark."
    ))
    add(P(
        "The diagnostic is useful precisely because it is leaky: it shows how much a strategy would gain if "
        "queries reused annotation wording. In the current snapshot, B rises from mAP 0.7608 on the main set to "
        "0.8998 on raw captions, while A and C fall. That contrast is the point of the diagnostic, and it is why "
        "the diagnostic must never be merged into the main table."
    ))

    # ======================================================================
    # 5 Execution
    # ======================================================================
    extend(H1("5", "Execution: how a frozen query becomes a result"))

    add(P(
        "There are two entry points. <font name=\"Consolas\" size=\"8\">tsr eval</font> evaluates the selected strategies and "
        "writes one <font name=\"Consolas\" size=\"8\">reports/eval_&lt;key&gt;.json</font> per strategy; it is the tool for "
        "diagnostic query sets. <font name=\"Consolas\" size=\"8\">tsr compare</font> evaluates every available strategy on the "
        "frozen set and writes the combined <font name=\"Consolas\" size=\"8\">compare.json</font>, "
        "<font name=\"Consolas\" size=\"8\">compare.html</font> and <font name=\"Consolas\" size=\"8\">failures.html</font>. The web page "
        "reads only the combined report."
    ))
    add(code_block(
        "python -m tsr make-eval --sources data/eval/draft_sources.json \\\n"
        "                        --rewrites data/eval/rewrites --force\n"
        "python -m tsr eval --strategy all                 # per-strategy JSON\n"
        "python -m tsr compare --ablations --boot 1000     # combined report + tests\n"
        "python -m tsr eval --queries data/eval/memecap_eval_v1.raw_captions.json\n"
        "                                                  # diagnostic only"
    ))

    extend(H2("5.1", "Strategy loading and availability"))
    add(P(
        "<font name=\"Consolas\" size=\"8\">StrategyManager</font> loads shared resources lazily. In evaluation the strategy "
        "is asked to <font name=\"Consolas\" size=\"8\">ensure_loaded()</font>, which serialises concurrent callers and then "
        "reports <font name=\"Consolas\" size=\"8\">ready</font>. A strategy that cannot load is recorded with its error and an "
        "empty metric set; the other strategies still run. This matters for C: its text encoder is available on the "
        "serving host even when the captioning GPU is not, so an index can be present and the strategy ready. "
        "Conversely, a partially captioned gallery is rejected by the 95% coverage rule."
    ))

    extend(H2("5.2", "One retrieval run, two consumers"))
    add(P(
        "The runner wraps each strategy in <font name=\"Consolas\" size=\"8\">_CachedRetriever</font>. The cache key is "
        "<font name=\"Consolas\" size=\"8\">(query_text, k)</font>, and the same object is passed to both "
        "<font name=\"Consolas\" size=\"8\">evaluate_queries()</font> (which computes the headline metrics and the per-query "
        "rows) and <font name=\"Consolas\" size=\"8\">per_query_values()</font> (which produces the vectors the bootstrap "
        "needs). The second call hits the cache, so each query is searched once per strategy. This is not just "
        "an optimisation: it guarantees that the paired test is computed on the same ranked lists that produced "
        "the headline number. If a strategy raises, the runner logs it, counts an error, caches an empty list, and "
        "keeps going; the <font name=\"Consolas\" size=\"8\">num_errors</font> field records how often that happened."
    ))

    extend(H2("5.3", "Retrieval depth"))
    add(P(
        "The evaluator does not retrieve only <font name=\"Consolas\" size=\"8\">max(top_k)</font>. It uses:"
    ))
    add(code_block("depth = max(max(top_k), largest relevant set, 100)"))
    add(P(
        "With top_k = 1/5/10, the depth is at least 100. This prevents recall@R and mAP from being truncated when a "
        "query has more relevant images than 10 — the largest current relevant set is 19 — and gives AP a stable "
        "denominator. The cost is one deep search per query per strategy, which is why the cache matters."
    ))
    add(P(
        "The returned list is mapped from SQLite primary keys to <font name=\"Consolas\" size=\"8\">image_id</font> values and "
        "compared with the qrels set. Per query, the evaluator stores the rank of the first relevant hit, AP, RR, "
        "nDCG@10 and the full retrieved list; these row-level records are what <font name=\"Consolas\" size=\"8\">failures.html</font> "
        "and the per-query bootstrap consume."
    ))

    extend(H2("5.4", "Grouping and availability in the report"))
    add(P(
        "Every query contributes to two groupings: its category (visual / intent / metaphor) and its split (test). "
        "The report stores <font name=\"Consolas\" size=\"8\">per_category</font> and <font name=\"Consolas\" size=\"8\">per_split</font> with "
        "count, mAP and MRR. Because the benchmark is test-only, the split table has a single row; the grouping "
        "still exists so the report format does not change if a future set includes another split. The web page "
        "renders both tables and shows a dash rather than a fabricated row when a group is absent."
    ))

    # ======================================================================
    # 6 Metrics
    # ======================================================================
    extend(H1("6", "Metrics: definitions and aggregation"))

    add(P(
        "All metrics are computed per query and then macro-averaged over the query set. Let R be the set of "
        "relevant image ids (grades 1 and 2), g(d) the grade of document d, and π the ranked list returned by the "
        "strategy."
    ))
    add(make_table(
        ["Metric", "Definition", "Notes"],
        [
            ["P@k", "number of d in π[:k] that are in R, divided by k", "Denominator is k, even if fewer than k results exist."],
            ["R@k", "number of d in π[:k] that are in R, divided by |R|", "Saturates when |R| < k; read with recall@R."],
            ["AP", "(1 / min(|R|, |π|)) x sum over hits i of hits(i)/i", "With depth ≥ 100 and |R| ≤ 19, the denominator is |R|."],
            ["RR", "1 / rank of the first relevant hit, else 0", "Mean over queries is MRR."],
            ["nDCG@k", "DCG@k / IDCG@k, gain = 2^g - 1, discount = log2(i+1)", "Graded; grade 2 contributes 3x grade 1."],
            ["recall@R", "number of d in π[:|R|] that are in R, divided by |R|", "Not truncated by the reported @k cuts."],
            ["median rank", "median first-relevant rank over queries with a hit", "Misses are excluded from the median."],
            ["misses", "count of queries with no relevant hit in the retrieved depth", "Reported as a count, not a rate."],
        ],
        [64, 236, 187],
        font=S_TD_SMALL,
    ))
    add(caption("Table 6.1. Metric primitives in evaluate.py. The aggregation is a simple mean over queries unless stated."))
    add(P(
        "The headline table also reports bootstrap confidence intervals for mAP, MRR and nDCG@10, and p50/p95 "
        "search latency. Latency is measured around <font name=\"Consolas\" size=\"8\">strategy.search()</font> only, on the host "
        "that ran compare. It is useful for spotting a pathological cost (B's cross-encoder rerank, for example) "
        "but it is not a controlled serving benchmark: hardware, thread pinning, cache warmth and concurrent load "
        "all move it."
    ))

    extend(H2("6.1", "Why mAP and recall@R are reported alongside the @k cuts"))
    add(P(
        "P@k and R@k at 1/5/10 are familiar but easy to misread when most queries have one relevant image. For a "
        "singleton query, P@1 is 1 only if the source is ranked first, R@10 is 1 if it appears anywhere in the top "
        "10, and mAP is 1/rank. The three views disagree by design. Reporting mAP and recall@R as well makes the "
        "rank distribution visible instead of collapsing it into a single cut."
    ))

    extend(H2("6.2", "Aggregation and grouping"))
    add(P(
        "Each metric is averaged over all queries; the report additionally stores per-category and per-split "
        "means for mAP and MRR. Per-category means are unweighted (each category has 80 queries in the current "
        "set, so they also sum to the overall mean). The per-query rows are retained in "
        "<font name=\"Consolas\" size=\"8\">strategies.&lt;key&gt;.per_query</font>, which is what allows failures.html and the "
        "paired bootstrap to work without re-running retrieval."
    ))

    extend(H2("6.3", "The singleton-relevance caveat"))
    add(P(
        "225 of 240 queries have exactly one relevant image. For those, mAP equals reciprocal rank, and the "
        "difference between mAP and MRR is entirely driven by the 15 queries with grade-1 neighbours. This is not "
        "a flaw in the metric code; it is the shape of the benchmark. It does mean that small mAP differences on "
        "the main table should be read as “rank of the source image” differences, and that the metaphor category "
        "(where grade-1 sets are larger) carries more of the graded-relevance signal than its 80-query share "
        "suggests."
    ))

    # ======================================================================
    # 7 Statistics
    # ======================================================================
    extend(H1("7", "Statistical comparison"))

    add(P(
        "The comparison asks whether one strategy is better than another on the same 240 queries. Because the "
        "queries are shared, the right unit is the per-query difference, not two independent means. The code "
        "implements three layers: bootstrap confidence intervals for each strategy, a paired bootstrap test for "
        "each pair and metric, and one Holm–Bonferroni correction across the whole family."
    ))
    add(StatsDiagram())
    add(caption(
        "Figure 3. From retrieval to inference. The per-query metric vectors produced during the cached run feed "
        "both the confidence intervals and the paired tests; the report is a snapshot of that one run."
    ))

    extend(H2("7.1", "Bootstrap confidence intervals"))
    add(P(
        "<font name=\"Consolas\" size=\"8\">bootstrap_ci()</font> resamples the n per-query values with replacement "
        "<font name=\"Consolas\" size=\"8\">n_boot</font> times (default 1,000), takes the mean of each resample, and reports "
        "the 2.5th and 97.5th percentiles around the observed mean. The seed is fixed (42 by default), so the "
        "intervals are reproducible. The interval describes uncertainty over the query sample, not over hardware "
        "or model weights."
    ))

    extend(H2("7.2", "Paired bootstrap test"))
    add(P(
        "For strategies A and B, <font name=\"Consolas\" size=\"8\">paired_bootstrap_test()</font> forms the per-query "
        "difference <font name=\"Consolas\" size=\"8\">d_i = B_i − A_i</font>, resamples the differences 1,000 times, and "
        "reports the mean delta with a 95% percentile interval and a two-sided p-value:"
    ))
    add(code_block(
        "p = 2 x min( P(boot ≤ 0), P(boot ≥ 0) )\n"
        "p = max(p, 1 / n_boot)"
    ))
    add(P(
        "The floor matters. A finite bootstrap cannot resolve a p-value below 1/n_boot; reporting 0 would "
        "overclaim certainty. With 1,000 resamples the smallest reportable p is 0.001, and after Holm the smallest "
        "adjusted value is still bounded by that resolution. The current report's significant comparisons show "
        "p_holm = 0.015, which is the Holm-adjusted value of the 0.001 floor across the family (see below), not a "
        "typo."
    ))

    extend(H2("7.3", "Holm–Bonferroni across one family"))
    add(P(
        "The five compared metrics are mAP, MRR, nDCG@10, P@1 and R@10. With three ready strategies there are "
        "three pairs × five metrics = 15 tests. Correcting separately inside each pair would treat the three "
        "comparisons as independent families; the report claims one family, so "
        "<font name=\"Consolas\" size=\"8\">holm_bonferroni()</font> is applied once to all 15 raw p-values. The adjusted "
        "value is the running maximum of <font name=\"Consolas\" size=\"8\">(m − i) · p_(i)</font> in sorted order, capped at 1. "
        "A comparison is marked significant when <font name=\"Consolas\" size=\"8\">p_holm &lt; 0.05</font>."
    ))
    add(P(
        "Holm controls the family-wise error rate without assuming independence, which is the right trade-off "
        "here: the five metrics are correlated (they are computed from the same ranked lists), and a Bonferroni "
        "correction would be needlessly conservative, while no correction would inflate the chance of a false "
        "positive across 15 tests."
    ))

    extend(H2("7.4", "Win / tie / loss"))
    add(P(
        "Alongside the delta, each comparison stores the per-query win/tie/loss count: B wins a query when its "
        "metric is higher, loses when lower, ties within 1e−9. This is descriptive, not a test. It is useful when "
        "the mean delta is driven by a few large gains rather than a broad shift, and it appears in the report "
        "next to the p-value so a reader can see both magnitude and breadth."
    ))

    extend(H2("7.5", "Ablations"))
    add(P(
        "<font name=\"Consolas\" size=\"8\">--ablations</font> re-instantiates a strategy with a modified configuration, runs "
        "the same frozen queries, and records the variant's mAP, CI and config hash. Variants are configuration "
        "changes, not retrained models: they isolate which signal or component contributes."
    ))
    add(make_table(
        ["Variant", "Change", "Current mAP"],
        [
            ["clip + prompt ensemble", "A with prompt ensemble enabled", "0.4415"],
            ["clip_nlp − BM25", "B with BM25 weight 0", "0.7040"],
            ["clip_nlp − dense doc", "B with dense document weight 0", "0.7043"],
            ["clip_nlp − CLIP", "B with CLIP weight 0", "0.7802"],
            ["clip_nlp − reranker", "B with cross-encoder rerank disabled", "0.6887"],
            ["vlm only (no CLIP)", "C with clip_weight 0", "0.3922"],
            ["vlm + CLIP 0.35", "C with clip_weight 0.35", "0.4720"],
        ],
        [120, 270, 97],
        align=["l", "l", "r"],
        font=S_TD_SMALL,
    ))
    add(caption(
        "Table 7.1. Ablation variants and their mAP in the local snapshot of " + ARTIFACT_DATE + ". "
        "The base B mAP is 0.7608, so removing CLIP improves B slightly in this run; that is a finding to report, "
        "not a reason to hide the variant."
    ))
    add(P(
        "An ablation is best-effort: a variant that fails to load is recorded as unavailable and does not abort "
        "the comparison. Each successful variant stores <font name=\"Consolas\" size=\"8\">config_hash</font>, so a reader can "
        "tell which configuration produced the number."
    ))

    extend(H2("7.6", "What the statistics do not establish"))
    extend(bullets([
        "The 240 queries are a sample from a 558-image test split, not a random sample of all possible meme-search "
        "intents. The CI covers query sampling variability under that construction, not distribution shift.",
        "The paired test assumes queries are exchangeable. Stratification (80 per category, drawn separately) means "
        "the effective sample is not a simple random sample; per-category tables are the honest place to look for "
        "category-specific effects.",
        "Significance is not effect size. A p_holm below 0.05 says the direction is unlikely under the null; the "
        "delta and its interval say whether the difference is worth caring about.",
        "The bootstrap resolution floor means very small p-values are not distinguishable; do not report p < 0.001 "
        "as if it were exact.",
    ]))

    # ======================================================================
    # 8 Reporting and serving
    # ======================================================================
    extend(H1("8", "Reporting and serving the evaluation"))

    add(P(
        "<font name=\"Consolas\" size=\"8\">compare_strategies()</font> assembles one report dictionary and writes three "
        "files: <font name=\"Consolas\" size=\"8\">compare.json</font> (machine-readable, the source of truth), "
        "<font name=\"Consolas\" size=\"8\">compare.html</font> (a self-contained human report) and "
        "<font name=\"Consolas\" size=\"8\">failures.html</font> (the ten worst queries per strategy, with relevant and top-5 "
        "retrieved images side by side). The web app serves the static pages; it does not recompute metrics."
    ))
    add(make_table(
        ["Report field", "Meaning"],
        [
            [mono("eval_set"), "Query-file stem, e.g. memecap_eval_v1. This is what the page shows as “Eval set”."],
            [mono("num_queries"), "len(queries) at compare time. The page shows this as “Evaluation queries”."],
            [mono("top_k / n_boot / seed"), "Evaluation configuration used; makes the report self-describing."],
            [mono("strategies.<key>.metrics"), "Mean metrics plus median rank, misses and num_queries."],
            [mono("strategies.<key>.ci"), "Bootstrap mean/low/high for each metric."],
            [mono("strategies.<key>.per_query"), "Per-query rank, AP, RR, nDCG@10 and the full retrieved list."],
            [mono("strategies.<key>.per_category"), "mAP and MRR by query category."],
            [mono("strategies.<key>.per_split"), "mAP and MRR by source split (test in the current set)."],
            [mono("pairwise"), "Per pair and metric: delta, CI, raw p, Holm p, win/tie/loss."],
            [mono("ablations"), "Variant results keyed by human-readable label."],
            [mono("diagnostics"), "Raw-caption mAP/MRR/nDCG@10, only if the matching diagnostic files exist."],
            [mono("probe"), "Gallery size, leakage pass/flag counts, VLM caption count under provenance filters."],
            [mono("failures"), "Worst queries with pk/path for the HTML strip; not part of the schema."],
        ],
        [150, 337],
        font=S_TD_SMALL,
    ))
    add(caption("Table 8.1. compare.json fields. EvalSummaryResponse mirrors every field except failures, "
                "which exists only so the HTML failure gallery can link back to images."))

    extend(H2("8.1", "The /evaluation page"))
    add(P(
        "The page is a static shell (<font name=\"Consolas\" size=\"8\">web/evaluation.html</font>) plus "
        "<font name=\"Consolas\" size=\"8\">web/eval.js</font>. On load it fetches <font name=\"Consolas\" size=\"8\">GET /api/eval/summary</font>; the "
        "API reads <font name=\"Consolas\" size=\"8\">data/reports/compare.json</font> through <font name=\"Consolas\" size=\"8\">load_report()</font> "
        "and returns it as <font name=\"Consolas\" size=\"8\">EvalSummaryResponse</font>. If the file is missing, the endpoint "
        "returns <b>404</b> with the instruction to run <font name=\"Consolas\" size=\"8\">tsr compare</font>; the HTML "
        "endpoints <font name=\"Consolas\" size=\"8\">/api/eval/report.html</font> and "
        "<font name=\"Consolas\" size=\"8\">/api/eval/failures.html</font> behave the same way. The page renders:"
    ))
    extend(bullets([
        "<b>Overall</b> — strategy rows with mAP [CI], MRR, nDCG@10, P@1, R@10, median rank and p50/p95 latency; "
        "unavailable strategies show the recorded error.",
        "<b>By category</b> — mAP/MRR for visual, intent and metaphor.",
        "<b>By split</b> — the report's per_split table (test).",
        "<b>Pairwise</b> — delta, interval, raw and Holm p, win/tie/loss.",
        "<b>Ablations</b> — variant rows when <font name=\"Consolas\" size=\"8\">--ablations</font> was used.",
        "<b>Diagnostics</b> — raw-caption mAP next to the main-set mAP, explicitly labelled leaky.",
        "<b>Probe</b> — gallery size, leakage pass/flag counts, VLM coverage.",
        "<b>Links</b> — the full HTML report and the failure gallery.",
    ]))
    add(callout(
        "Reports are snapshots, not live results",
        "If you change a strategy, rebuild an index, or edit a config, the page keeps showing the old numbers until "
        "<font name=\"Consolas\" size=\"7.8\">tsr compare</font> runs again. There is no cache invalidation and no version check "
        "against the code or index. Always read <font name=\"Consolas\" size=\"7.8\">generated_at</font> before quoting the page, "
        "and regenerate after any change to a strategy or config.",
        "red",
    ))

    extend(H2("8.2", "The probe block"))
    add(P(
        "The report's <font name=\"Consolas\" size=\"8\">probe</font> is a lightweight consistency check: "
        "<font name=\"Consolas\" size=\"8\">num_images</font> is <font name=\"Consolas\" size=\"8\">store.count()</font>; "
        "<font name=\"Consolas\" size=\"8\">leakage_passed</font>/<font name=\"Consolas\" size=\"8\">flagged</font> count the frozen set's "
        "verdicts; <font name=\"Consolas\" size=\"8\">vlm_captions</font> counts VLM documents under the current provenance "
        "filters. If <font name=\"Consolas\" size=\"8\">num_images</font> is smaller than expected, the index or store is "
        "incomplete. If <font name=\"Consolas\" size=\"8\">leakage_flagged</font> is non-zero, the set is not the frozen one. These "
        "are guard rails, not metrics."
    ))

    # ======================================================================
    # 9 Reproducibility
    # ======================================================================
    extend(H1("9", "Reproducibility, determinism and configuration"))

    add(P(
        "Three kinds of determinism are at work, and they are independent:"
    ))
    extend(bullets([
        "<b>Sampling determinism.</b> <font name=\"Consolas\" size=\"8\">draft_eval_set()</font> uses "
        "<font name=\"Consolas\" size=\"8\">random.Random(seed)</font> with seed 42, shuffling each category's candidate pool "
        "separately. The same manifest and config produce the same draft qids and source images.",
        "<b>Statistical determinism.</b> <font name=\"Consolas\" size=\"8\">bootstrap_ci()</font> and "
        "<font name=\"Consolas\" size=\"8\">paired_bootstrap_test()</font> use <font name=\"Consolas\" size=\"8\">np.random.default_rng(seed)</font> "
        "with seed 42, so intervals and p-values are reproducible for the same per-query values.",
        "<b>Content determinism.</b> The freeze compares the draft's annotation snapshot with the live record and "
        "rejects a stale source; the leakage verdict is stored. A rebuilt set from the same inputs and rewrites "
        "has the same content, even if file metadata such as the export timestamp differs.",
    ]))

    extend(H2("9.1", "Configuration knobs"))
    add(make_table(
        ["Option", "Default", "Role"],
        [
            [mono("eval.set_name"), "memecap_eval_v1", "Stem for the frozen files and the report's eval_set label."],
            [mono("eval.num_queries"), "240", "Default draft size when --num is omitted."],
            [mono("eval.per_category"), "80 / 80 / 80", "Stratification; scaled proportionally if --num differs."],
            [mono("eval.top_k"), "[1, 5, 10]", "Reported cuts. Retrieval depth is at least 100 regardless."],
            [mono("eval.seed"), "42", "Draft sampling and bootstrap seed."],
            [mono("eval.bootstrap"), "1000", "Resamples for CIs and paired tests."],
            [mono("eval.max_ngram_overlap"), "8", "Leakage n-gram size."],
            [mono("eval.min_copy_tokens"), "5", "Longest shared source span allowed (must be < this)."],
            [mono("eval.metaphor_max_df"), "10", "Shared metaphor labels must be rarer than this to earn grade 1."],
        ],
        [130, 96, 261],
        font=S_TD_SMALL,
    ))
    add(P(
        "Runtime-adjustable knobs (strategy weights, rerank settings) live in "
        "<font name=\"Consolas\" size=\"8\">settings.py</font>; build-time and structural options live in "
        "<font name=\"Consolas\" size=\"8\">configs/default.yaml</font>. Changing either without rerunning compare leaves the "
        "report describing the old configuration."
    ))

    extend(H2("9.2", "What invalidates what"))
    add(make_table(
        ["Change", "Required action"],
        [
            ["New or edited rewrites / sources", "Rerun <font name=\"Consolas\" size=\"7.4\">make-eval --sources --rewrites --force</font>, then eval and compare."],
            ["Annotation text or caption order", "The freeze will reject stale sources; rebuild the drafts or update the snapshot."],
            ["Strategy weights, rerank, prompt settings", "Rerun compare (and ablations if relevant); no re-freeze needed."],
            ["Index rebuild (same manifest)", "Rerun eval/compare; qrels and queries are unchanged."],
            ["Dataset rebuild (new manifest)", "Rebuild everything, including the benchmark; image ids may change."],
        ],
        [170, 317],
        font=S_TD_SMALL,
    ))

    extend(H2("9.3", "Tests that protect the evaluation code"))
    add(P(
        "The suite covers the pieces most likely to drift: draft stratification and per-category pools, leakage "
        "gating (including short-document containment and stale sources), graded qrels, bootstrap and Holm "
        "helpers, report assembly and diagnostic separation. Unit tests use fake encoders and never download a "
        "model, so they run in CI without GPU or network. Run:"
    ))
    add(code_block("PYTHONPATH=src pytest\nPYTHONPATH=src ruff check src tests"))

    # ======================================================================
    # 10 Audit notes
    # ======================================================================
    extend(H1("10", "Audit notes: what a careful reader should check"))

    add(P(
        "The following are not failures of the pipeline; they are properties a reviewer should know before "
        "quoting a number. Each has a concrete implication."
    ))
    notes = [
        ("Documentation counts disagree with the store.",
         "README says 6,382 images and 559 test images; the manifest and SQLite report 6,378 and 558. Fix the prose "
         "or state the pre-dedup annotation count explicitly."),
        ("Rewrite authorship is not recorded.",
         "The tag says llm_rewrite; the docs say “by hand”. The gate certifies non-copying, not authorship. Add a "
         "provenance note if the report makes a claim about who wrote the queries."),
        ("The benchmark covers 43% of the test split.",
         "240 distinct source images out of 558. That is a large sample but not the whole split; category-level "
         "estimates (80 queries each) are correspondingly noisier than the pooled number."),
        ("Most queries have one relevant image.",
         "225/240 are singletons, so mAP equals reciprocal rank for them. Graded nDCG is driven by the 15 queries "
         "with grade-1 neighbours, especially the metaphor category."),
        ("The gate is lexical.",
         "A semantic paraphrase with no shared n-gram passes. This is intended, but it means “leakage-free” is a "
         "lexical claim, not a semantic guarantee."),
        ("The diagnostic is not a result.",
         "Raw-caption mAP is an upper bound for annotation overlap. Never put it in the main table; the code "
         "refuses to compare on it, and the report labels it separately."),
        ("Latency is host- and run-specific.",
         "p50/p95 are useful for relative cost, not for a serving SLA. Thread pinning and cache warmth change them."),
        ("The p-value floor is 0.001.",
         "With 1,000 resamples, p_holm = 0.015 is a family-adjusted floor, not an exact probability. Do not "
         "describe it as p < 0.001."),
        ("The report has no code or index version.",
         "generated_at is the only freshness signal. A report generated before a strategy change looks identical "
         "to a current one. Consider recording the commit and index meta hash in compare.json."),
        ("Ablations are configuration variants.",
         "They are not retrained models. “B − CLIP” means the CLIP weight was set to zero, not that a new model "
         "was trained without CLIP."),
    ]
    rows = [[f"<b>{i+1}.</b> {title}", implication] for i, (title, implication) in enumerate(notes)]
    add(make_table(["Observation", "Implication"], rows, [202, 285], font=S_TD_SMALL))

    # ======================================================================
    # Appendix A
    # ======================================================================
    extend(H1("A", "Appendix A — Glossary and file map"))

    extend(H2("A.1", "Glossary"))
    add(make_table(
        ["Term", "Meaning in this codebase"],
        [
            ["qid", "Query identifier, q0001–q0240, assigned at draft time and preserved through the freeze."],
            ["qrels", "Graded relevance judgments: one (qid, image_id, grade) row per relevant image."],
            ["grade 2 / grade 1", "Source image / rule-derived neighbour. No grade 3; no model judge."],
            ["leak", "A query that shares a lexical span with an indexed annotation (source or gallery)."],
            ["freeze", "The one-way transition from reviewed drafts to the final query and qrel files."],
            ["diagnostic", "The raw-caption query set, leaky by construction, reported separately."],
            ["ablation", "A strategy re-instantiated with one configuration change, same queries."],
            ["paired bootstrap", "Resampling per-query differences to test whether B−A is centered on zero."],
            ["Holm", "A step-down family-wise correction applied once across all pair × metric tests."],
        ],
        [110, 377],
        font=S_TD_SMALL,
    ))

    extend(H2("A.2", "Where each responsibility lives"))
    add(make_table(
        ["Path", "Responsibility"],
        [
            [mono("src/tsr/evaluate.py"), "EvalQuery model; save/load; metric primitives; bootstrap; leakage checks; drafting; qrels; freezing."],
            [mono("src/tsr/compare.py"), "Strategy runner and cache; per-query values; paired tests; Holm; ablations; report/HTML writers."],
            [mono("src/tsr/cli.py"), "make-eval / eval / compare commands; diagnostic filename labels; refusal rules."],
            [mono("src/tsr/schemas.py"), "EvalSummaryResponse, the shape the API promises the UI."],
            [mono("src/tsr/api.py"), "/api/eval/summary, /api/eval/report.html, /api/eval/failures.html, /evaluation."],
            [mono("web/eval.js"), "Rendering of overall, category, split, pairwise, ablation, diagnostic and probe tables."],
            [mono("configs/default.yaml"), "eval.* knobs: set name, counts, seed, bootstrap, thresholds."],
            [mono("data/eval/"), "Draft sources, review batches, rewrites, frozen queries, qrels, diagnostic."],
            [mono("data/reports/"), "compare.json/html, failures.html, per-strategy eval_<key>.json."],
            [mono("tests/test_evaluate.py"), "Drafting, leakage, qrels, reviewed-set gating."],
            [mono("tests/test_compare.py"), "Report assembly, diagnostics, pairwise behaviour."],
        ],
        [150, 337],
        font=S_TD_SMALL,
    ))

    # ======================================================================
    # Appendix B — worked example
    # ======================================================================
    extend(H1("B", "Appendix B — One query, end to end"))

    add(P(
        "This is the full life of q0001, a visual query. Every value below is quoted from the local artifacts; it "
        "is the clearest way to see how the stages connect."
    ))
    add(make_table(
        ["Stage", "What happens", "Value"],
        [
            ["Source", "Meme-Cap annotation for <font name=\"Consolas\" size=\"7.4\">memecap_memes_yuirix</font> (test split).",
             "img_caption: “God is holding up a wine cup.”<br/>meme_caption: “Meme poster is conveying that God gave us all water then put salt in it so we can't drink it.”"],
            ["Draft", "Template around <font name=\"Consolas\" size=\"7.4\">img_captions[0]</font>, paraphrase applied; qid assigned.",
             "Stored in draft_sources.json as q0001 / visual; the draft text is a starting point, not the query."],
            ["Rewrite", "New text written for the qid.",
             "“a robed ancient man with a halo clutching a small cup of wine”"],
            ["Gate", "Five lexical checks against source and gallery.",
             "8-gram overlap 0; strict overlap 0; no contained short doc; max source span 0; Jaccard 0.20. Pass."],
            ["Qrels", "Grade 2 for the source; grade-1 rules evaluated.",
             "Only <font name=\"Consolas\" size=\"7.4\">memecap_memes_yuirix</font> (grade 2). No other image has a caption with Jaccard ≥ 0.8."],
            ["Freeze", "Query and qrels written; qid preserved.",
             "memecap_eval_v1.json + one qrels line."],
            ["Retrieval", "A / B / C each return a ranked list at depth ≥ 100.",
             "A: rank 14 · B: rank 2 · C: rank 3"],
            ["Scoring", "AP/RR/nDCG computed from the rank; vectors feed the bootstrap.",
             "A: AP 0.0714, RR 0.0714 · B: AP 0.50, RR 0.50 · C: AP 0.3333, RR 0.3333"],
        ],
        [60, 185, 242],
        font=S_TD_SMALL,
    ))
    add(P(
        "For this query the three strategies disagree sharply: B's annotation index places the source second, "
        "while CLIP alone places it 14th. Because the query is a singleton, AP is exactly 1/rank; the bootstrap "
        "will use those three numbers as one of its 240 paired observations. This is the unit of analysis the whole "
        "report is built from."
    ))
    add(callout(
        "Why q0001 is a useful sanity check",
        "The query avoids the annotation's proper noun (“God”) and its joke (“water… salt”). It describes the "
        "picture instead — robe, halo, cup — which is exactly what a visual query should do, and it is why A can "
        "still retrieve the source at rank 14 despite never seeing the annotation. If the gate were removed, a "
        "query copying “God is holding up a wine cup” would flatter B without measuring visual retrieval.",
        "teal",
    ))

    # ======================================================================
    # Appendix C — current snapshot
    # ======================================================================
    extend(H1("C", "Appendix C — Current local snapshot"))
    add(P(
        "These are the values in <font name=\"Consolas\" size=\"8\">data/reports/compare.json</font> generated "
        + ARTIFACT_DATE + " at 15:12 UTC. They describe the local artifacts at that moment, not a permanent "
        "result. All three strategies were ready; the gallery had 6,378 images and 6,378 VLM captions."
    ))
    add(make_table(
        ["Strategy", "P@1", "R@10", "mAP", "MRR", "nDCG@10", "median", "misses", "p50 / p95 (ms)"],
        [
            ["A clip", "0.3583", "0.6074", "0.4395", "0.4461", "0.4744", "2.0", "46", "14.4 / 20.8"],
            ["B clip_nlp", "0.7000", "0.8702", "0.7608", "0.7647", "0.7804", "1.0", "13", "225.6 / 285.9"],
            ["C vlm", "0.3833", "0.6259", "0.4519", "0.4632", "0.4881", "2.0", "38", "35.3 / 41.6"],
        ],
        [72, 45, 45, 46, 46, 52, 45, 45, 91],
        align=["l", "r", "r", "r", "r", "r", "r", "r", "r"],
        font=S_TD_SMALL,
    ))
    add(caption("Table C.1. Headline metrics for the frozen 240-query set. CIs are in the JSON; the page shows them on hover."))
    add(make_table(
        ["Strategy", "visual mAP", "intent mAP", "metaphor mAP"],
        [
            ["A clip", "0.3153", "0.4444", "0.5588"],
            ["B clip_nlp", "0.6313", "0.7389", "0.9121"],
            ["C vlm", "0.2469", "0.5390", "0.5697"],
        ],
        [100, 129, 129, 129],
        align=["l", "r", "r", "r"],
        font=S_TD_SMALL,
    ))
    add(caption("Table C.2. mAP by category. B's advantage is largest on metaphor, where its documents carry the human-written meaning."))
    add(make_table(
        ["Comparison", "mAP delta", "p (raw)", "p (Holm)", "Verdict"],
        [
            ["B − A", "+0.3213", "0.001", "0.015", "B significantly better on all five metrics"],
            ["B − C", "+0.3089", "0.001", "0.015", "B significantly better on all five metrics"],
            ["A − C", "-0.025 to -0.012", "> 0.05", "> 0.05", "No significant difference on any of the five"],
        ],
        [86, 88, 74, 74, 165],
        font=S_TD_SMALL,
    ))
    add(caption("Table C.3. Pairwise summary. The Holm family is 3 pairs × 5 metrics = 15 tests."))
    add(make_table(
        ["Diagnostic (raw captions)", "mAP", "Main-set mAP", "Reading"],
        [
            ["A clip", "0.3546", "0.4395", "Worse with raw captions; no annotation index to exploit."],
            ["B clip_nlp", "0.8998", "0.7608", "Rises sharply: it is matching text it indexed. Leaky upper bound."],
            ["C vlm", "0.2591", "0.4519", "Falls; raw captions are a poor query register for the VLM documents."],
        ],
        [100, 60, 80, 247],
        align=["l", "r", "r", "l"],
        font=S_TD_SMALL,
    ))
    add(caption("Table C.4. Diagnostic table. It is deliberately separate and must not be quoted as retrieval quality."))
    add(callout(
        "Reading the snapshot responsibly",
        "The gap B−A is large and consistent across metrics, which is what the paired test is designed to detect. "
        "The B−C gap is similar in size. A−C is not significant. The ablation table in Section 7.5 shows that "
        "removing CLIP from B slightly improves mAP in this run (0.7802 vs 0.7608), so “B = CLIP + NLP” is not a "
        "claim that every component helps. Report the components, not the slogan.",
        "blue",
    ))

    # ======================================================================
    # Appendix D — commands
    # ======================================================================
    extend(H1("D", "Appendix D — Command reference"))
    add(code_block(
        "# Build the gallery and strategy indexes\n"
        "python -m tsr prepare\n"
        "python -m tsr build --strategy gallery\n"
        "python -m tsr build --strategy all\n"
        "\n"
        "# Freeze the benchmark from reviewed sources\n"
        "python -m tsr make-eval --sources data/eval/draft_sources.json \\\n"
        "                        --rewrites data/eval/rewrites --force\n"
        "\n"
        "# Evaluate\n"
        "python -m tsr eval --strategy clip,clip_nlp,vlm\n"
        "python -m tsr compare --ablations --boot 1000\n"
        "\n"
        "# Leaky diagnostic (separate report, never the main table)\n"
        "python -m tsr eval --queries data/eval/memecap_eval_v1.raw_captions.json\n"
        "\n"
        "# Serve the report\n"
        "python -m tsr serve          # http://localhost:8000/evaluation\n"
        "\n"
        "# Tests and lint\n"
        "PYTHONPATH=src pytest\n"
        "PYTHONPATH=src ruff check src tests"
    ))
    add(Spacer(1, 8))
    add(P(
        "All commands accept <font name=\"Consolas\" size=\"8\">--config path.yaml</font> or respect "
        "<font name=\"Consolas\" size=\"8\">TSR_CONFIG</font>; environment overrides use "
        "<font name=\"Consolas\" size=\"8\">TSR_</font> plus the upper-cased dotted path, joined by "
        "<font name=\"Consolas\" size=\"8\">__</font>. The evaluation artifacts are written under "
        "<font name=\"Consolas\" size=\"8\">data/eval/</font> and <font name=\"Consolas\" size=\"8\">data/reports/</font>, both ignored by git."
    ))
    add(Spacer(1, 14))
    add(HRFlowable(width="100%", thickness=0.6, color=RULE))
    add(Spacer(1, 6))
    add(Paragraph(
        f"Document version 1.0 · generated {DOC_DATE} from commit {COMMIT}. "
        "The PDF is generated by docs/make_evaluation_pdf.py; edit that script, not the PDF.",
        S_SMALL,
    ))

    return story


compare_placeholder = mono("compare.json", 7.5)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def main() -> None:
    doc = Doc(
        PDF_PATH,
        leftMargin=LM,
        rightMargin=RM,
        topMargin=TM,
        bottomMargin=BM,
        title="Evaluation Architecture and Benchmark Pipeline",
        author="Meme-Cap T2I Retrieval — COMP5523",
        subject="How the 240-query benchmark is built, executed, tested and served",
        creator="docs/make_evaluation_pdf.py",
    )
    frame = Frame(LM, BM, CONTENT_W, PAGE_H - TM - BM, id="body", leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)
    doc.addPageTemplates([
        PageTemplate(id="Cover", frames=[frame], onPage=draw_cover),
        PageTemplate(id="Body", frames=[frame], onPage=lambda c, d: None),
    ])
    story = build_story()
    doc.multiBuild(story, canvasmaker=NumberedCanvas)
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
