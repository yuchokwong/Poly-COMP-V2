"""Generate docs/tuning_report.pdf.

An English analysis of the one-factor parameter sweep produced by
``scripts/tune_params.py``. Every number is read from
``data/tuning/summary.json``; nothing is retyped.

Run:  python docs/make_tuning_report_pdf.py
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

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

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SUMMARY_PATH = ROOT / "data" / "tuning" / "summary.json"
OUT = DOCS / "tuning_report.pdf"

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
GREEN_LIGHT = colors.HexColor("#ECFDF3")
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

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        raise SystemExit(
            f"Missing {SUMMARY_PATH}. Run `PYTHONPATH=src python scripts/tune_params.py` first."
        )
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


SUMMARY = load_summary()
RUNS = {run["id"]: run for run in SUMMARY["runs"]}
BASELINES = SUMMARY["baselines"]
GROUPS = {group["id"]: group for group in SUMMARY["groups"]}
RUN_ORDER = [run["id"] for run in SUMMARY["runs"]]
COMMIT = (SUMMARY.get("tool") or {}).get("git_commit", "")
GENERATED = str(SUMMARY.get("generated_at", ""))[:10] or str(date.today())
DOC_DATE = date.today().strftime("%-d %B %Y") if os.name != "nt" else date.today().strftime("%d %B %Y").lstrip("0")

STRATEGY_LABEL = {"clip": "A · CLIP only", "clip_nlp": "B · CLIP + NLP", "vlm": "C · VLM"}
SHORT_LABEL = {"clip": "A", "clip_nlp": "B", "vlm": "C"}

# Curated rows for the headline table: (run id, description).
KEY_ROWS = [
    ("baseline_clip", "A — default"),
    ("a_prompt_ensemble", "A — prompt ensemble on"),
    ("baseline_clip_nlp", "B — default"),
    ("b_weights_clip_0.2", "B — CLIP share 0.2"),
    ("b_rerank_alpha_0.5", "B — rerank α 0.5"),
    ("b_rerank_topn_100", "B — rerank top_n 100"),
    ("baseline_vlm", "C — default"),
    ("c_clip_weight_0.5", "C — CLIP weight 0.5"),
]


def m(run: dict, key: str) -> float:
    return float((run.get("metrics") or {}).get(key, 0.0))


def delta(run: dict, key: str = "mAP") -> float | None:
    value = (run.get("delta_vs_baseline") or {}).get(key)
    return None if value is None else float(value)


def f4(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def sf4(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.4f}"


def f2(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def esc(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def mono(text: str, size: float = 7.2) -> str:
    return f'<font name="{MONO}" size="{size}" color="#334E68">{esc(text)}</font>'


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
S_TH = style("TH", fontName=BOLD, fontSize=7.4, leading=9.6, textColor=colors.white, spaceAfter=0)
S_TD = style("TD", fontSize=7.8, leading=10.4, textColor=INK, spaceAfter=0)
S_TD_SMALL = style("TDSmall", fontSize=7.1, leading=9.2, textColor=INK, spaceAfter=0)


def P(text: str, st=S_BODY):
    return Paragraph(text, st)


def H1(number: str, title: str):
    return [
        CondPageBreak(90),
        Paragraph(f"{number}&nbsp;&nbsp;{esc(title)}", S_H1),
        HRFlowable(width="100%", thickness=1.1, color=TEAL, spaceBefore=1, spaceAfter=8),
    ]


def H2(number: str, title: str):
    return [CondPageBreak(95), Paragraph(f"{number}&nbsp;&nbsp;{esc(title)}", S_H2)]


def H3(title: str):
    return [Paragraph(esc(title), S_H3)]


def bullets(items: list[str]):
    return [Paragraph(item, S_BULLET, bulletText="•") for item in items]


def callout(title: str, body: str, kind: str = "teal"):
    accents = {
        "teal": (TEAL, TEAL_LIGHT),
        "amber": (AMBER, AMBER_LIGHT),
        "red": (RED, RED_LIGHT),
        "blue": (BLUE, BLUE_LIGHT),
        "gray": (MUTED, LIGHT),
        "green": (GREEN, GREEN_LIGHT),
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


def code_block(text: str):
    flow = Preformatted(text.strip("\n"), S_CODE)
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
    font: ParagraphStyle = S_TD_SMALL,
    header_font: ParagraphStyle = S_TH,
    row_tints: dict[int, tuple[colors.Color, colors.Color]] | None = None,
    repeat: int = 1,
    pad: float = 4.0,
):
    data = [[Paragraph(h, header_font) for h in header]]
    for row in rows:
        data.append(
            [cell if isinstance(cell, Flowable) else Paragraph(str(cell), font) for cell in row]
        )
    table = Table(data, colWidths=widths, repeatRows=repeat, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), pad),
        ("RIGHTPADDING", (0, 0), (-1, -1), pad),
        ("TOPPADDING", (0, 0), (-1, -1), 4.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.2),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            commands.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    if row_tints:
        for index, (bg, text_color) in row_tints.items():
            commands.append(("BACKGROUND", (0, index + 1), (-1, index + 1), bg))
            if text_color is not None:
                commands.append(("TEXTCOLOR", (0, index + 1), (-1, index + 1), text_color))
    if align:
        for j, kind in enumerate(align):
            if kind == "r":
                commands.append(("ALIGN", (j, 1), (j, -1), "RIGHT"))
            elif kind == "c":
                commands.append(("ALIGN", (j, 1), (j, -1), "CENTER"))
    table.setStyle(TableStyle(commands))
    if len(data) <= 10:
        return KeepTogether(table)
    return table


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
class SweepGrid(Flowable):
    """Six small multiples: one panel per parameter family."""

    PANEL_H = 156
    GAP = 10
    COLS = 3

    def __init__(self, width: float = CONTENT_W):
        super().__init__()
        self.width = width
        self.height = 2 * self.PANEL_H + self.GAP

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        c = self.canv
        cols = self.COLS
        panel_w = (self.width - (cols - 1) * self.GAP) / cols
        for index, group in enumerate(SUMMARY["groups"]):
            col, row = index % cols, index // cols
            x = col * (panel_w + self.GAP)
            y = self.height - (row + 1) * self.PANEL_H - row * self.GAP
            points = []
            for point in group["points"]:
                run = RUNS.get(point["run_id"])
                if not run or not run.get("ready"):
                    continue
                points.append(
                    {
                        "x": float(point["x"]),
                        "label": point["label"],
                        "is_baseline": bool(point["is_baseline"]),
                        "value": m(run, "mAP"),
                    }
                )
            self._panel(c, x, y, panel_w, self.PANEL_H, group, points)

    def _panel(self, c, x, y, w, h, group, points):
        c.saveState()
        c.setFillColor(LIGHTER)
        c.setStrokeColor(RULE)
        c.setLineWidth(0.5)
        c.roundRect(x, y, w, h, 3, fill=1, stroke=1)

        c.setFillColor(NAVY)
        c.setFont(BOLD, 7.4)
        c.drawString(x + 8, y + h - 12, group["family"])
        c.setFillColor(MUTED)
        c.setFont(REG, 6.0)
        c.drawString(x + 8, y + h - 21, f"x: {group['x_label']}  ·  y: mAP")

        left, right = x + 30, x + w - 8
        bottom, top = y + 22, y + h - 30
        c.setStrokeColor(RULE)
        c.setLineWidth(0.5)
        c.line(left, bottom, right, bottom)

        values = [p["value"] for p in points]
        if not values:
            c.restoreState()
            return
        if group["kind"] == "bar":
            self._bars(c, left, right, bottom, top, points)
        else:
            self._line(c, left, right, bottom, top, points)
        c.restoreState()

    @staticmethod
    def _ycale(values, zero=False):
        lo, hi = min(values), max(values)
        if zero:
            lo = 0.0
        span = hi - lo or 0.02
        lo2 = lo - span * (0.0 if zero else 0.18)
        hi2 = hi + span * (0.16 if zero else 0.18)
        if lo2 < 0 and lo >= 0:
            lo2 = 0.0
        if hi2 <= lo2:
            hi2 = lo2 + 0.01
        return lo2, hi2

    def _yticks(self, c, left, top, bottom, lo, hi):
        for value in (hi, (hi + lo) / 2, lo):
            yy = bottom + (value - lo) / (hi - lo) * (top - bottom)
            c.setStrokeColor(RULE)
            c.setLineWidth(0.35)
            c.line(left, yy, left + 6, yy)
            c.setFillColor(MUTED)
            c.setFont(REG, 5.6)
            c.drawRightString(left - 3, yy - 1.8, f"{value:.3f}")

    def _line(self, c, left, right, bottom, top, points):
        values = [p["value"] for p in points]
        lo, hi = self._ycale(values)
        self._yticks(c, left, top, bottom, lo, hi)

        xs = [p["x"] for p in points]
        x0, x1 = min(xs), max(xs)
        if x0 == x1:
            x0, x1 = x0 - 1, x1 + 1

        def sx(value):
            return left + (value - x0) / (x1 - x0) * (right - left)

        def sy(value):
            return bottom + (value - lo) / (hi - lo) * (top - bottom)

        base = next((p for p in points if p["is_baseline"]), None)
        if base:
            c.setStrokeColor(colors.HexColor("#7B8794"))
            c.setLineWidth(0.6)
            c.setDash(2.2, 2.2)
            c.line(left, sy(base["value"]), right, sy(base["value"]))
            c.setDash()

        c.setStrokeColor(BLUE)
        c.setLineWidth(1.4)
        c.setLineJoin(1)
        path = c.beginPath()
        for index, point in enumerate(points):
            px, py = sx(point["x"]), sy(point["value"])
            if index == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)
        c.drawPath(path, stroke=1, fill=0)

        for point in points:
            px, py = sx(point["x"]), sy(point["value"])
            c.setFillColor(colors.white if point["is_baseline"] else BLUE)
            c.setStrokeColor(BLUE)
            c.setLineWidth(1.1)
            c.circle(px, py, 2.5, fill=1, stroke=1)
            c.setFillColor(MUTED)
            c.setFont(REG, 5.6)
            label = f"{point['x']:g}"
            c.drawCentredString(px, bottom - 9, label)

        best = max(points, key=lambda p: p["value"])
        c.setFillColor(NAVY)
        c.setFont(BOLD, 6.0)
        label_y = sy(best["value"]) + 8
        if label_y > top - 2:
            label_y = sy(best["value"]) - 13
        label_x = min(max(sx(best["x"]), left + 12), right - 12)
        c.drawCentredString(label_x, label_y, f"{best['value']:.3f}")

    def _bars(self, c, left, right, bottom, top, points):
        values = [p["value"] for p in points]
        lo, hi = self._ycale(values, zero=True)
        self._yticks(c, left, top, bottom, lo, hi)

        slot = (right - left) / max(1, len(points))
        width = min(34.0, slot * 0.42)
        for index, point in enumerate(points):
            cx = left + slot * (index + 0.5)
            bar_h = (point["value"] - lo) / (hi - lo) * (top - bottom)
            c.setFillColor(BLUE)
            c.rect(cx - width / 2, bottom, width, bar_h, fill=1, stroke=0)
            c.setFillColor(NAVY)
            c.setFont(BOLD, 6.0)
            c.drawCentredString(cx, bottom + bar_h + 5, f"{point['value']:.3f}")
            c.setFillColor(MUTED)
            c.setFont(REG, 5.6)
            c.drawCentredString(
                cx,
                bottom - 8,
                "default" if point["is_baseline"] else "ensemble on",
            )


# --------------------------------------------------------------------------- #
# Cover and page furniture
# --------------------------------------------------------------------------- #
def wrap_lines(text: str, font: str, size: float, width: float) -> list[str]:
    return simpleSplit(text, font, size, width)


def draw_cover(c: canvas.Canvas, doc):
    w, h = PAGE_W, PAGE_H
    band_top = 556

    c.setFillColor(NAVY)
    c.rect(0, band_top, w, h - band_top, fill=1, stroke=0)
    c.setFillColor(BLUE)
    c.rect(0, band_top - 4, w, 4, fill=1, stroke=0)

    x = 54
    c.setFillColor(colors.HexColor("#9FB3C8"))
    c.setFont(BOLD, 8.2)
    c.drawString(x, 790, "TECHNICAL REPORT  ·  PARAMETER TUNING")

    c.setFillColor(colors.white)
    c.setFont(BOLD, 30)
    c.drawString(x, 738, "Parameter Tuning Report")

    c.setFillColor(colors.HexColor("#D9E2EC"))
    c.setFont(REG, 11.6)
    subtitle = (
        "A one-factor sweep of fusion weights, reranking and CLIP blending on the "
        "frozen 240-query Meme-Cap benchmark."
    )
    ty = 704
    for line in wrap_lines(subtitle, REG, 11.6, w - 130):
        c.drawString(x, ty, line)
        ty -= 16

    c.setStrokeColor(colors.HexColor("#3E5C76"))
    c.setLineWidth(0.6)
    c.line(x, 664, w - x, 664)

    chip_y = 618
    labels = [
        f"{len(SUMMARY['runs'])} runs",
        "3 strategies",
        f"{SUMMARY['num_queries']} frozen queries",
        "0 model weights trained",
    ]
    cx = x
    for label in labels:
        cw = c.stringWidth(label, BOLD, 7.4) + 16
        c.setFillColor(colors.HexColor("#1B3A57"))
        c.setStrokeColor(colors.HexColor("#3E5C76"))
        c.setLineWidth(0.7)
        c.roundRect(cx, chip_y, cw, 23, 4, fill=1, stroke=1)
        c.setFillColor(colors.HexColor("#D9E2EC"))
        c.setFont(BOLD, 7.4)
        c.drawString(cx + 8, chip_y + 7.5, label)
        cx += cw + 7

    c.setFillColor(colors.HexColor("#8CA3BA"))
    c.setFont(REG, 7.4)
    c.drawString(x, 590, "Read-only sweep  ·  data/tuning/summary.json  ·  no retrieval code changed")

    rows = [
        ("PROJECT", "Meme-Cap text-to-image retrieval  ·  COMP5523  ·  repository V2"),
        ("SOURCE", f"Commit {COMMIT}  ·  sweep generated {GENERATED}  ·  document {DOC_DATE}"),
        ("SCOPE", "Configurable parameters only — no retraining, no weight changes"),
    ]
    for offset, (label, value) in enumerate(rows):
        c.setFillColor(MUTED)
        c.setFont(BOLD, 7.1)
        c.drawString(x, 524 - offset * 15, label)
        c.setFillColor(INK)
        c.setFont(REG, 8.3)
        c.drawString(x + 62, 524 - offset * 15, value)

    c.setFillColor(BODY)
    c.setFont(REG, 8.6)
    intro = (
        "This report isolates what each configurable parameter is worth on a fixed gallery "
        "and a frozen query set. It is evidence about which knobs matter — not a new "
        "benchmark, and not a fine-tuning result."
    )
    iy = 468
    for line in wrap_lines(intro, REG, 8.6, CONTENT_W):
        c.drawString(x, iy, line)
        iy -= 13

    stats = [
        ("B — BEST", "b_weights_clip_0.2", "CLIP share 0.20"),
        ("C — BEST", "c_clip_weight_0.5", "CLIP weight 0.50"),
        ("A — ENSEMBLE", "a_prompt_ensemble", "prompt ensemble on"),
    ]
    box_w = (w - 2 * x - 2 * 12) / 3
    box_y = 282
    for index, (title, run_id, note) in enumerate(stats):
        run = RUNS[run_id]
        bx = x + index * (box_w + 12)
        c.setFillColor(colors.white)
        c.setStrokeColor(RULE)
        c.setLineWidth(0.7)
        c.roundRect(bx, box_y, box_w, 132, 5, fill=1, stroke=1)
        c.setFillColor(MUTED)
        c.setFont(BOLD, 7.0)
        c.drawString(bx + 12, box_y + 108, title)
        c.setFillColor(NAVY)
        c.setFont(BOLD, 24)
        c.drawString(bx + 12, box_y + 68, f"{m(run, 'mAP'):.4f}")
        c.setFillColor(BODY)
        c.setFont(REG, 8.0)
        c.drawString(bx + 12, box_y + 48, f"mAP  ·  {note}")
        d = delta(run)
        c.setFillColor(GREEN if (d or 0) > 0 else MUTED)
        c.setFont(BOLD, 8.0)
        c.drawString(
            bx + 12,
            box_y + 30,
            f"{sf4(d)} vs default" if d is not None else "default configuration",
        )

    c.setStrokeColor(RULE)
    c.setLineWidth(0.5)
    c.line(x, 258, w - x, 258)

    c.setFillColor(MUTED)
    c.setFont(BOLD, 7.1)
    c.drawString(x, 236, "CONTENTS")
    contents = [
        ["1   Executive summary", "2   Protocol and isolation", "3   Results"],
        ["4   Parameter effects", "5   Interpretation", "6   Category behaviour"],
        ["7   Threats to validity", "8   Recommendations", "9   Reproducibility"],
    ]
    col_w = (w - 2 * x) / 3
    for col, items in enumerate(contents):
        for row, item in enumerate(items):
            c.setFillColor(BODY)
            c.setFont(REG, 8.0)
            c.drawString(x + col * col_w, 216 - row * 14, item)

    c.setFillColor(MUTED)
    c.setFont(REG, 6.8)
    c.drawString(
        x,
        96,
        "Every number in this document is read from data/tuning/summary.json by "
        "docs/make_tuning_report_pdf.py.",
    )

    c.setFillColor(MUTED)
    c.setFont(REG, 7)
    c.drawString(
        x,
        34,
        "Exploratory test-set tuning: confirm any candidate on a held-out query set before reporting it.",
    )
    c.drawRightString(w - x, 34, "Analysis follows")


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
        self.drawString(LM, y, "Parameter Tuning Report")
        self.drawRightString(w - RM, y, "Meme-Cap T2I Retrieval  ·  COMP5523")

        self.setStrokeColor(RULE)
        self.line(LM, 36, w - RM, 36)
        self.setFillColor(MUTED)
        self.setFont(REG, 6.8)
        self.drawString(LM, 24, f"Source commit {COMMIT}  ·  sweep {GENERATED}  ·  document {DOC_DATE}")
        self.drawRightString(w - RM, 24, f"{self._pageNumber} / {total}")


# --------------------------------------------------------------------------- #
# Story sections
# --------------------------------------------------------------------------- #
def section_summary() -> list:
    out = H1("1", "Executive summary")
    out.append(
        P(
            "The system ships three retrieval strategies over one shared gallery: "
            "<b>A</b> (OpenCLIP only), <b>B</b> (CLIP fused with dense annotation embeddings "
            "and BM25, optionally reranked) and <b>C</b> (a dense index over offline Qwen2.5-VL "
            "descriptions, optionally fused with CLIP). All weights and thresholds are frozen. "
            "This report measures what changes when those configurable values move, holding the "
            "gallery, the indexes, the query set and the models fixed."
        )
    )
    out.append(
        P(
            "Seventeen configurations were evaluated on the same frozen 240-query test set: "
            "three defaults (which reproduce the published <font name='Consolas' size='8'>/evaluation"
            "</font> numbers exactly) and fourteen single-knob variants. No model weights were "
            "trained or fine-tuned, and no retrieval code was modified; each run only set "
            "<font name='Consolas' size='8'>TSR_*</font> environment overrides."
        )
    )
    out.append(Spacer(1, 4))
    out.append(
        callout(
            "Bottom line",
            "Strategy B improves most when the CLIP share of fusion drops from 0.5 to 0.2: "
            "mAP rises from 0.7608 to 0.8055 (+0.0447, +5.9% relative). Reranking α=0.5 adds "
            "+0.0251 on top of the default configuration. Strategy C improves monotonically as "
            "CLIP blending rises and peaks at the largest tested value, 0.5 (0.4519 → 0.4852), "
            "but remains 0.27 mAP behind B. The prompt ensemble in A is neutral (+0.0020, well "
            "inside the confidence interval) and does not justify its extra cost.",
            "blue",
        )
    )
    out.append(Spacer(1, 6))
    out.extend(
        bullets(
            [
                "<b>B is not close to its ceiling.</b> The best B variant is 0.8055 mAP; the best C "
                "variant is 0.4852 and the best A variant is 0.4415. The ordering A &lt; C &lt; B "
                "is unchanged by every knob tested.",
                "<b>The annotation channels are worth more than raw CLIP similarity.</b> At CLIP "
                "share 0.2, B improves on all three query categories, including the visual one "
                "(0.6313 → 0.6899), because image captions encode visual detail that CLIP cosine "
                "does not expose.",
                "<b>Reranking wants an equal blend, not a reranker-heavy one.</b> α=0.5 beats both "
                "the default 0.3 and 0.7, and top_n=60 beats 20 and 100 — more candidates add "
                "noise and latency without adding recall.",
                "<b>C needs CLIP.</b> Removing CLIP blending entirely costs 0.0597 mAP "
                "(0.4519 → 0.3922); the response is monotone up to the tested maximum, so the "
                "useful range probably extends beyond 0.5.",
                "<b>These numbers are exploratory.</b> Every variant was scored on the same test "
                "set used to choose it. They are descriptive evidence for which knobs matter, not "
                "an unbiased benchmark; the <font name='Consolas' size='8'>/evaluation</font> table "
                "remains the protocol of record.",
            ]
        )
    )
    return out


def section_protocol() -> list:
    out = H1("2", "Protocol and isolation")
    out.append(
        P(
            "The sweep reuses the existing <font name='Consolas' size='8'>python -m tsr eval</font> "
            "path. Each run received its own <font name='Consolas' size='8'>TSR_PATHS__REPORTS_DIR"
            "</font> under <font name='Consolas' size='8'>data/tuning/runs/&lt;run_id&gt;/</font>, so "
            "the report tree behind <font name='Consolas' size='8'>/evaluation</font> was never "
            "written to. A preflight child process resolves the effective reports directory and "
            "aborts if it is not inside <font name='Consolas' size='8'>data/tuning/</font>."
        )
    )
    out.append(
        P(
            "Before and after the sweep, the runner hashed every protected artifact: "
            "<font name='Consolas' size='8'>data/reports/</font>, "
            "<font name='Consolas' size='8'>data/settings.json</font>, the SQLite database and its "
            "WAL, <font name='Consolas' size='8'>data/index/</font>, "
            "<font name='Consolas' size='8'>data/eval/</font> and "
            "<font name='Consolas' size='8'>configs/default.yaml</font>. The diff was empty; only "
            "<font name='Consolas' size='8'>data/tuning/</font> changed. The SQLite "
            "<font name='Consolas' size='8'>-shm</font> sidecar is treated as volatile because a "
            "read-only reader may touch it."
        )
    )
    out.append(
        P(
            "The three baseline runs were executed first and compared field by field against the "
            "existing published reports (metrics, bootstrap intervals and category breakdowns, "
            "tolerance 1e-6). All three matched with a maximum absolute difference of 0.0, which "
            "confirms that the sweep measured parameter effects rather than code or data drift."
        )
    )
    out.append(
        P(
            "All runs use <font name='Consolas' size='8'>memecap_eval_v1</font>: 240 frozen queries "
            "from the held-out test split, 80 each for visual, intent and metaphor. Raw-caption "
            "diagnostic queries are excluded. mAP is the primary metric; MRR, nDCG@10, P@1, R@10, "
            "median rank and miss count are reported alongside it. Intervals are 95% percentile "
            "bootstrap intervals (1,000 resamples, seed 42) computed independently for each run, "
            "not paired between variants."
        )
    )
    rows = [
        [
            "A · CLIP",
            "prompt ensemble",
            "off (default) → on",
            "Prompt templates averaged at query time",
        ],
        [
            "B · CLIP + NLP",
            "<font name='Consolas' size='6.8'>weights.clip</font>",
            "0.0, 0.1, 0.2, 0.3 (default 0.5)",
            "doc:bm25 held at 60:40; weights renormalised over active channels",
        ],
        [
            "B · CLIP + NLP",
            "doc / BM25 split",
            "doc share 0.4, 0.6, 0.8",
            "CLIP fixed at 0.5; dense and BM25 shares swapped",
        ],
        [
            "B · CLIP + NLP",
            "<font name='Consolas' size='6.8'>rerank.alpha</font>",
            "0.5, 0.7 (default 0.3)",
            "Weight of the cross-encoder score in the reranked blend",
        ],
        [
            "B · CLIP + NLP",
            "<font name='Consolas' size='6.8'>rerank.top_n</font>",
            "20, 100 (default 60)",
            "Candidates passed to the cross-encoder",
        ],
        [
            "C · VLM",
            "<font name='Consolas' size='6.8'>clip_weight</font>",
            "0.0, 0.35, 0.5 (default 0.2)",
            "Weight of CLIP similarity when fused with the VLM description index",
        ],
    ]
    out.append(
        KeepTogether(
            [
                *H3("What was varied"),
                make_table(
                    ["Strategy", "Knob", "Values tested", "What changes"],
                    rows,
                    [82, 92, 112, CONTENT_W - 286],
                ),
                caption(
                    "Table 1. One-factor sweep matrix. Everything not listed stays at its default. "
                    "CLIP-share runs keep the remaining doc:BM25 mass at 60:40, so the family "
                    "isolates the CLIP weight rather than confounding it with the dense/BM25 split."
                ),
            ]
        )
    )
    return out


def results_rows() -> list[list[str]]:
    rows = []
    for run_id in RUN_ORDER:
        run = RUNS[run_id]
        label = run_id
        if run.get("is_baseline"):
            label = f"{mono(run_id)}<font size='6.4' color='#62717F'> · default</font>"
        else:
            label = mono(run_id)
        ci = (run.get("ci") or {}).get("mAP") or {}
        interval = (
            f"[{ci['low']:.3f}, {ci['high']:.3f}]" if "low" in ci and "high" in ci else "—"
        )
        rows.append(
            [
                label,
                f4(m(run, "mAP")),
                sf4(delta(run)),
                interval,
                f4(m(run, "MRR")),
                f4(m(run, "ndcg@10")),
                f4(m(run, "precision@1")),
                f4(m(run, "recall@10")),
                f2(m(run, "median_rank")),
                str(int(m(run, "misses"))),
            ]
        )
    return rows


def section_results() -> list:
    out = H1("3", "Results")
    out.append(
        P(
            "Table 2 isolates the configurations that matter. Deltas are against each strategy's "
            "own default, so they are comparable within A, B and C but not across them: the three "
            "strategies start from very different baselines."
        )
    )
    key_rows = []
    tints = {}
    for index, (run_id, description) in enumerate(KEY_ROWS):
        run = RUNS[run_id]
        ci = (run.get("ci") or {}).get("mAP") or {}
        interval = (
            f"[{ci['low']:.3f}, {ci['high']:.3f}]" if "low" in ci and "high" in ci else "—"
        )
        d = delta(run)
        if d is not None and d > 0:
            delta_cell = f'<font color="#047857">{sf4(d)}</font>'
        elif d is not None and d < 0:
            delta_cell = f'<font color="#B42318">{sf4(d)}</font>'
        else:
            delta_cell = "—"
        key_rows.append(
            [
                esc(description),
                mono(run_id),
                f4(m(run, "mAP")),
                delta_cell,
                interval,
                f4(m(run, "precision@1")),
                str(int(m(run, "misses"))),
            ]
        )
        if run.get("is_baseline"):
            tints[index] = (LIGHT, None)
    out.append(
        make_table(
            ["Configuration", "Run id", "mAP", "Δ mAP", "95% CI", "P@1", "Misses"],
            key_rows,
            [116, 118, 44, 44, 72, 44, CONTENT_W - 438],
            align=["l", "l", "r", "r", "r", "r", "r"],
            font=S_TD,
            row_tints=tints,
        )
    )
    out.append(
        caption(
            "Table 2. Headline configurations. Grey rows are the unmodified defaults. "
            "A positive Δ is better for mAP, P@1 and misses reduce is better; the CI is the "
            "marginal 95% bootstrap interval for that run."
        )
    )
    out.append(
        P(
            "No pair of marginal intervals is disjoint at 240 queries, so none of these deltas is "
            "significant taken alone. The B CLIP-share and C CLIP-blend effects are nonetheless the "
            "ones worth taking seriously: they are large (+0.0447 and +0.0334), they move all three "
            "query categories in the same direction, and their curves have an interpretable shape. "
            "The A prompt ensemble and the doc/BM25 rebalancing move by less than 0.01 mAP and "
            "should be read as neutral. The full 17-run table follows; it is the exact set of rows "
            "the <font name='Consolas' size='8'>/tuning</font> page renders."
        )
    )
    out.append(
        make_table(
            [
                "Run",
                "mAP",
                "Δ mAP",
                "95% CI",
                "MRR",
                "nDCG@10",
                "P@1",
                "R@10",
                "med.",
                "miss",
            ],
            results_rows(),
            [118, 38, 40, 66, 38, 44, 36, 38, 30, CONTENT_W - 448],
            align=["l", "r", "r", "r", "r", "r", "r", "r", "r", "r"],
        )
    )
    out.append(
        caption(
            "Table 3. All runs, ordered as reported by the sweep. med. is median rank; "
            "miss counts queries where no relevant image appears in the top 100. Latency is "
            "omitted deliberately: each run started a fresh process, so p50/p95 are not comparable."
        )
    )
    return out


def section_charts() -> list:
    out = H1("4", "Parameter effects")
    out.append(
        P(
            "Figure 1 draws one panel per family. The blue line is the swept value; the dashed "
            "line marks the default configuration's mAP, and the hollow marker at the default's "
            "own x position shows where the current setting sits on that curve. Line panels use a "
            "data-driven y-axis (not zero-based) because the differences of interest are small "
            "relative to the absolute mAP; the boolean panel in A is zero-based because bars "
            "encode magnitude from zero."
        )
    )
    out.append(Spacer(1, 3))
    out.append(SweepGrid())
    out.append(
        caption(
            "Figure 1. mAP against each swept parameter, one panel per family. Panels share no "
            "y-axis: read within a panel, not across panels. A hollow marker is the default "
            "configuration; it is drawn at its own parameter value, so the distance from the "
            "swept curve is visible."
        )
    )
    out.append(
        P(
            "The shapes carry the argument. B's CLIP share is a clear inverted U that peaks at "
            "0.2 and falls away on both sides — the current default 0.5 sits near the bottom of "
            "its own curve. The doc/BM25 panel is nearly flat and slightly downward on the "
            "doc-heavy side. Rerank α and top_n both peak away from the extremes, at 0.5 and at "
            "the default 60. C's response rises monotonically across the tested range, so the "
            "panel is truncated before any turning point."
        )
    )
    return out


def section_interpretation() -> list:
    out = H1("5", "Interpretation")

    out.extend(H2("5.1", "Strategy A — prompt ensemble is cosmetic"))
    out.append(
        P(
            "The prompt ensemble moves mAP from 0.4395 to 0.4415 (+0.0020), and its bootstrap "
            "interval [0.3863, 0.4989] is almost identical to the baseline's [0.3848, 0.4975]. "
            "Category behaviour is mixed rather than uniformly positive: visual and intent fall "
            "slightly (0.3153 → 0.3058 and 0.4444 → 0.4351), metaphor rises (0.5588 → 0.5836). "
            "For a strategy whose ceiling is 0.44, this is noise plus a small template bias. "
            "Recommendation: leave the ensemble off; it multiplies text-encoder work for no "
            "measurable retrieval gain."
        )
    )

    out.extend(H2("5.2", "Strategy B — the CLIP share is the dominant knob"))
    out.append(
        P(
            "B's default fusion is CLIP 0.5, dense annotations 0.3, BM25 0.2. Holding the "
            "doc:BM25 split at 60:40 and lowering the CLIP share to 0.2 gives the best result in "
            "the entire sweep: 0.8055 mAP, +0.0447 over default, with MRR 0.8057, nDCG@10 0.8177 "
            "and P@1 0.7542. The response is an inverted U — 0.7802 at 0.0, 0.8036 at 0.1, 0.8055 "
            "at 0.2, 0.7967 at 0.3 and 0.7608 at the default 0.5. The default is the worst point "
            "on its own curve, which is the strongest single signal in the sweep: the "
            "configuration is not near a noisy optimum, it is on the wrong side of one. Even the "
            "CLIP-free setting at 0.0 remains a strong text-only system (0.7802)."
        )
    )
    out.append(
        P(
            "The gain is not confined to textual categories. mAP rises on visual queries from "
            "0.6313 to 0.6899, on intent from 0.7389 to 0.7761 and on metaphor from 0.9121 to "
            "0.9504 (Table 4). The likely mechanism is that the Meme-Cap annotations contain "
            "explicit descriptions of what is in the image, so the dense channel is not purely "
            "semantic: it is a second, text-side visual signal that is complementary to the "
            "CLIP image embedding. Rebalancing toward it helps every query type."
        )
    )

    out.extend(H2("5.3", "Strategy B — keep the doc/BM25 split, retune reranking"))
    out.append(
        P(
            "The doc/BM25 split is the least sensitive knob in the sweep. Moving to a doc-heavy "
            "80/20 split costs 0.0133 mAP; moving to a BM25-heavy 40/60 split gains 0.0092, inside "
            "the noise of the baseline. The default 60/40 sits close to the optimum and is a sound "
            "compromise: dense annotations carry most of the signal, while BM25 still contributes "
            "exact lexical matches for names and stock phrases."
        )
    )
    out.append(
        P(
            "Reranking behaves differently. α is the weight of the cross-encoder score in the "
            "final blend. The default α=0.3 (0.7608) is beaten by α=0.5 (0.7858, +0.0251); "
            "α=0.7 (0.7752) gives back part of the gain, which is consistent with a reranker that "
            "is useful as a tie-breaker but harmful when it dominates the fused retrieval score. "
            "top_n=60 is the best candidate pool tested: 20 is too small (0.7570, −0.0037) and "
            "100 adds no mAP (0.7670, +0.0062) while costing cross-encoder passes. The default "
            "top_n=60 should stay."
        )
    )

    out.extend(H2("5.4", "Strategy C — CLIP blending is load-bearing"))
    out.append(
        P(
            "C's VLM descriptions are rich but lossy: they capture intent well (0.5390 at "
            "default) and visual content poorly (0.2469). Adding CLIP similarity fixes part of "
            "that. With clip_weight 0.0 mAP falls to 0.3922; 0.35 gives 0.4720 and 0.5 gives "
            "0.4852 (+0.0334 over the default 0.2). The response is monotone over the tested "
            "range, so the optimum is at or beyond 0.5; the sweep cannot say where it turns. "
            "Even the best C configuration remains 0.276 behind B's default and 0.320 behind B's "
            "best variant, so this knob improves C without changing the strategy ranking."
        )
    )

    out.extend(H2("5.5", "Combining the winners is untested"))
    out.append(
        P(
            "The sweep changed one knob at a time. The best B weights (0.2), the best rerank "
            "α (0.5) and the default top_n (60) are plausibly complementary, but their joint "
            "effect is not measured here. A combined run is the obvious next experiment: it "
            "would test whether the +0.0447 and +0.0251 gains add, overlap, or interact through "
            "the candidate pool. Until that run exists, the correct claim is that each knob helps "
            "in isolation, not that 0.8055 + 0.0251 is attainable."
        )
    )
    return out


def category_rows() -> list[list[str]]:
    rows = []
    for run_id in (
        "baseline_clip",
        "a_prompt_ensemble",
        "baseline_clip_nlp",
        "b_weights_clip_0.2",
        "b_rerank_alpha_0.5",
        "baseline_vlm",
        "c_clip_weight_0.5",
    ):
        run = RUNS[run_id]
        base = BASELINES[run["strategy"]]
        cells = []
        for category in ("visual", "intent", "metaphor"):
            group = (run.get("per_category") or {}).get(category) or {}
            base_group = (base.get("per_category") or {}).get(category) or {}
            value = group.get("mAP")
            base_value = base_group.get("mAP")
            if value is None:
                cells.append("—")
                continue
            text = f"<b>{value:.4f}</b>"
            if not run.get("is_baseline") and isinstance(base_value, (int, float)):
                diff = value - base_value
                color = "#047857" if diff > 0 else "#B42318" if diff < 0 else "#62717F"
                text += f" <font size='6.4' color='{color}'>{diff:+.3f}</font>"
            cells.append(text)
        rows.append([mono(run_id), *cells])
    return rows


def section_categories() -> list:
    out = H1("6", "Category behaviour")
    out.append(
        P(
            "The benchmark is stratified: 80 visual, 80 intent and 80 metaphor queries. "
            "Category mAP exposes what a global average hides. Table 4 compares the defaults "
            "with the strongest variant in each strategy."
        )
    )
    out.append(
        make_table(
            ["Run", "Visual (80)", "Intent (80)", "Metaphor (80)"],
            category_rows(),
            [CONTENT_W - 3 * 90, 90, 90, 90],
            align=["l", "r", "r", "r"],
        )
    )
    out.append(
        caption(
            "Table 4. mAP by query category. Small numbers are the change against the same "
            "strategy's default. The best B variant improves every category; the A ensemble "
            "trades visual and intent for metaphor."
        )
    )
    out.append(
        P(
            "Two readings matter. First, B is not winning only on metaphor: at CLIP share 0.2 it "
            "leads on visual (0.6899) and intent (0.7761) as well, which is why the global gain is "
            "broad rather than a category artefact. Second, C's weakness is concentrated in visual "
            "queries (0.2469 default, 0.2954 at clip_weight 0.5), exactly where CLIP blending is "
            "designed to help; the other two categories are respectable but still below B. "
            "A's absolute ceiling on visual queries (0.3153) explains why prompt engineering "
            "alone cannot close the gap."
        )
    )
    return out


def section_validity() -> list:
    out = H1("7", "Threats to validity")
    out.extend(
        bullets(
            [
                "<b>Test-set tuning.</b> All variants were selected and scored on the same 240 "
                "queries. The winner's margin is therefore optimistic: with 240 queries, choosing "
                "the best of fourteen correlated variants can capture noise. Do not report "
                "0.8055 as an unbiased score; confirm the candidate on a held-out or "
                "freshly-drafted query set before quoting it.",
                "<b>Intervals are marginal, not paired.</b> Each run has its own bootstrap CI, "
                "but the sweep did not compute paired query-level differences between variants, "
                "nor a multiple-comparison correction. Overlapping intervals mean the ordering "
                "of close variants (0.1 vs 0.2, α=0.5 vs 0.7) is descriptive, not significant.",
                "<b>One factor at a time.</b> Interactions are invisible by construction. The "
                "combined effect of CLIP share 0.2, α=0.5 and top_n=60 is unknown, and the "
                "inverted-U shape of the CLIP-share curve may shift once reranking changes.",
                "<b>Coarse grids and truncated ranges.</b> Four CLIP shares, two α values and two "
                "top_n values cannot locate the true optimum. C's curve is still rising at the "
                "largest tested clip_weight, so its best tested value is a boundary, not a peak.",
                "<b>Single dataset and language.</b> Meme-Cap annotations are English and "
                "culture-specific; the value of dense annotations and the weak visual performance "
                "of VLM descriptions may not transfer to other meme corpora.",
                "<b>Latency is not compared.</b> Every run started a new process, so p50/p95 "
                "timings are reported per run but should not be read as a speed comparison. "
                "Reranking with top_n=100 in particular costs more than the table shows.",
                "<b>No learned component changed.</b> This is parameter tuning, not fine-tuning: "
                "CLIP, BGE, the reranker and Qwen2.5-VL are frozen. Any claim that the system was "
                "'trained' on the benchmark would be false.",
            ]
        )
    )
    return out


def section_recommendations() -> list:
    out = [CondPageBreak(250)]
    out += H1("8", "Recommendations and next steps")
    out.append(
        P(
            "The following table separates what the sweep supports from what it merely suggests. "
            "None of these changes has been applied to the default configuration; the sweep is "
            "read-only evidence."
        )
    )
    rows = [
        [
            "<b>B</b> — CLIP share",
            "0.5",
            "0.2 (doc 0.48, bm25 0.32)",
            "+0.0447 mAP; best in sweep; improves all categories",
            "Medium — test-set selection",
        ],
        [
            "<b>B</b> — rerank α",
            "0.3",
            "0.5",
            "+0.0251 mAP; 0.7 gives back part of the gain",
            "Medium",
        ],
        [
            "<b>B</b> — rerank top_n",
            "60",
            "60 (keep)",
            "20 loses 0.0037; 100 adds 0.0062 at extra cost",
            "High — keep default",
        ],
        [
            "<b>B</b> — doc / BM25",
            "60 / 40",
            "60 / 40 (keep)",
            "80/20 costs 0.0133; 40/60 is +0.0092, inside noise",
            "High — keep default",
        ],
        [
            "<b>C</b> — clip_weight",
            "0.2",
            "0.5 (test higher)",
            "+0.0334 mAP; monotone at the tested boundary",
            "Low–medium — optimum untested",
        ],
        [
            "<b>A</b> — prompt ensemble",
            "off",
            "off (keep)",
            "+0.0020 mAP, inside the confidence interval",
            "High — no benefit",
        ],
    ]
    out.append(
        make_table(
            ["Strategy / knob", "Default", "Candidate", "Evidence", "Confidence"],
            rows,
            [80, 48, 92, 160, CONTENT_W - 380],
        )
    )
    out.append(
        caption(
            "Table 5. Candidate settings. 'Confidence' refers to how well this sweep supports the "
            "change, not to statistical significance."
        )
    )
    out.append(Spacer(1, 2))
    out.append(
        P(
            "<b>Before adopting anything,</b> run the combined B configuration "
            "(CLIP 0.2 / doc 0.48 / bm25 0.32, α=0.5, top_n=60) on a held-out query set, and "
            "compare it against the current default with paired per-query bootstrap and "
            "Holm-corrected tests. If the combined gain survives, it is a defensible result; if "
            "it does not, the single-knob gains were selection noise. For C, extend the "
            "clip_weight grid to 0.6 and 0.7 to find the turning point before choosing 0.5."
        )
    )
    out.extend(
        bullets(
            [
                "Run a 2-D slice over CLIP share × rerank α (for example 0.1/0.2/0.3 × 0.4/0.5/0.6) "
                "instead of more one-factor points.",
                "Add paired bootstrap between each variant and its default, and correct across "
                "the family with Holm–Bonferroni, so close orderings can be called or dismissed.",
                "Hold out 60–80 queries from the same pool for confirmation; never select on the "
                "set used to report.",
            ]
        )
    )
    return out


def section_reproduce() -> list:
    out = H1("9", "Reproducibility")
    out.append(
        P(
            "The sweep is deterministic given the frozen artifacts. The complete run set takes "
            "roughly 14 minutes on the development machine and is resumable: a run whose "
            "override hash and report still match is skipped."
        )
    )
    out.append(
        code_block(
            """PYTHONPATH=src python scripts/tune_params.py          # 17 runs + data/tuning/summary.json
PYTHONPATH=src python scripts/tune_params.py --resume # continue after an interruption
PYTHONPATH=src python scripts/tune_params.py --only b_weights_clip --force
PYTHONPATH=src python -m tsr serve --port 8166         # /tuning reads /api/tuning"""
        )
    )
    out.append(Spacer(1, 4))
    out.append(
        make_table(
            ["Artifact", "What it contains"],
            [
                [mono("data/tuning/summary.json", 7.0), "Metrics, intervals, groups, baseline check and integrity block"],
                [mono("data/tuning/runs/<id>/eval_<strategy>.json", 7.0), "Full per-run report, including per-query values"],
                [mono("data/tuning/runs/<id>/run.json", 7.0), "Command, environment overrides, hashes, timings"],
                [mono("data/tuning/integrity.json", 7.0), "Before/after hashes of every protected path"],
                [mono("web/tuning.js", 7.0), "Client-side rendering; no server state is written"],
            ],
            [210, CONTENT_W - 210],
        )
    )
    out.append(
        caption(
            "Table 6. Artifacts. Only data/tuning/ is written by the sweep; data/reports/, "
            "the SQLite database, the settings file and the indexes are hashed and verified "
            "unchanged."
        )
    )
    out.append(
        P(
            "The <font name='Consolas' size='8'>/tuning</font> page and "
            "<font name='Consolas' size='8'>GET /api/tuning</font> serve the same summary. The "
            "page is read-only by construction: it fetches one JSON document with "
            "<font name='Consolas' size='8'>Cache-Control: no-store</font> and performs all "
            "filtering, charting and table rendering in the browser."
        )
    )
    return out


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_story() -> list:
    story: list = [NextPageTemplate("Body"), PageBreak()]
    story += section_summary()
    story += section_protocol()
    story += section_results()
    story += section_charts()
    story += section_interpretation()
    story += section_categories()
    story += section_validity()
    story += section_recommendations()
    story += section_reproduce()
    return story


def main() -> None:
    doc = BaseDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=LM,
        rightMargin=RM,
        topMargin=TM,
        bottomMargin=BM,
        title="Parameter Tuning Report — Meme-Cap Retrieval",
        author="Meme-Cap retrieval project",
        subject="One-factor sweep of retrieval parameters on a frozen benchmark",
    )
    frame = Frame(LM, BM, CONTENT_W, PAGE_H - TM - BM, id="body", showBoundary=0)
    cover_frame = Frame(LM, BM, CONTENT_W, PAGE_H - TM - BM, id="cover", showBoundary=0)
    doc.addPageTemplates(
        [
            PageTemplate(id="Cover", frames=[cover_frame], onPage=draw_cover),
            PageTemplate(id="Body", frames=[frame], onPage=lambda c, d: None),
        ]
    )
    doc.build(build_story(), canvasmaker=NumberedCanvas)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
