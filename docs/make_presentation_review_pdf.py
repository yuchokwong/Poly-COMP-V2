"""Generate docs/beyond_vision_review.pdf.

A Traditional-Chinese review of docs/Beyond Vision_presentation.pdf: every
numeric and factual claim is checked against the repository artifacts, and the
places where the deck's wording outruns what the system/report supports are
listed with evidence and suggested phrasing.

Run:  python docs/make_presentation_review_pdf.py
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import Counter
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "beyond_vision_review.pdf"

# --------------------------------------------------------------------------- #
# Fonts (Traditional Chinese) and palette
# --------------------------------------------------------------------------- #
FONT_DIR = Path(r"C:\Windows\Fonts")
REG = "JhengHei"
BOLD = "JhengHeiBold"
MONO = "Consolas"


def _register() -> None:
    regular = FONT_DIR / "msjh.ttc"
    bold = FONT_DIR / "msjhbd.ttc"
    if not regular.exists():
        raise SystemExit(f"Missing Traditional-Chinese font: {regular}")
    pdfmetrics.registerFont(TTFont(REG, str(regular), subfontIndex=0))
    if bold.exists():
        pdfmetrics.registerFont(TTFont(BOLD, str(bold), subfontIndex=0))
    else:
        pdfmetrics.registerFont(TTFont(BOLD, str(regular), subfontIndex=0))
    consolas = FONT_DIR / "consola.ttf"
    if consolas.exists():
        pdfmetrics.registerFont(TTFont(MONO, str(consolas)))
    else:
        pdfmetrics.registerFont(TTFont(MONO, str(regular), subfontIndex=0))
    pdfmetrics.registerFontFamily(REG, normal=REG, bold=BOLD, italic=REG, boldItalic=BOLD)


_register()

NAVY = colors.HexColor("#102A43")
INK = colors.HexColor("#1F2933")
BODY = colors.HexColor("#243B53")
MUTED = colors.HexColor("#62717F")
TEAL = colors.HexColor("#0F766E")
TEAL_LIGHT = colors.HexColor("#E6F4F1")
AMBER = colors.HexColor("#B45309")
AMBER_LIGHT = colors.HexColor("#FDF6EC")
RED = colors.HexColor("#B42318")
RED_LIGHT = colors.HexColor("#FEF3F2")
GREEN = colors.HexColor("#047857")
GREEN_LIGHT = colors.HexColor("#ECFDF3")
LIGHT = colors.HexColor("#F5F7FA")
ZEBRA = colors.HexColor("#F7F9FB")
RULE = colors.HexColor("#D9E2EC")
WHITE = colors.white

PAGE_W, PAGE_H = A4
LM = RM = 48
TM = 58
BM = 50
CONTENT_W = PAGE_W - LM - RM


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
def style(name, **kw):
    base = dict(
        fontName=REG, fontSize=9.2, leading=14.2, textColor=BODY,
        spaceBefore=0, spaceAfter=4, wordWrap="CJK",
    )
    base.update(kw)
    return ParagraphStyle(name, **base)


S_TITLE = style("Title", fontName=BOLD, fontSize=24, leading=32, textColor=NAVY, spaceAfter=6)
S_SUB = style("Sub", fontSize=11.5, leading=18, textColor=MUTED, spaceAfter=14)
S_H1 = style("H1", fontName=BOLD, fontSize=15.5, leading=21, textColor=NAVY, spaceAfter=2)
S_H2 = style("H2", fontName=BOLD, fontSize=10.4, leading=15, textColor=TEAL, spaceBefore=10, spaceAfter=2)
S_BODY = style("Body")
S_SMALL = style("Small", fontSize=8.2, leading=12.4, textColor=MUTED)
S_TH = style("TH", fontName=BOLD, fontSize=7.9, leading=11, textColor=WHITE, spaceAfter=0)
S_TD = style("TD", fontSize=8.0, leading=11.8, textColor=INK, spaceAfter=0)
S_TD_SM = style("TDSm", fontSize=7.4, leading=10.8, textColor=INK, spaceAfter=0)
S_FIND = style("Find", fontSize=10.2, leading=15, textColor=INK, spaceAfter=5)
S_FIELD = style("Field", fontSize=8.8, leading=13.6, textColor=BODY, spaceAfter=2.5)
S_QUOTE = style("Quote", fontSize=8.7, leading=13.6, textColor=MUTED, leftIndent=8, spaceAfter=4)
S_CAPTION = style("Caption", fontSize=7.4, leading=10.8, textColor=MUTED, spaceBefore=2, spaceAfter=10)

SEV = {
    "高": (RED, RED_LIGHT),
    "中": (AMBER, AMBER_LIGHT),
    "低": (MUTED, LIGHT),
    "通": (GREEN, GREEN_LIGHT),
}


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def mono(text: str, size: float = 7.7) -> str:
    return f'<font name="{MONO}" size="{size}" color="#334E68">{esc(text)}</font>'


def P(text: str, st=S_BODY) -> Paragraph:
    return Paragraph(text, st)


def H1(number: str, title: str) -> list:
    return [
        Paragraph(f"{number}&nbsp;&nbsp;{esc(title)}", S_H1),
        HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceBefore=2, spaceAfter=10),
    ]


def bullets(items: list[str]) -> list[Paragraph]:
    st = style("Bullet", leftIndent=14, bulletIndent=2, spaceAfter=4)
    return [Paragraph(item, st, bulletText="•") for item in items]


def table(data, widths, *, header: bool = True, font=S_TD, align=None) -> Table:
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.35, RULE),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
    ]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
    for i in range(1, len(data)):
        if i % 2 == 0:
            cmds.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    if align:
        for col, a in enumerate(align):
            cmds.append(("ALIGN", (col, 0), (col, -1), a))
    t.setStyle(TableStyle(cmds))
    return t


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data() -> dict:
    d: dict = {}
    report = json.loads((ROOT / "data/reports/compare.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data/manifest.meta.json").read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "data/eval/memecap_eval_v1.json").read_text(encoding="utf-8"))

    d["generated_at"] = report.get("generated_at", "")
    d["strategies"] = report["strategies"]
    d["ablations"] = report["ablations"]
    d["manifest"] = manifest
    d["frozen"] = frozen
    d["counts"] = Counter(q["category"] for q in frozen)
    d["unique_sources"] = len({q["metadata"]["image_id"] for q in frozen})

    pair = next(
        p for p in report["pairwise"]
        if {p["a"], p["b"]} == {"clip", "vlm"}
    )
    d["ac"] = {c["metric"]: c for c in pair["comparisons"]}
    d["n_boot"] = report.get("n_boot")

    conn = sqlite3.connect(f"file:{ROOT / 'data/gallery.db'}?mode=ro", uri=True)
    try:
        d["splits"] = dict(conn.execute(
            "select split, count(*) from images group by split"
        ).fetchall())
        d["literal_rows"] = conn.execute(
            "select count(*) from metaphors where meaning = ?",
            ("not related to the meme context",),
        ).fetchone()[0]
        d["metaphor_rows"] = conn.execute("select count(*) from metaphors").fetchone()[0]
    finally:
        conn.close()

    api_src = (ROOT / "src/tsr/api.py").read_text(encoding="utf-8")
    d["api_routes"] = len(re.findall(r"@app\.(?:get|put|post|delete)\(", api_src))
    d["html_files"] = len(list((ROOT / "web").glob("*.html")))

    import yaml  # available in the project environment

    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    d["cfg"] = cfg
    return d


def m(strategy: str, key: str) -> float:
    return D["strategies"][strategy]["metrics"][key]


def cat(strategy: str, category: str) -> float:
    return D["strategies"][strategy]["per_category"][category]["mAP"]


def p50(strategy: str) -> float:
    return D["strategies"][strategy]["latency"]["p50_ms"]


def ab(name: str) -> dict:
    return D["ablations"][name]


D = load_data()


# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #
def _footer(c: canvas.Canvas) -> None:
    c.saveState()
    c.setStrokeColor(RULE)
    c.setLineWidth(0.6)
    c.line(LM, BM - 14, PAGE_W - RM, BM - 14)
    c.setFont(REG, 7)
    c.setFillColor(MUTED)
    c.drawString(LM, BM - 26, "Beyond Vision 簡報核對 · 內容以 repo artifacts 為準")
    c.drawRightString(PAGE_W - RM, BM - 26, str(c.getPageNumber()))
    c.restoreState()


def _header(c: canvas.Canvas) -> None:
    c.saveState()
    c.setFont(REG, 7)
    c.setFillColor(MUTED)
    c.drawString(LM, PAGE_H - 30, "簡報內容與專案 artifacts 差異核對")
    c.drawRightString(PAGE_W - RM, PAGE_H - 30, "COMP5523 · Meme-Cap T2I retrieval")
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
# Finding blocks
# --------------------------------------------------------------------------- #
def finding(no: int, sev: str, pages: str, title: str, quote: str,
            actual: str, evidence: str, suggestion: str) -> KeepTogether:
    accent, bg = SEV[sev]
    head = Paragraph(
        f'<font color="{accent.hexval()}"><b>{sev}</b></font>'
        f'&nbsp;&nbsp;<b>F{no}. {esc(title)}</b>'
        f'&nbsp;&nbsp;<font size="7.6" color="{MUTED.hexval()}">投影片 {esc(pages)}</font>',
        S_FIND,
    )
    parts = [
        head,
        Paragraph(f"<b>投影片原文</b>　{quote}", S_QUOTE),
        Paragraph(f"<b>系統實際</b>　{actual}", S_FIELD),
        Paragraph(f"<b>證據</b>　{evidence}", S_FIELD),
        Paragraph(f"<b>建議寫法</b>　{suggestion}", S_FIELD),
    ]
    box = Table([[parts]], colWidths=[CONTENT_W])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 3, accent),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return KeepTogether(box)


# --------------------------------------------------------------------------- #
# Story
# --------------------------------------------------------------------------- #
def build_story() -> list:
    story: list = []
    a, b, c = D["strategies"]["clip"], D["strategies"]["clip_nlp"], D["strategies"]["vlm"]
    ac = D["ac"]["mAP"]
    mf = D["manifest"]
    counts = D["counts"]
    vlm035 = ab("vlm + CLIP 0.35")["per_category"]["visual"]["mAP"]
    b_no_clip = ab("clip_nlp - CLIP")["metrics"]["mAP"]
    b_no_rr = ab("clip_nlp - reranker")["metrics"]["mAP"]
    c_no_clip = ab("vlm only (no CLIP)")["metrics"]["mAP"]
    pct = (cat("vlm", "intent") / cat("clip", "intent") - 1) * 100
    ppt = (cat("vlm", "intent") - cat("clip", "intent")) * 100

    # ---------------------------------------------------------------- cover
    story.append(Spacer(1, 46))
    story.append(P("Beyond Vision 簡報核對報告", S_TITLE))
    story.append(P(
        "11 頁投影片 <font name=\"Consolas\" size=\"9\">docs/Beyond Vision_presentation.pdf</font> "
        "與專案 artifacts 的逐項比對：數字是否一致、結論是否被統計支持、標籤是否與系統相符。",
        S_SUB))

    summary = Table(
        [
            [Paragraph("嚴重度", S_TH), Paragraph("數量", S_TH), Paragraph("代表問題", S_TH)],
            [Paragraph("高", S_TD), Paragraph("4", S_TD),
             Paragraph("C「勝過」A、clip=0.35 的 visual 結論、A 的 worst 誤標、B 的 upper bound", S_TD)],
            [Paragraph("中", S_TD), Paragraph("2", S_TD),
             Paragraph("1,968 的語意、Sentiment / Intent 命名", S_TD)],
            [Paragraph("低", S_TD), Paragraph("7", S_TD),
             Paragraph("頁數、模型數、unseen、grade-1 簡化、延遲並排、Reaction Suggestion、排版", S_TD)],
        ],
        colWidths=[68, 48, 383],
    )
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.35, RULE),
        ("BACKGROUND", (0, 2), (-1, 2), ZEBRA),
    ]))
    story.append(Spacer(1, 8))
    story.append(summary)
    story.append(Spacer(1, 16))

    callout = Table(
        [[P(
            "<b>一句話結論</b>　主要表格的數字幾乎都與 artifacts 一致；需要修正的是"
            "<b>四處結論性措辭</b>：C 並非顯著勝過 A、clip = 0.35 沒有補上 visual 缺口、"
            "B 不是上界、1,968 不是資料集筆數。其餘為命名與簡化問題。",
            style("Callout", fontSize=9.6, leading=15.5, textColor=NAVY))]],
        colWidths=[CONTENT_W],
    )
    callout.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), TEAL_LIGHT),
        ("LINEBEFORE", (0, 0), (0, -1), 3, TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(callout)

    story.append(Spacer(1, 18))
    story.append(P("<b>怎麼讀這份報告</b>", style("ReadH", fontName=BOLD, fontSize=9.4,
                                                  leading=14, textColor=TEAL)))
    story.extend(bullets([
        "第 2 節把所有發現壓成一頁表格，先看嚴重度與頁碼。",
        "第 3–4 節逐項列出投影片原文、系統實際值、證據路徑與建議寫法；"
        "每一項都可以用右側的檔案與 JSON key 重現。",
        "第 6 節列出<i>已核對一致</i>的內容，避免把整份簡報誤判為錯誤。",
    ]))

    story.append(Spacer(1, 22))
    meta = Table(
        [
            [Paragraph("投影片", S_TH), Paragraph("本報告", S_TH)],
            [Paragraph("docs/Beyond Vision_presentation.pdf（11 頁）", S_TD_SM),
             Paragraph(f"beyond_vision_review.pdf（{date.today().isoformat()}）", S_TD_SM)],
            [Paragraph("資料快照", S_TH), Paragraph("核對範圍", S_TH)],
            [Paragraph(D["generated_at"], S_TD_SM),
             Paragraph("評估指標、資料量、ablation、API 與 UI、類別命名", S_TD_SM)],
        ],
        colWidths=[230, 269],
    )
    meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("BACKGROUND", (0, 2), (-1, 2), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LINEBELOW", (0, 0), (-1, -1), 0.35, RULE),
    ]))
    story.append(meta)
    story.append(Spacer(1, 26))
    story.append(P(
        "本報告由 " + mono("docs/make_presentation_review_pdf.py", 7.2)
        + " 讀取 repo artifacts 生成；未修改簡報、程式或設定。", S_SMALL))

    story.append(PageBreak())

    # ---------------------------------------------------------------- method
    story.extend(H1("1", "核對方法與資料來源"))
    story.append(P(
        "所有「系統實際」都以 repo 內可重跑的 artifacts 為準，不採用文件中的二手數字。"
        "簡報頁碼指 PDF 的實際頁次。"))
    story.append(table(
        [
            [Paragraph("Artifact", S_TH), Paragraph("用途", S_TH)],
            [Paragraph(mono("data/reports/compare.json"), S_TD_SM),
             Paragraph("策略指標、per-category mAP、latency、paired bootstrap、Holm、ablations", S_TD_SM)],
            [Paragraph(mono("data/manifest.meta.json"), S_TD_SM),
             Paragraph("6,382 annotations → 6,378 records、4 筆重複", S_TD_SM)],
            [Paragraph(mono("data/gallery.db"), S_TD_SM),
             Paragraph("split 計數；1,968 筆 literal metaphor 列", S_TD_SM)],
            [Paragraph(mono("data/eval/memecap_eval_v1.json"), S_TD_SM),
             Paragraph("240 題、80/80/80、類別名稱、來源 split", S_TD_SM)],
            [Paragraph(mono("configs/default.yaml"), S_TD_SM),
             Paragraph("B/C 權重、模型、index 型別、evaluation 門檻", S_TD_SM)],
            [Paragraph(mono("src/tsr/api.py") + " / " + mono("web/"), S_TD_SM),
             Paragraph("API endpoint 數、HTML 檔數", S_TD_SM)],
        ],
        [200, 312],
    ))
    story.append(Paragraph("表 1. 核對所使用的檔案。", S_CAPTION))
    story.append(P(
        "判讀原則：<b>數字正確</b>與<b>結論正確</b>分開看。簡報的表格數字幾乎全部吻合，"
        "但部分「勝過 / 最佳 / 上界」的句子超過了 artifacts 支持的範圍。"))

    # ---------------------------------------------------------------- summary
    story.extend(H1("2", "摘要：所有發現"))
    rows = [
        [Paragraph("編號", S_TH), Paragraph("嚴重度", S_TH), Paragraph("頁", S_TH),
         Paragraph("發現", S_TH), Paragraph("與系統的差異", S_TH)],
        [Paragraph("F1", S_TD), Paragraph("高", S_TD), Paragraph("p.8", S_TD),
         Paragraph("C「beats」pure CLIP", S_TD),
         Paragraph("delta +0.0124，p_holm = 1.0，CI 跨 0；統計上不顯著", S_TD_SM)],
        [Paragraph("F2", S_TD), Paragraph("高", S_TD), Paragraph("p.8", S_TD),
         Paragraph("clip weight 調到 0.35 可追上 CLIP", S_TD),
         Paragraph("0.35 的 visual mAP = 0.2715，仍低於 CLIP 的 0.3153", S_TD_SM)],
        [Paragraph("F3", S_TD), Paragraph("高", S_TD), Paragraph("p.8", S_TD),
         Paragraph("A 的 worst = 0.444；+9.5%", S_TD),
         Paragraph("A 的 worst 是 visual 0.3153；0.444 是 intent；+9.5 是百分點", S_TD_SM)],
        [Paragraph("F4", S_TD), Paragraph("高", S_TD), Paragraph("pp.4/8", S_TD),
         Paragraph("B 是 upper bound / oracle / ceiling", S_TD),
         Paragraph("B without CLIP = 0.7802 > 0.7608；B 也未看 qrels", S_TD_SM)],
        [Paragraph("F5", S_TD), Paragraph("中", S_TD), Paragraph("p.4", S_TD),
         Paragraph("1,968 noisy entries stripped", S_TD),
         Paragraph("1,968 是 literal metaphor 列；資料集只少 4 筆重複", S_TD_SM)],
        [Paragraph("F6", S_TD), Paragraph("中", S_TD), Paragraph("pp.1/3/8", S_TD),
         Paragraph("Sentiment 與 Intent 混用", S_TD),
         Paragraph("系統類別是 intent（meme_captions），不是 sentiment", S_TD_SM)],
        [Paragraph("F7", S_TD), Paragraph("低", S_TD), Paragraph("多頁", S_TD),
         Paragraph("六項措辭／簡化問題", S_TD),
         Paragraph("見第 5 節，不影響主要數字", S_TD_SM)],
    ]
    story.append(table(rows, [36, 42, 44, 137, 240]))
    story.append(Paragraph("表 2. 依嚴重度排序；F1–F4 會影響評審對結果的解讀。", S_CAPTION))

    story.append(Spacer(1, 8))
    story.append(P("<b>嚴重度定義</b>", style("SevH", fontName=BOLD, fontSize=9.4,
                                               leading=14, textColor=TEAL)))
    story.extend(bullets([
        "<b>高</b>：句子會改變評審對結果的解讀，或把不顯著的差異寫成勝負；必須改。",
        "<b>中</b>：數字正確，但名詞或單位指向錯的東西（資料筆數、類別名稱）；"
        "會誤導但不影響主表。",
        "<b>低</b>：措辭、簡化、排版與單頁並排造成的誤讀；口頭報告前修掉即可。",
    ]))
    story.append(P(
        "未列入的細節（例如動畫順序、圖示選擇）不影響事實陳述，因此不在此報告。", S_SMALL))

    story.append(PageBreak())

    # ---------------------------------------------------------------- high
    story.extend(H1("3", "高嚴重度：會改變結論的四項"))
    story.append(finding(
        1, "高", "p.8",
        "「C beats pure CLIP」在統計上不成立",
        "「Best Annotation-Free Model: C · VLM, mAP = 0.4519 — Beats pure CLIP (0.4395)」",
        f"C 的 mAP 點估計為 {m('vlm','mAP'):.4f}，A 為 {m('clip','mAP'):.4f}，"
        f"delta = <b>+{ac['delta']:.4f}</b>。但 paired bootstrap 的 95% CI 是 "
        f"<b>[{ac['ci_low']:.4f}, {ac['ci_high']:.4f}]</b>，跨過 0；"
        f"p_raw = {ac['p_raw']:.3f}，<b>p_holm = {ac['p_holm']:.1f}</b>，"
        f"<font name=\"Consolas\" size=\"8\">significant = false</font>。"
        "其餘四個指標（MRR、nDCG@10、P@1、R@10）也全部不顯著。"
        "專案報告的結論是 A 與 C 沒有可偵測的差異。",
        mono("data/reports/compare.json") + " → "
        + mono("pairwise[clip, vlm].comparisons[mAP]") +
        "。Holm family = 3 組配對 × 5 指標 = 15 個檢定。",
        "「C 的 mAP 略高於 A（+0.012），但 paired bootstrap 顯示差異不顯著"
        "（95% CI 跨 0，p_holm = 1.0），兩者可視為同等。」",
    ))
    story.append(Spacer(1, 8))
    story.append(finding(
        2, "高", "p.8",
        "pixel weight 調到 0.35 並沒有讓 C 在 visual 上追上 CLIP",
        "「Default Strategy C loses to pure CLIP (0.315) on literal visual queries "
        "<b>until pixel weight is raised to 0.35</b>.」",
        f"在 <font name=\"Consolas\" size=\"8\">vlm + CLIP 0.35</font> 這個 ablation 中，"
        f"C 的 visual mAP 是 <b>{vlm035:.4f}</b>，仍低於 A 的 <b>{cat('clip','visual'):.4f}</b>。"
        f"0.35 只把 C 的整體 mAP 從 {m('vlm','mAP'):.4f} 提高到 "
        f"{ab('vlm + CLIP 0.35')['metrics']['mAP']:.4f}；"
        "「追上 CLIP」的因果與數值都不成立。",
        mono("data/reports/compare.json") + " → "
        + mono("ablations['vlm + CLIP 0.35'].per_category.visual.mAP") +
        "；對照 " + mono("strategies.clip.per_category.visual.mAP") + "。",
        "「提高 pixel weight 會改善 C 的整體 mAP，但 visual 類別仍低於純 CLIP"
        "（0.2715 vs 0.3153），因此 C 的 visual 缺口不是單靠調權重就能補上。」",
    ))
    story.append(Spacer(1, 8))
    story.append(finding(
        3, "高", "p.8",
        "「A 的 worst 0.444」與「+9.5%」的類別和單位都不對",
        "「on Intent queries, Strategy C jumps from <b>Strategy A's worst score of 0.444</b> "
        "up to 0.539 (<b>+9.5%</b>)」",
        f"A 的三個類別是 visual <b>{cat('clip','visual'):.4f}</b>、intent "
        f"<b>{cat('clip','intent'):.4f}</b>、metaphor {cat('clip','metaphor'):.4f}；"
        f"<b>A 的最差是 visual，不是 0.444</b>。C 的 intent 為 {cat('vlm','intent'):.4f}，"
        f"與 A 的差是 <b>{ppt:.1f} 個百分點</b>；若以相對增幅表示則是 "
        f"<b>+{pct:.1f}%</b>。原文把「百分點」寫成「百分比」，又把 A 的中段成績標成 worst。",
        mono("data/reports/compare.json") + " → "
        + mono("strategies.{clip,vlm}.per_category.*.mAP") + "。",
        "「在 intent 類別，A 為 0.444、C 為 0.539，C 高出 9.5 個百分點（相對 +21.3%）；"
        "A 的最差類別其實是 visual（0.315）。」",
    ))
    story.append(Spacer(1, 8))
    story.append(finding(
        4, "高", "pp.4/8",
        "B 不是 upper bound / oracle，0.7608 也不是 ceiling",
        "「B · CLIP + NLP — <b>Human-annotated upper bound</b>」；"
        "「Sets the performance ceiling … Yes (<b>Oracle</b>)」",
        f"B 的預設配置 mAP 是 {m('clip_nlp','mAP'):.4f}，但同一份報告的 ablation 顯示 "
        f"<b>B without CLIP pixels = {b_no_clip:.4f}</b>（+{b_no_clip - m('clip_nlp','mAP'):.4f}），"
        f"比預設更高；因此 0.7608 既不是 B 的最佳配置，也不是系統上界。"
        "B 只索引人類標註，並沒有讀取 qrels，不具備 oracle 的條件；"
        "它是 annotation-rich 的強基準（reference），不是 upper bound。",
        mono("data/reports/compare.json") + " → " + mono("ablations['clip_nlp - CLIP']") +
        "；qrels 只存在於 " + mono("data/eval/*.qrels.jsonl") + "，策略端不讀取。",
        "「B 是人類標註豐富的參考基準（annotation-rich reference）；"
        "移除 CLIP 後 mAP 反而升到 0.7802，說明 0.7608 不是上限。」",
    ))

    story.append(PageBreak())

    # ---------------------------------------------------------------- medium
    story.extend(H1("4", "中嚴重度：數字對，但語意放錯位置"))
    story.append(finding(
        5, "中", "p.4",
        "「1,968 noisy entries stripped」把 metaphor 列當成資料集筆數",
        "「MemeCap: 6,378 verified memes (5,820 trainval + 558 test) · "
        "<b>1,968 noisy entries stripped</b>」",
        f"資料集本身是 <b>{mf['num_annotations']:,} → {mf['num_records']:,}</b>，"
        f"只少了 <b>{mf['num_duplicates']} 筆 SHA-1 重複</b>，沒有刪掉 1,968 張圖。"
        f"<b>1,968</b> 是 <font name=\"Consolas\" size=\"8\">metaphors</font> 表中 "
        f"meaning = <font name=\"Consolas\" size=\"8\">not related to the meme context</font> "
        f"的列數（總共 {D['metaphor_rows']:,} 列），在建立 B/C 文件時被排除。"
        "它既不是圖片，也不是 meme 條目。",
        mono("data/manifest.meta.json") + "；" + mono("data/gallery.db") + " → "
        + mono("select count(*) from metaphors where meaning='not related to the meme context'")
        + f" = {D['literal_rows']:,}。",
        "「資料集去重後為 6,378 張（4 筆重複）；另有 1,968 筆 literal metaphor 列"
        "在建構文件時排除，不計入 B/C 的文字證據。」",
    ))
    story.append(Spacer(1, 8))
    story.append(finding(
        6, "中", "pp.1/3/8",
        "Sentiment 與 Intent 混用，且與系統類別名稱不一致",
        "「Search by <b>Sentiment</b>: abstract emotion → reaction meme」；"
        "表頭「<b>Sentiment</b> (80)」；但 p.7 寫的是「80 <b>Intent</b>」",
        "系統與 <font name=\"Consolas\" size=\"8\">/evaluation</font> 頁使用的類別鍵是 "
        "<b>intent</b>，來源欄位是 <font name=\"Consolas\" size=\"8\">meme_captions</font>"
        "（intended meaning，也就是笑點/意圖），不是情緒正負的分類。"
        "Intent 與 sentiment 是不同概念；同一份 deck 內兩種叫法並存，會讓評審不清楚"
        "到底量的是哪一種。",
        mono("data/eval/memecap_eval_v1.json") + " → " + mono("category ∈ {visual, intent, metaphor}")
        + "；" + mono("web/eval.js") + " 直接以類別鍵渲染表格，沒有改名為 Sentiment。",
        "全份統一使用「Intent（intended meaning）」；若要在口頭表達「情緒型查詢」，"
        "可加一句說明，但不要把它當成類別名稱。",
    ))

    # ---------------------------------------------------------------- low
    story.append(Spacer(1, 12))
    story.extend(H1("5", "低嚴重度：措辭、簡化與排版"))
    low_rows = [
        [Paragraph("頁", S_TH), Paragraph("投影片說法", S_TH), Paragraph("實際情況", S_TH)],
        [Paragraph("p.2", S_TD_SM), Paragraph("Dataset &amp; 3 Models", S_TD_SM),
         Paragraph("3 條 strategies；模型元件有 4 個：CLIP、BGE-small、BGE-reranker、Qwen2.5-VL-3B", S_TD_SM)],
        [Paragraph("p.5", S_TD_SM), Paragraph("7 HTML/JS pages", S_TD_SM),
         Paragraph(f"服務 7 條 page routes，但只有 {D['html_files']} 個 HTML 檔（search.html 重複使用）", S_TD_SM)],
        [Paragraph("p.7", S_TD_SM), Paragraph("558 unseen test memes", S_TD_SM),
         Paragraph("數量正確（240 題來自 240 張不同 test 圖）；但沒有訓練階段，B 也索引 test 圖的標註，"
                   "「unseen」只是 split 說法，真正的保護是 leakage gate", S_TD_SM)],
        [Paragraph("p.7", S_TD_SM), Paragraph("Grade 1 = near-equivalent or shared rare metaphor", S_TD_SM),
         Paragraph("還要求共享 metaphor 的 meaning 相同、且 df ≤ 10；只共享常見標籤不算", S_TD_SM)],
        [Paragraph("p.8", S_TD_SM), Paragraph("B 226 ms（43 ms no-rerank）", S_TD_SM),
         Paragraph(f"43 ms 屬於沒有 reranker 的配置（mAP {b_no_rr:.4f}），不是 {m('clip_nlp','mAP'):.4f} 的延遲；"
                   "並排容易讓人以為同一品質可以 43 ms 跑完", S_TD_SM)],
        [Paragraph("p.1", S_TD_SM), Paragraph("Search &amp; Reaction Suggestion", S_TD_SM),
         Paragraph("系統只有檢索與詳情；repo 內沒有 reaction suggestion 的 endpoint 或 UI", S_TD_SM)],
        [Paragraph("p.4/9", S_TD_SM), Paragraph("「metaphors pair」、「BGE -reranker-base」", S_TD_SM),
         Paragraph("漏右括號；模型名稱連字號位置錯誤（BGE-reranker-base）", S_TD_SM)],
    ]
    story.append(table(low_rows, [34, 172, 293]))
    story.append(Paragraph("表 3. 不影響主要數字，但建議在口頭報告前修掉。", S_CAPTION))

    story.append(PageBreak())

    # ---------------------------------------------------------------- correct
    story.extend(H1("6", "與系統一致的內容"))
    story.append(P(
        "為避免把整份簡報當成錯誤，以下逐項核對結果是<b>一致</b>的；"
        "簡報的主要數據表可以直接沿用。"))
    ok_rows = [
        [Paragraph("項目", S_TH), Paragraph("投影片", S_TH), Paragraph("系統實際", S_TH)],
        [Paragraph("資料量與 split", S_TD_SM), Paragraph("6,378（5,820 + 558）", S_TD_SM),
         Paragraph(f"{mf['num_records']:,}（{D['splits']['trainval']:,} + {D['splits']['test']:,}）", S_TD_SM)],
        [Paragraph("mAP：A / B / C", S_TD_SM), Paragraph("0.4395 / 0.7608 / 0.4519", S_TD_SM),
         Paragraph(f"{m('clip','mAP'):.4f} / {m('clip_nlp','mAP'):.4f} / {m('vlm','mAP'):.4f}", S_TD_SM)],
        [Paragraph("Visual mAP：A / B / C", S_TD_SM), Paragraph("0.315 / 0.631 / 0.247", S_TD_SM),
         Paragraph(f"{cat('clip','visual'):.4f} / {cat('clip_nlp','visual'):.4f} / {cat('vlm','visual'):.4f}", S_TD_SM)],
        [Paragraph("Intent mAP：A / B / C", S_TD_SM), Paragraph("0.444 / 0.739 / 0.539", S_TD_SM),
         Paragraph(f"{cat('clip','intent'):.4f} / {cat('clip_nlp','intent'):.4f} / {cat('vlm','intent'):.4f}", S_TD_SM)],
        [Paragraph("Metaphor mAP：A / B / C", S_TD_SM), Paragraph("0.559 / 0.912 / 0.570", S_TD_SM),
         Paragraph(f"{cat('clip','metaphor'):.4f} / {cat('clip_nlp','metaphor'):.4f} / {cat('vlm','metaphor'):.4f}", S_TD_SM)],
        [Paragraph("p50 latency：A / B / C", S_TD_SM), Paragraph("14 / 226 / 35 ms", S_TD_SM),
         Paragraph(f"{p50('clip'):.2f} / {p50('clip_nlp'):.2f} / {p50('vlm'):.2f} ms", S_TD_SM)],
        [Paragraph("Ablation 位移", S_TD_SM), Paragraph("-0.0721 / +0.0194 / -0.0597", S_TD_SM),
         Paragraph(f"{b_no_rr - m('clip_nlp','mAP'):+.4f} / {b_no_clip - m('clip_nlp','mAP'):+.4f} / "
                   f"{c_no_clip - m('vlm','mAP'):+.4f}", S_TD_SM)],
        [Paragraph("Query 數量與類別", S_TD_SM), Paragraph("240；80/80/80", S_TD_SM),
         Paragraph(f"{len(D['frozen'])}；{counts['visual']}/{counts['intent']}/{counts['metaphor']}", S_TD_SM)],
        [Paragraph("Gates", S_TD_SM), Paragraph("8-gram、短文件、copy run &lt;5、Jaccard &lt;0.6", S_TD_SM),
         Paragraph("與 src/tsr/evaluate.py 的 check_leakage 一致", S_TD_SM)],
        [Paragraph("模型與維度", S_TD_SM), Paragraph("CLIP ViT-L-14 768-d；BGE 384-d", S_TD_SM),
         Paragraph("config 與 index meta 一致", S_TD_SM)],
        [Paragraph("Index / 後端", S_TD_SM), Paragraph("FAISS IndexFlatIP；FastAPI", S_TD_SM),
         Paragraph("index.type = flat；IndexFlatIP；FastAPI", S_TD_SM)],
        [Paragraph("API endpoints", S_TD_SM), Paragraph("18", S_TD_SM),
         Paragraph(f"{D['api_routes']} 個 @app.get/put 路由", S_TD_SM)],
        [Paragraph("VLM 覆蓋", S_TD_SM), Paragraph("6,378 descriptions", S_TD_SM),
         Paragraph("probe.vlm_captions = 6,378", S_TD_SM)],
        [Paragraph("B / C 權重", S_TD_SM), Paragraph("B 0.50/0.30/0.20；C 0.2 CLIP", S_TD_SM),
         Paragraph("configs/default.yaml 相同", S_TD_SM)],
        [Paragraph("VLM 六欄位", S_TD_SM), Paragraph("scene/OCR/meaning/<br/>metaphors/emotions/keywords", S_TD_SM),
         Paragraph("encoders/vlm.py 的 prompt schema 相同", S_TD_SM)],
    ]
    story.append(table(ok_rows, [118, 183, 198], font=S_TD_SM))

    # ---------------------------------------------------------------- method note
    story.append(Spacer(1, 18))
    story.extend(H1("7", "方法論備註：缺了顯著性，才會出現「beats」"))
    story.append(P(
        "簡報第 7–9 頁完整交代了 leakage gate 與 ablation，但<b>沒有提到 bootstrap CI 與 "
        "Holm–Bonferroni</b>；而專案的 evaluation 設計正是以這兩者為核心。"
        "少了這一層，點估計的排序就會被寫成「最好 / 勝過 / 上界」。"))
    story.extend(bullets([
        f"A 與 C 的 mAP 差為 +{ac['delta']:.4f}，95% CI "
        f"[{ac['ci_low']:.4f}, {ac['ci_high']:.4f}]，p_holm = {ac['p_holm']:.1f}；"
        "這在報告裡是「無顯著差異」，不是「C 勝」。",
        "B 對 A、B 對 C 的五個指標都顯著（p_holm = 0.015），這部分簡報的「B 最好」是站得住的；"
        "但「upper bound」仍是比「顯著較好」更強的宣稱。",
        f"Bootstrap 解析度下限是 1/{D['n_boot']} = 0.001；報告中的 0.001 不應被解讀為精確的 p 值。",
    ]))
    story.append(Spacer(1, 6))
    story.append(P(
        "建議在口頭報告時補一句：「B 的優勢在 Holm 校正後仍顯著；A 與 C 的差異則落在雜訊範圍內。」"
        "這樣既保留 B 的結論，也不會讓 A/C 的比較被追問。", S_SMALL))

    # ---------------------------------------------------------------- closing
    story.append(Spacer(1, 14))
    story.append(P(
        "本報告只做核對，未修改簡報、程式或設定。所有數值可由 "
        + mono("python -m tsr compare") + " 與 "
        + mono("data/reports/compare.json") + " 重現。", S_SMALL))

    return story


def main() -> None:
    doc = BaseDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=LM,
        rightMargin=RM,
        topMargin=TM,
        bottomMargin=BM,
        title="Beyond Vision 簡報核對報告",
        author="Meme-Cap T2I retrieval",
        subject="Beyond Vision presentation vs repository artifacts",
    )
    frame_first = Frame(LM, BM, CONTENT_W, PAGE_H - TM - BM, id="first")
    frame_later = Frame(LM, BM, CONTENT_W, PAGE_H - TM - BM, id="later")
    doc.addPageTemplates([
        PageTemplate(id="first", frames=[frame_first], onPage=on_first),
        PageTemplate(id="later", frames=[frame_later], onPage=on_later),
    ])
    story = build_story()
    from reportlab.platypus import NextPageTemplate
    story.insert(0, NextPageTemplate("later"))
    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
