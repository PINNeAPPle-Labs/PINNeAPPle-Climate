"""Shared figure style + PDF paper builder for the PINNeAPPle experiment reports.

Figures: one validated categorical order (fixed, never cycled), single-hue sequential blue ramp,
blue<->red diverging ramp with a neutral grey midpoint, recessive axes, no dual axes.
PDF: reportlab/platypus, STIX serif (full Greek/math glyphs), numbered sections, figures, tables,
captions, references, page numbers.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

# ----------------------------------------------------------------------------- palette
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f3a8a7", "#e34948", "#9c1f1f"])

_MPL_FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def setup_mpl():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlesize": 9.5, "axes.titlecolor": INK,
        "axes.titleweight": "bold", "axes.linewidth": 0.7, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5, "legend.frameon": False,
        "lines.linewidth": 1.6, "lines.markersize": 4.5, "axes.prop_cycle": matplotlib.cycler(color=CAT),
        "figure.dpi": 150, "savefig.dpi": 220, "savefig.bbox": "tight", "figure.facecolor": "white",
        "image.cmap": "seq_blue",
    })
    for cm_ in (SEQ, DIV):
        try:
            matplotlib.colormaps.register(cm_)
        except ValueError:
            pass


def savefig(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


# ----------------------------------------------------------------------------- PDF

_SUP = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ", "0123456789+-=()ni"))
_SUB = dict(zip("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₚₜ", "0123456789+-=()aeoxpt"))


def sanitize(text: str) -> str:
    """Turn Unicode super/subscript runs into reportlab <super>/<sub> markup (the glyphs are missing in most fonts)."""
    out, i = [], 0
    while i < len(text):
        for table, tag in ((_SUP, "super"), (_SUB, "sub")):
            if text[i] in table:
                j = i
                while j < len(text) and text[j] in table:
                    j += 1
                out.append(f"<{tag}>" + "".join(table[c] for c in text[i:j]) + f"</{tag}>")
                i = j
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _para(text, style, **kw):
    return Paragraph(sanitize(str(text)), style, **kw)

def _fonts():
    if "STIX" in pdfmetrics.getRegisteredFontNames():
        return
    pdfmetrics.registerFont(TTFont("STIX", str(_MPL_FONT / "STIXGeneral.ttf")))
    pdfmetrics.registerFont(TTFont("STIX-B", str(_MPL_FONT / "STIXGeneralBol.ttf")))
    pdfmetrics.registerFont(TTFont("STIX-I", str(_MPL_FONT / "STIXGeneralItalic.ttf")))
    pdfmetrics.registerFont(TTFont("STIX-BI", str(_MPL_FONT / "STIXGeneralBolIta.ttf")))
    pdfmetrics.registerFont(TTFont("Mono", str(_MPL_FONT / "DejaVuSansMono.ttf")))
    from reportlab.lib.fonts import addMapping
    addMapping("STIX", 0, 0, "STIX"); addMapping("STIX", 1, 0, "STIX-B")
    addMapping("STIX", 0, 1, "STIX-I"); addMapping("STIX", 1, 1, "STIX-BI")


class Paper:
    def __init__(self, title: str, subtitle: str = "", authors: str = "PINNeAPPle Labs",
                 affiliation: str = "PINNeAPPle Labs — Physics AI research notes",
                 date: Optional[str] = None, report_id: str = ""):
        _fonts()
        self.title, self.subtitle, self.authors, self.affiliation = title, subtitle, authors, affiliation
        self.date = date or _dt.date.today().strftime("%B %d, %Y")
        self.report_id = report_id
        self.story: List = []
        self.n_sec = self.n_sub = self.n_fig = self.n_tab = self.n_eq = 0
        base = dict(fontName="STIX", textColor=colors.HexColor(INK))
        self.S = {
            "title": ParagraphStyle("t", fontSize=17.5, leading=21.5, alignment=TA_CENTER, spaceAfter=4, fontName="STIX-B", textColor=colors.HexColor(INK)),
            "subtitle": ParagraphStyle("st", fontSize=11, leading=14, alignment=TA_CENTER, spaceAfter=6, fontName="STIX-I", textColor=colors.HexColor(INK2)),
            "meta": ParagraphStyle("m", fontSize=9.5, leading=12, alignment=TA_CENTER, textColor=colors.HexColor(INK2), fontName="STIX"),
            "abs_h": ParagraphStyle("ah", fontSize=10, leading=13, fontName="STIX-B", spaceBefore=10, spaceAfter=3, textColor=colors.HexColor(INK)),
            "abs": ParagraphStyle("a", fontSize=9.6, leading=12.6, alignment=TA_JUSTIFY, leftIndent=18, rightIndent=18, **base),
            "h1": ParagraphStyle("h1", keepWithNext=1, fontSize=12.5, leading=15, fontName="STIX-B", spaceBefore=12, spaceAfter=5, textColor=colors.HexColor(INK)),
            "h2": ParagraphStyle("h2", keepWithNext=1, fontSize=10.8, leading=13.5, fontName="STIX-B", spaceBefore=8, spaceAfter=3, textColor=colors.HexColor(INK)),
            "p": ParagraphStyle("p", fontSize=10.2, leading=13.6, alignment=TA_JUSTIFY, spaceAfter=5, **base),
            "bullet": ParagraphStyle("b", fontSize=10.2, leading=13.4, leftIndent=14, bulletIndent=4, spaceAfter=2, alignment=TA_JUSTIFY, **base),
            "cap": ParagraphStyle("c", fontSize=8.9, leading=11.2, alignment=TA_JUSTIFY, spaceBefore=3, spaceAfter=9, textColor=colors.HexColor(INK2), fontName="STIX"),
            "eq": ParagraphStyle("e", fontSize=10.4, leading=14, alignment=TA_CENTER, spaceBefore=3, spaceAfter=5, fontName="STIX-I", textColor=colors.HexColor(INK)),
            "cell": ParagraphStyle("cell", fontSize=8.4, leading=10.2, fontName="STIX", textColor=colors.HexColor(INK)),
            "cellb": ParagraphStyle("cellb", fontSize=8.4, leading=10.2, fontName="STIX-B", textColor=colors.HexColor(INK)),
            "ref": ParagraphStyle("r", fontSize=8.9, leading=11.2, leftIndent=16, firstLineIndent=-16, spaceAfter=2.5, **base),
            "code": ParagraphStyle("code", fontSize=7.8, leading=9.6, fontName="Mono", leftIndent=10, textColor=colors.HexColor(INK2), spaceAfter=4),
            "box": ParagraphStyle("box", fontSize=9.6, leading=12.6, alignment=TA_JUSTIFY, **base),
        }
        self._front()

    # -- front matter
    def _front(self):
        self.story += [Spacer(1, 0.3 * cm), _para(self.title, self.S["title"])]
        if self.subtitle:
            self.story.append(_para(self.subtitle, self.S["subtitle"]))
        self.story += [_para(self.authors, self.S["meta"]), _para(self.affiliation, self.S["meta"]),
                       _para(self.date + (f" · {self.report_id}" if self.report_id else ""), self.S["meta"]),
                       Spacer(1, 0.2 * cm)]

    def abstract(self, text: str, keywords: Sequence[str] = ()):
        self.story.append(_para("Abstract", self.S["abs_h"]))
        self.story.append(_para(text, self.S["abs"]))
        if keywords:
            self.story.append(Spacer(1, 3))
            self.story.append(_para("<b>Keywords:</b> " + "; ".join(keywords), self.S["abs"]))
        self.story.append(Spacer(1, 6))

    def box(self, title: str, lines: Sequence[str]):
        """Highlighted summary box (e.g. 'What this study shows / does not show')."""
        rows = [[_para(f"<b>{title}</b>", self.S["box"])]] + [[_para("• " + l, self.S["box"])] for l in lines]
        t = Table(rows, colWidths=[16.2 * cm])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f3f0")),
                               ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#c9c7c0")),
                               ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                               ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
        self.story += [Spacer(1, 4), t, Spacer(1, 8)]

    # -- body
    def section(self, title: str):
        self.n_sec += 1
        self.n_sub = 0
        self.story.append(_para(f"{self.n_sec}&nbsp;&nbsp;{title}", self.S["h1"]))

    def subsection(self, title: str):
        self.n_sub += 1
        self.story.append(_para(f"{self.n_sec}.{self.n_sub}&nbsp;&nbsp;{title}", self.S["h2"]))

    def p(self, *texts: str):
        for t in texts:
            self.story.append(_para(t, self.S["p"]))

    def bullets(self, items: Sequence[str]):
        for it in items:
            self.story.append(_para(it, self.S["bullet"], bulletText="•"))
        self.story.append(Spacer(1, 3))

    def eq(self, text: str) -> int:
        self.n_eq += 1
        t = Table([[_para(text, self.S["eq"]), _para(f"({self.n_eq})", self.S["p"])]], colWidths=[14.6 * cm, 1.6 * cm])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        self.story.append(t)
        return self.n_eq

    def code(self, text: str):
        for line in text.strip("\n").split("\n"):
            self.story.append(_para(line.replace(" ", "&nbsp;").replace("<", "&lt;") or "&nbsp;", self.S["code"]))

    def figure(self, path: str, caption: str, width_cm: float = 15.5) -> int:
        from reportlab.lib.utils import ImageReader
        self.n_fig += 1
        iw, ih = ImageReader(path).getSize()
        w = width_cm * cm
        h = w * ih / iw
        if h > 19 * cm:
            h = 19 * cm
            w = h * iw / ih
        self.story.append(KeepTogether([Image(path, width=w, height=h),
                                        _para(f"<b>Figure {self.n_fig}.</b> {caption}", self.S["cap"])]))
        return self.n_fig

    def next_fig(self) -> int:
        return self.n_fig + 1

    def next_tab(self) -> int:
        return self.n_tab + 1

    def table(self, rows: Sequence[Sequence], caption: str, col_widths_cm: Optional[Sequence[float]] = None,
              header_rows: int = 1, highlight_rows: Sequence[int] = ()) -> int:
        self.n_tab += 1
        data = [[c if not isinstance(c, str) else _para(c, self.S["cellb"] if r < header_rows else self.S["cell"])
                 for c in row] for r, row in enumerate(rows)]
        cw = [w * cm for w in col_widths_cm] if col_widths_cm else None
        t = Table(data, colWidths=cw, repeatRows=header_rows, hAlign="CENTER")
        st = [("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.HexColor(INK)),
              ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.6, colors.HexColor(INK)),
              ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.HexColor(INK)),
              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
              ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5)]
        for r in highlight_rows:
            st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#eef4fc")))
        t.setStyle(TableStyle(st))
        self.story.append(KeepTogether([_para(f"<b>Table {self.n_tab}.</b> {caption}", self.S["cap"]), t, Spacer(1, 8)]))
        return self.n_tab

    def references(self, refs: Sequence[str]):
        self.story.append(_para("References", self.S["h1"]))
        for i, r in enumerate(refs, 1):
            self.story.append(_para(f"[{i}]&nbsp;&nbsp;{r}", self.S["ref"]))

    def pagebreak(self):
        self.story.append(PageBreak())

    def build(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        title = self.title

        def deco(canvas, doc):
            canvas.saveState()
            canvas.setFont("STIX", 8)
            canvas.setFillColor(colors.HexColor(MUTED))
            if doc.page > 1:
                canvas.drawString(2.2 * cm, A4[1] - 1.3 * cm, (title[:95] + "…") if len(title) > 96 else title)
            canvas.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, f"{doc.page}")
            canvas.drawString(2.2 * cm, 1.2 * cm, "PINNeAPPle Labs · technical report")
            canvas.restoreState()

        doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.0 * cm,
                                bottomMargin=2.0 * cm, title=self.title, author=self.authors, subject=self.subtitle)
        doc.build(self.story, onFirstPage=deco, onLaterPages=deco)
        return path


def fmt(x, nd=3):
    """Compact number formatting for tables."""
    if x is None:
        return "—"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if x != x:
        return "n/a"
    if abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"
