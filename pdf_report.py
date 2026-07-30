"""
pdf_report.py — the student report (returns PDF bytes).

Structure
---------
  1  Cover              student, prediction, confidence, risk level, summary
  2  Prediction results E3 / E2 / E1 with comparison graph
  3  SHAP explainability waterfall + top positive / negative contributors
  4  Student profile     every feature: description, value, ideal, status, influence
  5  Strengths
  6  Weaknesses
  7  Recommendations     immediate, each measured by re-running the model
  8  Suggestions         long-term
  9  Intervention priority
 10  Final summary       teacher / parent / student
  A  Appendix - what each question means
  B  Appendix - method, data quality and limitations
"""
import io
import os
import re
import tempfile
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, Image,
                                KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

from spp_engine import EXP_NAME, EXP_ORDER, EXP_WINDOW, STATUS_COLOR, Engine

ACCENT = colors.HexColor("#22364a")
ACCENT_L = colors.HexColor("#e8edf2")
RISK = colors.HexColor("#c0392b")
OK = colors.HexColor("#1e8449")
WARN = colors.HexColor("#d97706")
MUTED = colors.HexColor("#6b7280")
LINE = colors.HexColor("#d6dde4")

PAGE_W, PAGE_H = A4
LM = RM = 16 * mm
CW = PAGE_W - LM - RM

_UNSAFE = {"\u2192": "->", "\u2190": "<-", "\u2265": ">=", "\u2264": "<=",
           "\u2022": "-", "\u2013": "-", "\u2014": "-", "\u25b2": "^", "\u25bc": "v",
           "\u2713": "yes", "\u2717": "no", "\u00b7": "-"}


def _safe(t) -> str:
    s = str(t)
    for k, v in _UNSAFE.items():
        s = s.replace(k, v)
    s = re.sub(r"[\U00010000-\U0010ffff]", "", s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _hx(c) -> str:
    return "#%02x%02x%02x" % (int(c.red * 255), int(c.green * 255), int(c.blue * 255))


def _scol(status):
    return STATUS_COLOR.get(status, "#6b7280")


# ------------------------------------------------------------------- styles
def _styles():
    ss = getSampleStyleSheet()
    body = ss["Normal"]
    body.fontName, body.fontSize, body.leading = "Helvetica", 9.5, 13
    add = lambda n, **kw: ss.add(ParagraphStyle(n, **kw))
    add("H1", parent=ss["Heading1"], fontSize=16, leading=19, textColor=ACCENT,
        spaceBefore=2, spaceAfter=6)
    add("H2", parent=ss["Heading2"], fontSize=12.5, leading=15, textColor=ACCENT,
        spaceBefore=10, spaceAfter=4)
    add("H3", parent=ss["Heading3"], fontSize=10.5, leading=13, textColor=ACCENT,
        spaceBefore=8, spaceAfter=3)
    add("Small", parent=body, fontSize=7.8, leading=10, textColor=MUTED)
    add("Cell", parent=body, fontSize=8.2, leading=10.5)
    add("CellB", parent=body, fontSize=8.2, leading=10.5, fontName="Helvetica-Bold")
    add("CellS", parent=body, fontSize=7.4, leading=9.4, textColor=MUTED)
    add("Lead", parent=body, fontSize=10, leading=14)
    add("Big", parent=body, fontSize=30, leading=33, fontName="Helvetica-Bold", alignment=1)
    add("Mid", parent=body, fontSize=16, leading=19, fontName="Helvetica-Bold", alignment=1)
    add("Lbl", parent=body, fontSize=8.3, leading=10.5, alignment=1, textColor=MUTED)
    add("Tick", parent=body, fontSize=9, leading=12.5, leftIndent=11, bulletIndent=2,
        spaceAfter=2)
    return ss


def _tbl(rows, widths, ss, header=True, zebra=True):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONTSIZE", (0, 0), (-1, -1), 8.2), ("GRID", (0, 0), (-1, -1), .4, LINE),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), ACCENT),
               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    if zebra:
        st.append(("ROWBACKGROUNDS", (0, 1), (-1, -1),
                   [colors.white, colors.HexColor("#f7f9fb")]))
    t.setStyle(TableStyle(st))
    return t


def _status_p(status, ss):
    return Paragraph(f'<font color="{_scol(status)}"><b>{_safe(status)}</b></font>', ss["Cell"])


def _panel(flow, bg=ACCENT_L, border=None, pad=8):
    t = Table([[flow]], colWidths=[CW])
    st = [("BACKGROUND", (0, 0), (-1, -1), bg), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), pad), ("RIGHTPADDING", (0, 0), (-1, -1), pad),
          ("TOPPADDING", (0, 0), (-1, -1), pad), ("BOTTOMPADDING", (0, 0), (-1, -1), pad)]
    if border:
        st.append(("LINEBEFORE", (0, 0), (0, -1), 3, border))
    t.setStyle(TableStyle(st))
    return t


def _img(path, max_w=CW, max_h=120 * mm):
    iw, ih = ImageReader(path).getSize()
    w, h = max_w, max_w * ih / iw
    if h > max_h:
        h, w = max_h, max_h * iw / ih
    return Image(path, width=w, height=h)


# -------------------------------------------------------------------- charts
def _save(path, fig, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _wrap(s, n):
    words, out, cur = str(s).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n and cur:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    out.append(cur)
    return "\n".join(out[:3])


def _chart_horizons(results, tmp):
    order = sorted(results.values(), key=lambda r: EXP_ORDER.get(r["exp"], 9))
    names = [f"{r['exp']}\n{r['exp_name']}" for r in order]
    probs = [r["fail_prob"] * 100 for r in order]
    lo = [max(.1, (r["fail_prob"] - r["confidence"]["interval"][0]) * 100) for r in order]
    hi = [max(.1, (r["confidence"]["interval"][1] - r["fail_prob"]) * 100) for r in order]
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    bars = ax.bar(names, probs, width=.5, zorder=3,
                  color=[r["risk_colour"] for r in order])
    ax.errorbar(names, probs, yerr=[lo, hi], fmt="none", ecolor="#334155",
                elinewidth=1.2, capsize=6, zorder=4)
    ax.axhline(50, ls="--", lw=1, color="#94a3b8", zorder=2)
    ax.text(len(names) - .38, 52, "decision threshold", fontsize=7, color="#64748b")
    ax.set_ylabel("Failure probability (%)", fontsize=8.5)
    ax.set_ylim(0, 108)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="#eef2f6", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for b, p, r in zip(bars, probs, order):
        ax.text(b.get_x() + b.get_width() / 2, p + 4.5, f"{p:.0f}%", ha="center",
                fontsize=9, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 3, r["risk_level"], ha="center",
                fontsize=7, color="white", fontweight="bold")
    return _save(os.path.join(tmp, "hz.png"), fig)


def _chart_conf(conf, tmp, tag):
    parts = [("Decision margin", conf["margin"]), ("Model stability", conf["stability"]),
             ("Typical of cohort", conf["typicality"]), ("Inputs supplied", conf["completeness"])]
    fig, ax = plt.subplots(figsize=(3.5, 1.65))
    y = np.arange(len(parts))[::-1]
    vals = [v for _, v in parts]
    ax.barh(y, [1] * len(vals), color="#eef2f6", height=.55, zorder=1)
    ax.barh(y, vals, height=.55, zorder=3,
            color=["#1e8449" if v >= .7 else "#d97706" if v >= .45 else "#c0392b" for v in vals])
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _ in parts], fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .5, 1])
    ax.set_xticklabels(["0", "", "1"], fontsize=7)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    for yi, v in zip(y, vals):
        ax.text(min(v + .03, .88), yi, f"{v:.2f}", va="center", fontsize=7.2)
    return _save(os.path.join(tmp, f"cf_{tag}.png"), fig)


def _chart_waterfall(res, tmp):
    plt.close("all")
    shap.plots.waterfall(res["explanation"], max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(7.6, 5.0)
    fig.suptitle(f"{res['exp']} ({res['exp_name']}) - how each answer moved the prediction",
                 fontsize=9, y=1.01)
    return _save(os.path.join(tmp, f"wf_{res['exp']}.png"), fig, dpi=140)


def _chart_contrib(res, tmp):
    """Top positive and negative SHAP contributors side by side."""
    fs = sorted(res["factors"], key=lambda f: f["shap"])
    neg = [f for f in fs if f["shap"] < 0][:6]
    pos = [f for f in fs if f["shap"] > 0][-6:]
    items = neg + pos
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(6.6, max(2.0, .34 * len(items) + .8)))
    y = np.arange(len(items))
    vals = [f["shap"] for f in items]
    ax.barh(y, vals, color=["#1e8449" if v < 0 else "#c0392b" for v in vals],
            height=.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{f['label'][:34]}  ({f['display'][:14]})" for f in items], fontsize=7.4)
    ax.axvline(0, color="#334155", lw=.9)
    ax.set_xlabel("SHAP value  (negative = toward passing, positive = toward failing)", fontsize=7.6)
    ax.grid(axis="x", color="#eef2f6", zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    for yi, v in zip(y, vals):
        ax.text(v + (.02 if v >= 0 else -.02), yi, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7)
    return _save(os.path.join(tmp, f"ct_{res['exp']}.png"), fig)


def _chart_plan(res, tmp):
    steps = res["what_if"].get("steps") or []
    if not steps:
        return None
    labels = ["Now"] + [f"+{s['label']}" for s in steps]
    probs = [res["what_if"]["base_prob"] * 100] + [s["new_prob"] * 100 for s in steps]
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.plot(range(len(probs)), probs, "-o", color="#22364a", lw=1.8, ms=6, zorder=3)
    ax.fill_between(range(len(probs)), probs, 0, color="#22364a", alpha=.07, zorder=2)
    ax.axhline(50, ls="--", lw=1, color="#94a3b8", zorder=1)
    for i, p in enumerate(probs):
        ax.annotate(f"{p:.0f}%", (i, p), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.2, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([_wrap(l, 14) for l in labels], fontsize=7.2)
    ax.set_ylabel("Failure probability (%)", fontsize=8.5)
    ax.set_ylim(0, 112)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color="#eef2f6", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return _save(os.path.join(tmp, f"pl_{res['exp']}.png"), fig)


# --------------------------------------------------------------- page canvas
def _decorate(canvas, doc, student_id, gen):
    canvas.saveState()
    canvas.setFillColor(ACCENT)
    canvas.rect(0, PAGE_H - 13 * mm, PAGE_W, 13 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(LM, PAGE_H - 8.6 * mm, "Student Performance Report")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(PAGE_W - RM, PAGE_H - 8.6 * mm, _safe(student_id))
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(.5)
    canvas.line(LM, 13 * mm, PAGE_W - RM, 13 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.2)
    canvas.drawString(LM, 9 * mm, f"Generated {gen}  |  Decision support only - not a "
                                  f"substitute for professional judgement")
    canvas.drawRightString(PAGE_W - RM, 9 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


# ============================================================== 1. COVER
def _cover(story, ss, results, ov, student_id, tmp, engine):
    head = results[ov["exp"]]
    story.append(Paragraph("Student Performance Report", ss["H1"]))
    story.append(Paragraph(
        _safe(f"Student: {student_id}    |    Generated: {datetime.now():%d %B %Y}    |    "
              f"Horizons run: {', '.join(ov['horizons'])}"), ss["Lead"]))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", color=LINE))
    story.append(Spacer(1, 10))

    rc = colors.HexColor(ov["risk_colour"])
    ccol = OK if ov["level"] == "High" else WARN if ov["level"] == "Moderate" else RISK
    lo, hi = ov["interval"]
    tiles = [[
        [Paragraph(f'<font color="{_hx(rc)}">{ov["fail_prob"]*100:.0f}%</font>', ss["Big"]),
         Paragraph("failure probability", ss["Lbl"])],
        [Paragraph(f'<font color="{_hx(OK)}">{ov["pass_prob"]*100:.0f}%</font>', ss["Big"]),
         Paragraph("pass probability", ss["Lbl"])],
        [Paragraph(f'<font color="{_hx(ccol)}">{ov["confidence"]*100:.0f}%</font>', ss["Big"]),
         Paragraph(f"confidence ({ov['level']})", ss["Lbl"])],
        [Paragraph(f'<font color="{_hx(rc)}">{_safe(ov["risk_level"])}</font>', ss["Mid"]),
         Paragraph("risk level", ss["Lbl"])],
    ]]
    t = Table(tiles, colWidths=[CW / 4.0] * 4)
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), .6, LINE),
                           ("INNERGRID", (0, 0), (-1, -1), .6, LINE),
                           ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fbfcfd")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 10),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(_panel([Paragraph(
        _safe(f"Likely range {lo*100:.0f}-{hi*100:.0f}%. {ov['note']}"), ss["Cell"])]))
    story.append(Spacer(1, 10))

    # ---- overall summary
    story.append(Paragraph("Overall summary", ss["H2"]))
    verdict_txt = ("The student is likely to pass." if ov["verdict"] == "On track"
                   else "The student is at risk of failing.")
    story.append(Paragraph(f"<b>{_safe(verdict_txt)}</b>", ss["Lead"]))
    story.append(Spacer(1, 4))

    def _bul(title, items, colour, fmt, empty):
        story.append(Paragraph(f'<font color="{_hx(colour)}"><b>{title}</b></font>', ss["H3"]))
        if not items:
            story.append(Paragraph(_safe(empty), ss["Small"]))
            return
        for x in items:
            story.append(Paragraph(_safe(fmt(x)), ss["Tick"], bulletText="-"))

    _bul("Strongest positive indicators", head["strengths"][:3], OK,
         lambda f: f"{f['label']} - {f['display']} ({f['status']}). {f['comment']}.",
         "No factor currently sits above the passing benchmark.")
    _bul("Major concerns", head["concerns"][:3], RISK,
         lambda f: f"{f['label']} - {f['display']} ({f['status']}). {f['comment']}.",
         "No factor currently sits in the range where students fail.")

    story.append(Spacer(1, 6))
    story.append(_panel(
        [Paragraph(f'<b>Immediate intervention: '
                   f'<font color="{_hx(rc)}">{_safe(ov["urgency"]).upper()}</font></b>',
                   ss["Lead"]),
         Spacer(1, 2),
         Paragraph(_safe(ov["urgency_advice"]), ss["Cell"])],
        bg=colors.HexColor("#fdf3f2") if ov["risk_level"] in ("HIGH", "CRITICAL")
        else colors.HexColor("#eaf4ec"), border=rc))

    # ---- data quality warnings
    warns = []
    if head.get("leaks_used"):
        warns.append("TARGET LEAKAGE: the model for this horizon uses "
                     + ", ".join(head["leaks_used"]) +
                     ". These columns define the pass/fail outcome, so the reported "
                     "accuracy is not a real forecast. See Appendix B.")
    if engine is not None and getattr(engine, "owners", None):
        warns.append("This dataset merges five separate studies, so each answer is graded "
                     "against the study it came from rather than the whole file. See Appendix B.")
    if len(results) > 1 and not ov["agree"]:
        warns.append("The horizons disagree on the verdict. Later horizons see more "
                     f"information, so {ov['exp']} carries the most weight, but the "
                     "disagreement itself indicates a changing trajectory.")
    if warns:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Read this first", ss["H3"]))
        for w in warns:
            story.append(Paragraph(_safe(w), ss["Tick"], bulletText="!"))


# ======================================================= 2. PREDICTION RESULTS
def _results_section(story, ss, results, ov, tmp):
    story.append(PageBreak())
    story.append(Paragraph("Prediction results", ss["H1"]))
    story.append(Paragraph(
        "Three models are trained on progressively more information. E3 is available on day "
        "one but knows least; E1 is the most accurate but arrives late.", ss["Small"]))
    story.append(Spacer(1, 6))
    story.append(_img(_chart_horizons(results, tmp), max_h=62 * mm))
    story.append(Spacer(1, 6))

    rows = [["Model", "Stage", "Data it uses", "Fail", "Pass", "Likely range",
             "Confidence", "Risk", "Verdict"]]
    for r in sorted(results.values(), key=lambda r: EXP_ORDER.get(r["exp"], 9)):
        c = r["confidence"]
        rows.append([Paragraph(f"<b>{r['exp']}</b>", ss["Cell"]),
                     Paragraph(_safe(r["exp_name"]), ss["Cell"]),
                     Paragraph(_safe(r["window"]), ss["CellS"]),
                     Paragraph(f"{r['fail_prob']*100:.1f}%", ss["CellB"]),
                     Paragraph(f"{r['pass_prob']*100:.1f}%", ss["Cell"]),
                     Paragraph(f"{c['interval'][0]*100:.0f}-{c['interval'][1]*100:.0f}%", ss["Cell"]),
                     Paragraph(f"{_safe(c['level'])} ({c['score']*100:.0f}%)", ss["Cell"]),
                     Paragraph(f'<font color="{r["risk_colour"]}"><b>{_safe(r["risk_level"])}</b></font>',
                               ss["Cell"]),
                     Paragraph(f'<font color="{_hx(RISK if r["verdict"]=="AT RISK" else OK)}">'
                               f'<b>{_safe(r["verdict"])}</b></font>', ss["Cell"])])
    story.append(_tbl(rows, [13 * mm, 17 * mm, 40 * mm, 14 * mm, 14 * mm, 20 * mm,
                             24 * mm, 20 * mm, CW - 162 * mm], ss))
    story.append(Spacer(1, 6))
    story.append(_panel([Paragraph(
        _safe("How to read confidence: it combines how far the probability sits from the 50% "
              "threshold, how stable the prediction was across the model's trees, how typical "
              "this student is of the training data, and how many inputs were supplied."),
        ss["Cell"])]))


# ================================================== 3-9. PER-HORIZON SECTIONS
def _horizon(story, ss, r, tmp, engine):
    ex = r["exp"]
    story.append(PageBreak())
    story.append(Paragraph(_safe(f"{ex} - {r['exp_name']} model"), ss["H1"]))
    story.append(Paragraph(_safe(r["window"]), ss["Small"]))
    story.append(Spacer(1, 6))

    c = r["confidence"]
    rc = colors.HexColor(r["risk_colour"])
    left = [Paragraph(f'<font color="{_hx(rc)}"><b>{_safe(r["verdict"])}</b> - '
                      f'{r["fail_prob"]*100:.1f}% fail / {r["pass_prob"]*100:.1f}% pass  '
                      f'(risk {_safe(r["risk_level"])})</font>', ss["Lead"]),
            Spacer(1, 3),
            Paragraph(_safe(f"Confidence {c['level']} ({c['score']*100:.0f}%). Likely range "
                            f"{c['interval'][0]*100:.0f}-{c['interval'][1]*100:.0f}%."), ss["Cell"]),
            Spacer(1, 3), Paragraph(_safe(c["note"]), ss["CellS"])]
    two = Table([[left, _img(_chart_conf(c, tmp, ex), max_w=72 * mm, max_h=40 * mm)]],
                colWidths=[CW - 76 * mm, 76 * mm])
    two.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                             ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(two)
    story.append(Spacer(1, 4))
    story.append(_panel([Paragraph(_safe(Engine.narrative(r)), ss["Cell"])]))

    # ---------------------------------------------------- SHAP explainability
    story.append(Paragraph("SHAP explainability", ss["H2"]))
    story.append(Paragraph(
        "Each bar is one answer pushing the prediction away from the cohort average. "
        "Red pushes toward failing, blue toward passing; length is the size of the push.",
        ss["Small"]))
    story.append(Spacer(1, 3))
    story.append(_img(_chart_waterfall(r, tmp), max_h=104 * mm))

    story.append(PageBreak())
    story.append(Paragraph(_safe(f"{ex} - top contributors"), ss["H2"]))
    img = _chart_contrib(r, tmp)
    if img:
        story.append(_img(img, max_h=88 * mm))
    if r["risk"]:
        top = r["risk"][0]
        story.append(Spacer(1, 3))
        story.append(Paragraph(
            _safe(f"Read it like this: {top[0]} contributed {top[1]:+.2f} toward predicting "
                  f"failure for this student."), ss["Small"]))

    # ------------------------------------------------- student profile analysis
    story.append(PageBreak())
    story.append(Paragraph(_safe(f"{ex} - student profile analysis"), ss["H2"]))
    story.append(Paragraph(
        "Status compares this student against students in the same source study who actually "
        "passed. Influence is how strongly the model used the answer for this student. The two "
        "can disagree: a good value can still be the model's main worry if everything else is "
        "stronger.", ss["Small"]))
    story.append(Spacer(1, 4))
    rows = [["Feature", "Value", "Ideal", "Status", "Influence", "Effect on this prediction"]]
    for f in r["factors"]:
        ecol = RISK if f["shap"] > .001 else OK if f["shap"] < -.001 else MUTED
        nm = _safe(f["label"])
        if not f["actionable"]:
            nm += ' <font size="6" color="#6b7280">(background)</font>'
        if f["leak"]:
            nm += ' <font size="6" color="#c0392b">(LEAK)</font>'
        rows.append([Paragraph(nm, ss["Cell"]),
                     Paragraph(_safe(f["display"]), ss["CellB"]),
                     Paragraph(_safe(f["ideal"]), ss["CellS"]),
                     _status_p(f["status"], ss),
                     Paragraph(_safe(f["influence"]), ss["Cell"]),
                     Paragraph(f'<font color="{_hx(ecol)}">{_safe(f["effect"])}</font>'
                               f'<font size="6" color="#6b7280"> ({f["shap"]:+.2f})</font>',
                               ss["Cell"])])
    story.append(_tbl(rows, [46 * mm, 20 * mm, 26 * mm, 24 * mm, 17 * mm,
                             CW - 133 * mm], ss))

    # ------------------------------------------------------ strengths/weakness
    def _bucket(title, items, colour, note, empty):
        story.append(Paragraph(f'<font color="{_hx(colour)}"><b>{title}</b></font>', ss["H3"]))
        if not items:
            story.append(Paragraph(_safe(empty), ss["Small"]))
            return
        story.append(Paragraph(_safe(note), ss["Small"]))
        for f in items[:7]:
            story.append(Paragraph(_safe(f"{f['label']} - {f['display']}. {f['comment']}."),
                                   ss["Tick"], bulletText="-"))

    story.append(Paragraph("Strengths", ss["H2"]))
    _bucket("Excellent areas", [f for f in r["strengths"] if f["status"] == "Excellent"], OK,
            "In the top quartile of students who passed.", "None on this horizon.")
    _bucket("Good areas", [f for f in r["strengths"] if f["status"] == "Good"], OK,
            "At or above the typical passing student.", "None on this horizon.")

    story.append(Paragraph("Weaknesses", ss["H2"]))
    _bucket("Critical risk factors", r["critical"], RISK,
            "Worse than three quarters of the students who failed. Address these first.",
            "No factor reached critical level.")
    _bucket("Needs improvement",
            [f for f in r["concerns"] if f["status"] == "Needs improvement"], WARN,
            "In the range where students in this dataset usually fail.", "None on this horizon.")
    _bucket("Borderline - watch", r["watch"], WARN,
            "Between the typical failing and typical passing student. These slide easily.",
            "None on this horizon.")
    if r["background"]:
        story.append(Spacer(1, 3))
        story.append(Paragraph(_safe(
            "Background factors raising this student's modelled risk: "
            + ", ".join(r["background"]) + ". The school cannot change these and they must "
            "never be used to judge the student or lower expectations - they indicate who "
            "needs more resourcing."), ss["Small"]))

    # ----------------------------------------------------------- recommendations
    story.append(PageBreak())
    story.append(Paragraph(_safe(f"{ex} - recommendations"), ss["H2"]))
    recs = r["recommendations"]
    if not recs:
        story.append(Paragraph(
            "No single controllable change moved this model's prediction. Escalate to a human "
            "case review rather than a one-off intervention.", ss["Cell"]))
    else:
        story.append(Paragraph(
            "Each row was produced by changing the value and re-running the model. Targets come "
            "from what passing students in the same source study actually look like. Apply in "
            "order - each figure already assumes the ones above it.", ss["Small"]))
        story.append(Spacer(1, 4))
        rows = [["#", "Change", "Now", "Target", "Failure risk", "Alone", "Status"]]
        for x in recs:
            rows.append([Paragraph(str(x["rank"]), ss["CellB"]),
                         Paragraph(_safe(x["label"]), ss["Cell"]),
                         Paragraph(_safe(x["current"]), ss["Cell"]),
                         Paragraph(_safe(x["target"]), ss["CellB"]),
                         Paragraph(_safe(f"{x['from_prob']*100:.0f}% to {x['to_prob']*100:.0f}%"),
                                   ss["Cell"]),
                         Paragraph(_safe(x["solo_text"]), ss["CellS"]),
                         _status_p(x["status"], ss)])
        story.append(_tbl(rows, [7 * mm, 42 * mm, 20 * mm, 22 * mm, 24 * mm, 25 * mm,
                                 CW - 140 * mm], ss))
        plan = r["what_if"].get("plan")
        if plan:
            story.append(Spacer(1, 5))
            story.append(_panel([Paragraph(_safe(
                f"Combined effect: doing all {len(plan['steps'])} takes the failure probability "
                f"from {plan['base_prob']*100:.0f}% to {plan['new_prob']*100:.0f}%."), ss["CellB"])],
                bg=colors.HexColor("#eaf4ec"), border=OK))
            im = _chart_plan(r, tmp)
            if im:
                story.append(Spacer(1, 5))
                story.append(_img(im, max_h=60 * mm))

    if r["immediate"]:
        story.append(Paragraph("Immediate actions", ss["H3"]))
        story.append(Paragraph("Things to start this week.", ss["Small"]))
        for a in r["immediate"]:
            story.append(Paragraph(_safe(a), ss["Tick"], bulletText="-"))

    if r["suggestions"]:
        story.append(Paragraph("Suggestions (longer term)", ss["H2"]))
        story.append(Paragraph(
            "Sustained changes that hold the improvement in place. These come from professional "
            "practice rather than the model.", ss["Small"]))
        for s in r["suggestions"]:
            story.append(Paragraph(_safe(s), ss["Tick"], bulletText="-"))

    # ------------------------------------------------------ intervention priority
    story.append(Paragraph("Intervention priority", ss["H2"]))
    now = [x["action"] for x in recs[:2]] or ["Case review with the year lead."]
    month = [x["action"] for x in recs[2:4]] or ["Re-run this report after the next assessment."]
    lterm = r["suggestions"][:3] or ["Maintain current support and monitor."]
    rows = [["When", "What"]]
    for when, items, col in [("Immediate\n(this week)", now, RISK),
                             ("Within one month", month, WARN),
                             ("Long term\n(this term and beyond)", lterm, OK)]:
        rows.append([Paragraph(f'<font color="{_hx(col)}"><b>{_safe(when)}</b></font>', ss["Cell"]),
                     Paragraph("<br/>".join("- " + _safe(i) for i in items), ss["Cell"])])
    story.append(_tbl(rows, [38 * mm, CW - 38 * mm], ss))


# ==================================================== 10. FINAL SUMMARY
def _final_summary(story, ss, results, ov, student_id):
    head = results[ov["exp"]]
    sm = Engine.audience_summaries(head, student_id)
    story.append(PageBreak())
    story.append(Paragraph("Final summary", ss["H1"]))
    story.append(Paragraph("The same result, written for three different readers.", ss["Small"]))
    story.append(Spacer(1, 6))
    for title, key, col in [("For the teacher", "teacher", ACCENT),
                            ("For the parent or guardian", "parent", OK),
                            ("For the student", "student", WARN)]:
        story.append(Paragraph(f'<font color="{_hx(col)}"><b>{title}</b></font>', ss["H3"]))
        story.append(_panel([Paragraph(_safe(sm[key]), ss["Cell"])],
                            bg=colors.HexColor("#f7f9fb"), border=col))
        story.append(Spacer(1, 6))


# ==================================================== APPENDIX A - glossary
def _glossary(story, ss, engine, results):
    if engine is None:
        return
    feats = []
    for r in sorted(results.values(), key=lambda r: EXP_ORDER.get(r["exp"], 9)):
        for f in r["factors"]:
            if f["feature"] not in feats:
                feats.append(f["feature"])
    if not feats:
        return
    story.append(PageBreak())
    story.append(Paragraph("Appendix A - what each question means", ss["H1"]))
    story.append(Paragraph(
        "Every value range below is computed from the training data itself, comparing students "
        "who passed with students who failed inside the same source study. They describe this "
        "dataset, not universal standards.", ss["Small"]))
    story.append(Spacer(1, 6))
    for f in feats:
        try:
            q = engine.question_help(f)
        except Exception:
            continue
        head = (_safe(q["label"]) + f'  <font size="7" color="#6b7280">({_safe(f)})</font>'
                + ("" if q["actionable"] else
                   ' <font size="7" color="#c0392b">background - not actionable</font>')
                + (' <font size="7" color="#c0392b">TARGET LEAK</font>' if q["leak"] else ""))
        block = [Paragraph(head, ss["H3"]),
                 Paragraph(_safe(q["description"]), ss["Cell"]), Spacer(1, 2),
                 Paragraph(f"<b>Possible values:</b> {_safe(q['range_text'])}", ss["Cell"])]
        if q["owners"]:
            block.append(Paragraph(_safe(
                f"Only recorded in the {', '.join(q['owners'])} study, so it is graded against "
                f"those {q['ref_n']:,} students."), ss["Small"]))
        if q["leak"]:
            block.append(Paragraph(f'<font color="#c0392b">{_safe(q["leak_reason"])}</font>',
                                   ss["Small"]))
        if q["contradicts"]:
            block.append(Paragraph(_safe(
                "In this data the relationship runs opposite to common sense, most likely because "
                "of confounding, so it is deliberately left ungraded rather than producing "
                "harmful advice."), ss["Small"]))
        if q["bands"]:
            rows = [["Value", "Interpretation", "Meaning in this cohort"]]
            for b in q["bands"]:
                rows.append([Paragraph(_safe(b["range"]), ss["CellB"]),
                             _status_p(b["status"], ss),
                             Paragraph(_safe(b["meaning"]), ss["CellS"])])
            block += [Spacer(1, 3), _tbl(rows, [42 * mm, 30 * mm, CW - 72 * mm], ss)]
        else:
            block.append(Paragraph(
                "This field does not separate passing from failing students in its own cohort, "
                "so no quality banding is given.", ss["Small"]))
        block.append(Spacer(1, 9))
        story.append(KeepTogether(block))


# ============================================= APPENDIX B - method & limits
def _method(story, ss, engine, results):
    story.append(PageBreak())
    story.append(Paragraph("Appendix B - method, data quality and limitations", ss["H1"]))

    def sec(h, paras, colour=None):
        story.append(Paragraph(
            f'<font color="{_hx(colour)}"><b>{h}</b></font>' if colour else f"<b>{h}</b>",
            ss["H3"]))
        for p in paras:
            story.append(Paragraph(_safe(p), ss["Cell"]))
            story.append(Spacer(1, 3))

    if engine is not None and getattr(engine, "owners", None):
        src = []
        if "source" in engine.raw.columns:
            vc = engine.raw["source"].value_counts()
            src = [f"{k} ({v:,} rows)" for k, v in vc.items()]
        sec("This dataset merges five studies", [
            "The training file stacks several different studies on top of each other: "
            + "; ".join(src) + ".",
            "Each study only fills in its own columns. Every other column holds a constant "
            "placeholder, which makes whole-file statistics misleading. Tutoring is the clearest "
            "example: across the whole file, students with tutoring appear to pass far less "
            "often, because 'no tutoring' silently includes thousands of placeholder rows from "
            "studies with very high pass rates. Compared only against students from the same "
            "study, tutoring is associated with a higher pass rate, which is what you would "
            "expect.",
            "Every benchmark, percentile, pass rate and recommendation target in this report is "
            "therefore computed inside the study that the column came from. The number of "
            "students used for each comparison is stated in Appendix A."], ACCENT)

    leaks = {}
    for r in results.values():
        for f in r["factors"]:
            if f["leak"]:
                leaks[f["label"]] = True
    if leaks:
        sec("Target leakage warning", [
            "The following features define the pass/fail outcome rather than predicting it: "
            + ", ".join(leaks) + ".",
            "A model containing them will look extremely accurate while learning nothing useful, "
            "and it cannot be used before results are known - which is the entire point of early "
            "warning. Retrain without them before drawing conclusions from the accuracy figures."],
            RISK)

    sec("The three horizons", [
        "E3 uses only enrolment and background data, so it is available on day one but is the "
        "least accurate. E2 adds attendance and engagement. E1 adds grades and is the most "
        "accurate but arrives too late to change much. Read them together: E3 says who to watch, "
        "E1 says what happened."])
    sec("Probability and risk level", [
        "The failure probability is the model's estimate for a student with this profile. 50% is "
        "the decision threshold, but it is a convention: a student at 48% is not meaningfully "
        "safer than one at 52%.",
        "Risk levels group the probability into bands - LOW under 20%, MODERATE 20-40%, HIGH "
        "40-70%, CRITICAL above 70% - so that action can be prioritised across a class."])
    sec("Confidence", [
        "A 0-100% score from four measurements. Decision margin: distance from the 50% threshold. "
        "Model stability: whether the prediction stayed put as trees were added, and whether any "
        "snapshot disagreed with the final verdict. Typicality: whether the answers fall inside "
        "the range the model was trained on. Completeness: how many inputs were supplied.",
        "Low confidence does not mean the prediction is wrong. It means it should not be acted "
        "on by itself."])
    sec("Status grades", [
        "Excellent, Good, Moderate, Needs improvement and Critical come from comparing the "
        "student against the real distribution of passing and failing students in the same "
        "source study. Neutral means the factor does not separate the two groups well enough to "
        "grade honestly. Background marks fields nobody can act on."])
    sec("Recommendations versus suggestions", [
        "Recommendations are counterfactual simulations: the value is changed, the model is "
        "re-run, and the resulting change in failure probability is reported. They are ordered "
        "greedily, so each figure already accounts for the ones above it.",
        "Suggestions are practical guidance on how to achieve those changes and come from "
        "professional practice, not from the model. The system is built so that these can be "
        "replaced by generated, personalised guidance later without changing any of the numbers."])
    sec("Limitations - read before acting", [
        "The model learns associations, not causes. A simulated drop in risk is what the model "
        "predicts for a student who already has that value; it is not a guarantee that changing "
        "it will produce that outcome.",
        "Some relationships are confounded. Tutoring and tutoring sessions are allocated to "
        "students who are already behind, so their raw association with failure reflects who "
        "receives them, not what they do. Where the data contradicts the obvious direction, this "
        "report leaves the factor ungraded rather than advising something harmful.",
        "Background factors such as gender, income and parental education may carry predictive "
        "weight because of inequities in the underlying data. They identify students who need "
        "more support. They must never be used to lower expectations of a student.",
        "Predictions describe a probability across similar students, not a destiny for this one. "
        "Share the reasoning with the student and let a human make the decision."], RISK)


# ---------------------------------------------------------------- entrypoint
def build_pdf(student_id, results: dict, engine=None, raw_answers=None) -> bytes:
    ss = _styles()
    tmp = tempfile.mkdtemp()
    buf = io.BytesIO()
    gen = datetime.now().strftime("%d %b %Y, %H:%M")

    doc = BaseDocTemplate(buf, pagesize=A4, leftMargin=LM, rightMargin=RM,
                          topMargin=19 * mm, bottomMargin=17 * mm,
                          title=f"Student Performance Report - {student_id}",
                          author="Student Performance Prediction")
    frame = Frame(LM, 17 * mm, CW, PAGE_H - 36 * mm, id="f")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                       onPage=lambda c, d: _decorate(c, d, student_id, gen))])

    story = []
    if not results:
        story += [Paragraph("Student Performance Report", ss["H1"]),
                  Paragraph("Not enough information was supplied to run any model. Answer at "
                            "least the very-early (E3) questions and generate the report again.",
                            ss["Lead"])]
        doc.build(story)
        return buf.getvalue()

    ov = Engine.overall(results)
    _cover(story, ss, results, ov, student_id, tmp, engine)
    _results_section(story, ss, results, ov, tmp)
    for r in sorted(results.values(), key=lambda r: EXP_ORDER.get(r["exp"], 9)):
        _horizon(story, ss, r, tmp, engine)
    _final_summary(story, ss, results, ov, student_id)
    _glossary(story, ss, engine, results)
    _method(story, ss, engine, results)
    doc.build(story)
    return buf.getvalue()
