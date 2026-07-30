import os
import re
from datetime import datetime
from functools import partial
from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)


# ============================================================== fonts
# Modern sans-serif (Lato) instead of the default PDF-standard Helvetica.
# Font files must sit in a "fonts/" folder next to this script. If they're
# missing for any reason, this quietly falls back to Helvetica so the PDF
# still builds.
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def _register_fonts():
    try:
        pdfmetrics.registerFont(TTFont("Lato", os.path.join(FONT_DIR, "Lato-Regular.ttf")))
        pdfmetrics.registerFont(TTFont("Lato-Bold", os.path.join(FONT_DIR, "Lato-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("Lato-Italic", os.path.join(FONT_DIR, "Lato-Italic.ttf")))
        pdfmetrics.registerFontFamily(
            "Lato", normal="Lato", bold="Lato-Bold",
            italic="Lato-Italic", boldItalic="Lato-Bold",
        )
        return True
    except Exception:
        return False


_FONTS_OK = _register_fonts()
FONT_REGULAR = "Lato" if _FONTS_OK else "Helvetica"
FONT_BOLD = "Lato-Bold" if _FONTS_OK else "Helvetica-Bold"
FONT_ITALIC = "Lato-Italic" if _FONTS_OK else "Helvetica-Oblique"


# ============================================================== palette
NAVY = HexColor("#0B2545")
BANNER_SUB = HexColor("#AFC2DE")
INK = HexColor("#1F2937")
MUTED = HexColor("#6B7280")
CARD_BG = HexColor("#F7F9FC")
CARD_BORDER = HexColor("#E2E8F0")
BAR_TRACK = HexColor("#E5E9F0")
BLUE = HexColor("#2563EB")

RISK_COLORS = {
    "AT RISK": HexColor("#DC2626"),
    "BORDERLINE": HexColor("#D97706"),
    "ON TRACK": HexColor("#16A34A"),
    "SAFE": HexColor("#16A34A"),
    "PASS": HexColor("#16A34A"),
}

# full text width = page width (8.5in) minus 0.75in margins on each side
CONTENT_WIDTH = 7.0 * inch

SECTION_THEMES = [
    (("strength",), HexColor("#16A34A")),
    (("attention", "concern", "weakness"), HexColor("#DC2626")),
    (("immediate",), HexColor("#D97706")),
    (("long-term", "recommend"), BLUE),
    (("key finding",), BLUE),
    (("teacher",), HexColor("#7C3AED")),
    (("parent",), HexColor("#0891B2")),
    (("student summary",), HexColor("#059669")),
    (("final note",), MUTED),
    (("executive summary",), NAVY),
]


def _risk_color(prediction: str) -> HexColor:
    p = (prediction or "").upper()
    for key, color in RISK_COLORS.items():
        if key in p:
            return color
    return BLUE


def _section_color(heading: str) -> HexColor:
    h = (heading or "").lower()
    for keys, color in SECTION_THEMES:
        if any(k in h for k in keys):
            return color
    return NAVY


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))


def _inline_markdown(text: str) -> str:
    """Escape XML first, then translate **bold** / *italic* into reportlab tags."""
    text = _escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)([^*]+?)\*(?!\*)", r"<i>\1</i>", text)
    return text


class AIReportPDF:

    def __init__(self):
        base = getSampleStyleSheet()

        self.section_title_style = ParagraphStyle(
            "SectionTitle", parent=base["Heading1"],
            fontName=FONT_BOLD, fontSize=15.5,
            textColor=HexColor("#FFFFFF"), spaceAfter=0, spaceBefore=0,
            leading=18,
        )
        self.body = ParagraphStyle(
            "Body", parent=base["BodyText"],
            fontName=FONT_REGULAR, fontSize=12.5, leading=18.5,
            textColor=INK, spaceAfter=6,
        )
        self.bullet = ParagraphStyle(
            "Bullet", parent=self.body,
            leftIndent=17, firstLineIndent=-17, spaceAfter=6,
        )
        self.label_style = ParagraphStyle(
            "Label", parent=base["Normal"], fontName=FONT_BOLD,
            fontSize=10.5, textColor=MUTED, leading=13,
        )
        self.value_style = ParagraphStyle(
            "Value", parent=base["Normal"], fontName=FONT_BOLD,
            fontSize=18.5, textColor=NAVY, leading=21,
        )
        self.chip_style = ParagraphStyle(
            "Chip", parent=base["Normal"], fontName=FONT_BOLD,
            fontSize=15, textColor=HexColor("#FFFFFF"), alignment=TA_CENTER,
            leading=17,
        )
        self.footer_note = ParagraphStyle(
            "FooterNote", parent=base["Normal"], fontName=FONT_ITALIC,
            fontSize=10.5, textColor=MUTED,
        )

    # ---------------------------------------------------------- header/footer
    def _draw_header_footer(self, canv, doc, meta):
        width, height = letter
        canv.saveState()

        # top banner
        canv.setFillColor(NAVY)
        canv.rect(0, height - 0.95 * inch, width, 0.95 * inch, stroke=0, fill=1)
        canv.setFillColor(HexColor("#FFFFFF"))
        canv.setFont(FONT_BOLD, 18)
        canv.drawString(0.75 * inch, height - 0.5 * inch, "AI Student Performance Report")
        canv.setFillColor(BANNER_SUB)
        canv.setFont(FONT_REGULAR, 10.5)
        canv.drawString(0.75 * inch, height - 0.72 * inch,
                         "Confidential  ·  generated to support educator, parent and student decisions")
        canv.setFont(FONT_REGULAR, 10.5)
        canv.drawRightString(width - 0.75 * inch, height - 0.5 * inch, meta["date"])
        canv.setFillColor(meta["risk_color"])
        canv.rect(0, height - 0.98 * inch, width, 0.06 * inch, stroke=0, fill=1)

        # footer
        canv.setStrokeColor(CARD_BORDER)
        canv.setLineWidth(0.6)
        canv.line(0.75 * inch, 0.62 * inch, width - 0.75 * inch, 0.62 * inch)
        canv.setFillColor(MUTED)
        canv.setFont(FONT_ITALIC, 10)
        canv.drawString(0.75 * inch, 0.45 * inch,
                         "AI-generated to inform, not replace, teacher judgment.")
        canv.setFont(FONT_REGULAR, 10)
        canv.drawRightString(width - 0.75 * inch, 0.45 * inch, f"Page {doc.page}")
        canv.restoreState()

    # ---------------------------------------------------------- overview card
    def _overview_card(self, student_name, prediction, confidence, risk_color):
        bar_w = 1.7 * inch
        filled = max(0.02, min(1.0, confidence / 100.0)) * bar_w
        remaining = bar_w - filled

        bar = Table(
            [[""]] if filled <= 0 else [["", ""]],
            colWidths=[filled, remaining] if filled > 0 else [bar_w],
            rowHeights=[7],
            hAlign="LEFT",
        )
        bar_cmds = [
            ("BACKGROUND", (0, 0), (0, 0), risk_color),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]
        if filled > 0 and remaining > 0:
            bar_cmds.append(("BACKGROUND", (1, 0), (1, 0), BAR_TRACK))
        bar.setStyle(TableStyle(bar_cmds))

        student_cell = [
            Paragraph("STUDENT", self.label_style),
            Spacer(1, 3),
            Paragraph(_inline_markdown(student_name), self.value_style),
        ]

        chip = Table([[Paragraph(_inline_markdown(prediction.upper()), self.chip_style)]],
                     colWidths=[1.9 * inch], rowHeights=[0.34 * inch], hAlign="LEFT")
        chip.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), risk_color),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        prediction_cell = [
            Paragraph("PREDICTION", self.label_style),
            Spacer(1, 3),
            chip,
        ]

        confidence_cell = [
            Paragraph("PREDICTION CONFIDENCE", self.label_style),
            Spacer(1, 3),
            Paragraph(f"{confidence:.1f}%", self.value_style),
            Spacer(1, 3),
            bar,
        ]

        col_w = CONTENT_WIDTH / 3
        card = Table(
            [[student_cell, prediction_cell, confidence_cell]],
            colWidths=[col_w, col_w, col_w],
            hAlign="LEFT",
        )
        card.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
            ("BOX", (0, 0), (-1, -1), 0.75, CARD_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.75, CARD_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ("LEFTPADDING", (0, 0), (-1, -1), 16),
            ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ]))
        return card

    # ---------------------------------------------------------- section card
    def _section_bar(self, heading, color):
        bar = Table([[Paragraph(_inline_markdown(heading), self.section_title_style)]],
                    colWidths=[CONTENT_WIDTH], hAlign="LEFT")
        bar.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), color),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 14),
            ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ]))
        return bar

    # ---------------------------------------------------------- markdown parse
    def markdown_to_story(self, markdown):
        story = []
        current_color = NAVY
        pending = []  # paragraphs belonging to the current section (for KeepTogether on the heading)

        def flush_pending():
            if pending:
                story.extend(pending)
                pending.clear()

        for raw_line in markdown.splitlines():
            line = raw_line.strip()

            if not line:
                pending.append(Spacer(1, 0.08 * inch))
                continue

            # Horizontal rule (---, ***, ___, or any run of 3+ of these chars)
            if len(line) >= 3 and len(set(line.replace(" ", ""))) == 1 and line[0] in "-*_":
                pending.append(Spacer(1, 0.06 * inch))
                pending.append(HRFlowable(width="100%", thickness=0.6, color=CARD_BORDER))
                pending.append(Spacer(1, 0.1 * inch))
                continue

            # Heading 1 / Heading 2 -> colored section bar
            if line.startswith("# ") or line.startswith("## "):
                flush_pending()
                heading_text = line[2:].strip() if line.startswith("# ") else line[3:].strip()
                current_color = _section_color(heading_text)
                story.append(Spacer(1, 0.24 * inch))
                story.append(KeepTogether([
                    self._section_bar(heading_text, current_color),
                    Spacer(1, 0.14 * inch),
                ]))
                continue

            # Bullet point
            if line.startswith("- ") or line.startswith("* "):
                content = _inline_markdown(line[2:].strip())
                bullet_html = (f'<font color="#{current_color.hexval()[2:]}">&#8226;</font>'
                               f'&nbsp;&nbsp;{content}')
                pending.append(Paragraph(bullet_html, self.bullet))
                continue

            # Regular paragraph
            pending.append(Paragraph(_inline_markdown(line), self.body))

        flush_pending()
        return story

    # ---------------------------------------------------------------- build
    def build(self, student_name, prediction, confidence, markdown_report):
        buffer = BytesIO()
        risk_color = _risk_color(prediction)

        doc = SimpleDocTemplate(
            buffer, pagesize=letter,
            topMargin=1.35 * inch, bottomMargin=0.85 * inch,
            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        )

        story = [
            self._overview_card(student_name, prediction, confidence, risk_color),
            Spacer(1, 0.05 * inch),
            HRFlowable(width="100%", thickness=0.75, color=CARD_BORDER,
                      spaceBefore=10, spaceAfter=2),
        ]
        story.extend(self.markdown_to_story(markdown_report))

        meta = {
            "date": datetime.now().strftime("%d %b %Y"),
            "risk_color": risk_color,
        }
        page_fn = partial(self._draw_header_footer, meta=meta)
        doc.build(story, onFirstPage=page_fn, onLaterPages=page_fn)

        pdf = buffer.getvalue()
        buffer.close()
        return pdf