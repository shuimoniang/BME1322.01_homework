import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assignment1" / "report.md"
OUTPUT = ROOT / "assignment1" / "report.pdf"

NAVY = colors.HexColor("#18324A")
BLUE = colors.HexColor("#2F6B9A")
INK = colors.HexColor("#24313B")
MUTED = colors.HexColor("#64727E")
GRID = colors.HexColor("#D9E1E7")
PALE = colors.HexColor("#F3F6F8")
ALT = colors.HexColor("#F7FAFC")


pdfmetrics.registerFont(TTFont("ReportCJK", r"C:\Windows\Fonts\msyh.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("ReportCJKBold", r"C:\Windows\Fonts\msyhbd.ttc", subfontIndex=0))


def styles():
    sheet = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=sheet["Title"],
            fontName="ReportCJKBold",
            fontSize=23,
            leading=31,
            textColor=colors.black,
            alignment=TA_LEFT,
            spaceAfter=10,
            keepWithNext=True,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=sheet["Normal"],
            fontName="ReportCJK",
            fontSize=10.5,
            leading=16,
            textColor=MUTED,
            spaceAfter=18,
            keepWithNext=True,
        ),
        "h1": ParagraphStyle(
            "ReportH1",
            parent=sheet["Heading1"],
            fontName="ReportCJKBold",
            fontSize=16,
            leading=22,
            textColor=colors.black,
            spaceBefore=15,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "ReportH2",
            parent=sheet["Heading2"],
            fontName="ReportCJKBold",
            fontSize=12.5,
            leading=18,
            textColor=colors.black,
            spaceBefore=11,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=sheet["BodyText"],
            fontName="ReportCJK",
            fontSize=10.2,
            leading=16.8,
            textColor=INK,
            alignment=TA_LEFT,
            firstLineIndent=2 * 10.2,
            spaceAfter=7,
            wordWrap="CJK",
        ),
        "list": ParagraphStyle(
            "ReportList",
            parent=sheet["BodyText"],
            fontName="ReportCJK",
            fontSize=10,
            leading=16,
            textColor=INK,
            leftIndent=0,
            firstLineIndent=0,
            wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "ReportTable",
            parent=sheet["BodyText"],
            fontName="ReportCJK",
            fontSize=8.1,
            leading=11.5,
            textColor=INK,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "ReportTableHeader",
            parent=sheet["BodyText"],
            fontName="ReportCJKBold",
            fontSize=8.1,
            leading=11.5,
            textColor=colors.white,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "ReportCode",
            parent=sheet["Code"],
            fontName="ReportCJK",
            fontSize=8,
            leading=12.2,
            textColor=colors.HexColor("#1E2A33"),
            leftIndent=0,
            rightIndent=0,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "ReportCaption",
            parent=sheet["BodyText"],
            fontName="ReportCJK",
            fontSize=8.5,
            leading=12,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=10,
        ),
    }


STYLES = styles()


def inline_markup(text):
    links = []

    def save_link(match):
        label = match.group(1)
        if label.startswith("`") and label.endswith("`"):
            label = label[1:-1]
        links.append(html.escape(label))
        return f"@@LINK{len(links) - 1}@@"

    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", save_link, text)
    codes = []

    def save_code(match):
        codes.append(html.escape(match.group(1)))
        return f"@@CODE{len(codes) - 1}@@"

    text = re.sub(r"`([^`]+)`", save_code, text)
    text = html.escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    for index, value in enumerate(codes):
        text = text.replace(
            f"@@CODE{index}@@",
            f'<font name="ReportCJK" color="#1D5678">{value}</font>',
        )
    for index, value in enumerate(links):
        text = text.replace(f"@@LINK{index}@@", value)
    return text


def split_cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def column_widths(count, available):
    presets = {
        2: [0.30, 0.70],
        3: [0.42, 0.29, 0.29],
        4: [0.43, 0.20, 0.18, 0.19],
        5: [0.31, 0.25, 0.16, 0.14, 0.14],
        6: [0.19, 0.11, 0.17, 0.17, 0.20, 0.16],
        7: [0.18, 0.09, 0.13, 0.11, 0.13, 0.19, 0.17],
    }
    fractions = presets.get(count, [1 / count] * count)
    return [available * value for value in fractions]


def make_table(rows, available):
    data = []
    for row_index, row in enumerate(rows):
        style = STYLES["table_header"] if row_index == 0 else STYLES["table"]
        data.append([Paragraph(inline_markup(cell), style) for cell in row])
    table = LongTable(
        data,
        colWidths=column_widths(len(rows[0]), available),
        repeatRows=1,
        hAlign="LEFT",
    )
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for row_index in range(1, len(rows)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), ALT))
    table.setStyle(TableStyle(commands))
    return table


def make_code(lines, available):
    rendered = "<br/>".join(html.escape(line).replace(" ", "&nbsp;") for line in lines)
    inner = Paragraph(rendered or " ", STYLES["code"])
    table = Table([[inner]], colWidths=[available])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.5, GRID),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def build_story(markdown, available):
    lines = markdown.splitlines()
    story = []
    figure_number = 0
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            index += 1
            continue
        if line.startswith("# "):
            story.append(Spacer(1, 12 * mm))
            story.append(Paragraph(inline_markup(line[2:]), STYLES["title"]))
            index += 1
            continue
        if line.startswith("> "):
            story.append(Paragraph(inline_markup(line[2:]), STYLES["subtitle"]))
            index += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:]), STYLES["h1"]))
            index += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:]), STYLES["h2"]))
            index += 1
            continue
        if line.startswith("```"):
            index += 1
            code_lines = []
            while index < len(lines) and not lines[index].startswith("```"):
                code_lines.append(lines[index].rstrip())
                index += 1
            index += 1
            story.extend([make_code(code_lines, available), Spacer(1, 7)])
            continue
        image_match = re.match(r"!\[([^\]]+)\]\(([^\)]+)\)", line)
        if image_match:
            figure_number += 1
            image_path = SOURCE.parent / image_match.group(2)
            graphic = Image(str(image_path))
            scale = min(available / graphic.imageWidth, 82 * mm / graphic.imageHeight)
            graphic.drawWidth = graphic.imageWidth * scale
            graphic.drawHeight = graphic.imageHeight * scale
            graphic.hAlign = "CENTER"
            caption = Paragraph(
                f"图 {figure_number} {inline_markup(image_match.group(1))}",
                STYLES["caption"],
            )
            story.append(KeepTogether([graphic, caption]))
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and re.match(r"^\|[\s:\-|]+\|$", lines[index + 1]):
            rows = [split_cells(line)]
            index += 2
            while index < len(lines) and lines[index].startswith("|"):
                rows.append(split_cells(lines[index]))
                index += 1
            story.extend([make_table(rows, available), Spacer(1, 8)])
            continue
        if re.match(r"^- ", line):
            items = []
            while index < len(lines) and re.match(r"^- ", lines[index]):
                items.append(
                    ListItem(
                        Paragraph(inline_markup(lines[index][2:]), STYLES["list"]),
                        leftIndent=12,
                    )
                )
                index += 1
            story.append(
                ListFlowable(items, bulletType="bullet", leftIndent=18, bulletFontName="ReportCJK")
            )
            story.append(Spacer(1, 5))
            continue
        if re.match(r"^\d+\. ", line):
            items = []
            while index < len(lines) and re.match(r"^\d+\. ", lines[index]):
                text = re.sub(r"^\d+\. ", "", lines[index])
                items.append(ListItem(Paragraph(inline_markup(text), STYLES["list"]), leftIndent=14))
                index += 1
            story.append(
                ListFlowable(
                    items,
                    bulletType="1",
                    start="1",
                    leftIndent=20,
                    bulletFontName="ReportCJK",
                )
            )
            story.append(Spacer(1, 5))
            continue

        paragraph_lines = [line]
        index += 1
        while index < len(lines):
            candidate = lines[index].rstrip()
            if not candidate:
                break
            if (
                candidate.startswith(("#", ">", "```", "|", "![", "- "))
                or re.match(r"^\d+\. ", candidate)
            ):
                break
            paragraph_lines.append(candidate)
            index += 1
        story.append(Paragraph(inline_markup(" ".join(paragraph_lines)), STYLES["body"]))
    return story


def page_decor(canvas, doc):
    canvas.saveState()
    width, height = A4
    page = canvas.getPageNumber()
    if page > 1:
        canvas.setFont("ReportCJK", 8.2)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, height - 18 * mm, "广义五子棋与对抗搜索项目报告")
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, height - 20 * mm, width - doc.rightMargin, height - 20 * mm)
    canvas.setFont("ReportCJK", 8.2)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(width / 2, 12 * mm, str(page))
    canvas.restoreState()


def main():
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=24 * mm,
        bottomMargin=20 * mm,
        title="广义五子棋与对抗搜索项目报告",
        author="",
        subject="BME1322.01 Artificial Intelligence Assignment 1",
    )
    available = A4[0] - doc.leftMargin - doc.rightMargin
    story = build_story(SOURCE.read_text(encoding="utf-8"), available)
    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    print(OUTPUT)


if __name__ == "__main__":
    main()
