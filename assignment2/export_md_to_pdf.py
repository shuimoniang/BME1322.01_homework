r"""Export a Markdown report to PDF (self-contained, no network access).

Pipeline: report.md -> HTML -> Edge headless print-to-pdf.

Design notes:
  * LaTeX maths ($...$ and $$...$$) is rendered locally with matplotlib's
    mathtext and embedded as base64 PNG images, so no TeX/KaTeX/CDN is needed.
    \text{...} is rewritten to \mathrm{...} before rendering (mathtext has no
    \text).
  * Images referenced by the Markdown are inlined as base64 data URIs, which
    avoids file:// path issues (the repo path contains non-ASCII characters).
  * The Markdown subset used by the reports (headings, paragraphs, tables,
    images, fenced code, lists, bold, inline code, links) is parsed by the
    built-in converter in this file; no third-party Markdown package is
    required.
  * Microsoft Edge is used only as a local HTML printer; if it is running,
    a throwaway user-data dir keeps the headless invocation independent.

Usage:
    python export_md_to_pdf.py                 # report.md -> report.pdf
    python export_md_to_pdf.py --input a/report.md --output b/report.pdf
"""

from __future__ import annotations

import argparse
import base64
import io
import re
import struct
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EDGE_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)


def find_edge() -> Path:
    for candidate in EDGE_CANDIDATES:
        if Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("Microsoft Edge not found in standard install locations")


def render_math_png(tex: str, *, display: bool) -> tuple[str, int, int]:
    r"""Render one LaTeX snippet with mathtext; return (base64 png, w, h).

    matplotlib draws multi-line maths as plain text (silently), so line
    breaks are folded into spaces first -- LaTeX treats them as whitespace
    anyway.  \Big/\big sizing commands are unsupported by mathtext and are
    dropped; a parse failure aborts the export instead of emitting a
    source-code image.
    """
    tex = re.sub(r"\\[Bb]igg*", "", tex)
    tex = re.sub(r"\s+", " ", tex).strip()
    tex = tex.replace(r"\text{", r"\mathrm{")
    fontsize = 11.5 if display else 10.5
    figure = plt.figure(figsize=(0.01, 0.01))
    figure.text(0, 0, f"${tex}$", fontsize=fontsize)
    buffer = io.BytesIO()
    try:
        figure.savefig(
            buffer,
            format="png",
            dpi=220,
            bbox_inches="tight",
            pad_inches=0.02,
            transparent=True,
        )
    except ValueError as error:
        plt.close(figure)
        raise SystemExit(f"mathtext could not parse: {tex[:100]}\n{error}")
    plt.close(figure)
    png = buffer.getvalue()
    width, height = struct.unpack(">II", png[16:24])
    return base64.b64encode(png).decode("ascii"), width, height


def extract_and_render_math(markdown_text: str) -> tuple[str, dict[str, str]]:
    """Replace maths with ASCII placeholders and render each one.

    Display maths are processed first so that inline $...$ matching cannot
    split a $$...$$ block.  Placeholders are pure alphanumerics, which the
    Markdown converter passes through untouched.
    """
    replacements: dict[str, str] = {}
    counter = 0

    def store(tex: str, *, display: bool) -> str:
        nonlocal counter
        key = f"ZZMATHREF{counter}ZZ"
        counter += 1
        encoded, width, height = render_math_png(tex, display=display)
        css_width = width * 96 / 220  # rendered at 220 dpi, CSS px are 96 dpi
        style = f' style="width: {css_width:.1f}px;"'
        replacements[key] = (
            f'<img class="{"display-math" if display else "inline-math"}"{style} '
            f'src="data:image/png;base64,{encoded}">'
        )
        return key

    def display_sub(match: re.Match) -> str:
        return store(match.group(1).strip(), display=True)

    def inline_sub(match: re.Match) -> str:
        return store(match.group(1).strip(), display=False)

    text = re.sub(r"\$\$(.+?)\$\$", display_sub, markdown_text, flags=re.DOTALL)
    text = re.sub(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", inline_sub, text)
    return text, replacements


def inline_images(html: str, base_dir: Path) -> str:
    """Replace local image references with base64 data URIs."""

    def replace(match: re.Match) -> str:
        source = base_dir / match.group(1)
        encoded = base64.b64encode(source.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{encoded}"'

    return re.sub(r'src="([^"]+\.png)"', replace, html)


def mark_captions(html: str) -> str:
    """Style '图 N　…' / '表 N　…' paragraphs as centred captions."""
    html = re.sub(r"<p>(图 \d+　)", r'<p class="caption">\1', html)
    return re.sub(r"<p>(表 \d+　)", r'<p class="caption">\1', html)


CSS = """
@page { size: A4; margin: 20mm 18mm; }
* { box-sizing: border-box; }
body {
  font-family: "SimSun", "Songti SC", serif;
  font-size: 10.5pt;
  line-height: 1.7;
  color: #1a1a1a;
  margin: 0;
}
h1 {
  font-family: "Microsoft YaHei", "SimHei", sans-serif;
  font-size: 17pt;
  text-align: center;
  margin: 0.2em 0 1.2em 0;
  color: #111;
}
h2 {
  font-family: "Microsoft YaHei", "SimHei", sans-serif;
  font-size: 13.5pt;
  color: #861720;
  border-bottom: 1.2pt solid #861720;
  padding-bottom: 3px;
  margin: 1.5em 0 0.6em 0;
  page-break-after: avoid;
}
h3 {
  font-family: "Microsoft YaHei", "SimHei", sans-serif;
  font-size: 11.5pt;
  margin: 1.2em 0 0.4em 0;
  page-break-after: avoid;
}
p { margin: 0.45em 0; text-align: justify; }
table {
  border-collapse: collapse;
  margin: 0.7em auto;
  font-size: 9.5pt;
  page-break-inside: avoid;
}
th, td { border: 0.5pt solid #8a8a8a; padding: 3px 8px; vertical-align: top; }
th { background: #f0efed; font-family: "Microsoft YaHei", sans-serif; font-weight: bold; }
tr { page-break-inside: avoid; }
img { max-width: 100%; page-break-inside: avoid; }
p > img:only-child:not(.inline-math) { display: block; margin: 0.7em auto; }
.display-math { display: block; margin: 0.55em auto; }
.inline-math { vertical-align: -0.16em; margin: 0 1px; }
.caption {
  text-align: center;
  font-size: 9pt;
  color: #444;
  margin: 0.1em 0 0.9em 0;
  font-family: "Microsoft YaHei", sans-serif;
}
code {
  font-family: Consolas, "Courier New", monospace;
  font-size: 9.5pt;
  background: #f4f4f4;
  padding: 1px 4px;
  border-radius: 3px;
}
pre {
  background: #f7f7f7;
  border: 0.5pt solid #ddd;
  padding: 7px 10px;
  page-break-inside: avoid;
}
pre code { background: none; padding: 0; }
ul, ol { margin: 0.4em 0; padding-left: 2em; }
li { margin: 0.2em 0; text-align: justify; }
a { color: #0d3b66; text-decoration: none; }
strong { font-weight: bold; }
"""

HTML_SHELL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_inline(text: str) -> str:
    """Inline Markdown: escape, then `code`, **bold**, [link](url)."""
    text = html_escape(text)
    text = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)\s]+)\)",
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        text,
    )
    return text


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table_alignment(line: str) -> list[str]:
    aligns = []
    for cell in _split_table_row(line):
        left, right = cell.startswith(":"), cell.endswith(":")
        aligns.append("center" if left and right else "right" if right else "left")
    return aligns


_TABLE_DIVIDER = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_TABLE_ROW = re.compile(r"^\s*\|.+\|\s*$")
_FENCE = re.compile(r"^```")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_ITEM = re.compile(r"^\s*-\s+(.*)$")
_NUMBERED_ITEM = re.compile(r"^\s*\d+\.\s+(.*)$")
_IMAGE_LINE = re.compile(r"^\s*!\[(.*?)\]\((.+?)\)\s*$")
_HR = re.compile(r"^\s*(-{3,}|\*{3,})\s*$")


def _is_block_start(line: str) -> bool:
    return bool(
        _HEADING.match(line)
        or _TABLE_ROW.match(line)
        or _FENCE.match(line)
        or _LIST_ITEM.match(line)
        or _NUMBERED_ITEM.match(line)
        or _IMAGE_LINE.match(line)
        or _HR.match(line)
        or not line.strip()
    )


def render_block(line: str, lines: list[str], index: int) -> tuple[str, int]:
    """Convert one block starting at lines[index]; return (html, next index)."""
    fence = _FENCE.match(line)
    if fence:
        body: list[str] = []
        index += 1
        while index < len(lines) and not _FENCE.match(lines[index]):
            body.append(lines[index])
            index += 1
        index += 1
        return f"<pre><code>{html_escape(chr(10).join(body))}</code></pre>", index

    if _TABLE_ROW.match(line) and index + 1 < len(lines) and _TABLE_DIVIDER.match(lines[index + 1]):
        header = _split_table_row(line)
        aligns = _table_alignment(lines[index + 1])
        index += 2
        body_rows = []
        while index < len(lines) and _TABLE_ROW.match(lines[index]):
            body_rows.append(_split_table_row(lines[index]))
            index += 1
        align_style = [
            f" text-align: {aligns[column]};"
            if column < len(aligns) else ""
            for column in range(max([len(header)] + [len(row) for row in body_rows]))
        ]
        head_cells = "".join(
            f'<th style="{align_style[column]}">{render_inline(cell)}</th>'
            for column, cell in enumerate(header)
        )
        body_cells = "".join(
            "<tr>"
            + "".join(
                f'<td style="{align_style[column]}">{render_inline(cell)}</td>'
                for column, cell in enumerate(row)
            )
            + "</tr>"
            for row in body_rows
        )
        return f"<table><thead><tr>{head_cells}</tr></thead><tbody>{body_cells}</tbody></table>", index

    heading = _HEADING.match(line)
    if heading:
        level = len(heading.group(1))
        return f"<h{level}>{render_inline(heading.group(2))}</h{level}>", index + 1

    image = _IMAGE_LINE.match(line)
    if image:
        source = html_escape(image.group(2))
        return f'<p><img src="{source}" alt="{html_escape(image.group(1))}"></p>', index + 1

    if _HR.match(line):
        return "<hr>", index + 1

    list_items: list[str] | None = None
    ordered = bool(_NUMBERED_ITEM.match(line))
    if ordered or _LIST_ITEM.match(line):
        pattern = _NUMBERED_ITEM if ordered else _LIST_ITEM
        list_items = []
        while index < len(lines):
            match = pattern.match(lines[index])
            if not match:
                break
            list_items.append(match.group(1))
            index += 1
    if list_items is not None:
        tag = "ol" if ordered else "ul"
        items = "".join(f"<li>{render_inline(item)}</li>" for item in list_items)
        return f"<{tag}>{items}</{tag}>", index

    paragraph = [line]
    index += 1
    while index < len(lines) and not _is_block_start(lines[index]):
        paragraph.append(lines[index])
        index += 1
    return f"<p>{render_inline(' '.join(part.strip() for part in paragraph))}</p>", index


def markdown_to_html(markdown_text: str) -> str:
    """Convert the report's Markdown subset to HTML (no third-party deps)."""
    lines = markdown_text.split("\n")
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        block, index = render_block(lines[index], lines, index)
        blocks.append(block)
    return "\n".join(blocks)


def export(input_path: Path, output_path: Path) -> None:
    markdown_text = input_path.read_text(encoding="utf-8")
    markdown_text, math_replacements = extract_and_render_math(markdown_text)

    html_body = markdown_to_html(markdown_text)
    for key, image_tag in math_replacements.items():
        html_body = html_body.replace(key, image_tag)
    html_body = inline_images(html_body, input_path.parent)
    html_body = mark_captions(html_body)

    html_path = output_path.with_suffix(".html")
    html_path.write_text(
        HTML_SHELL.format(css=CSS, body=html_body), encoding="utf-8"
    )

    with tempfile.TemporaryDirectory() as profile_dir:
        command = [
            str(find_edge()),
            "--headless",
            "--disable-gpu",
            "--no-first-run",
            f"--user-data-dir={profile_dir}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={output_path}",
            html_path.resolve().as_uri(),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if not output_path.is_file():
        raise SystemExit(f"Edge did not produce a PDF.\nstdout: {result.stdout}\nstderr: {result.stderr}")
    html_path.unlink()
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="report.md", help="Markdown source file")
    parser.add_argument("--output", default="report.pdf", help="PDF output file")
    arguments = parser.parse_args()
    input_path = Path(arguments.input).resolve()
    output_path = Path(arguments.output).resolve()
    export(input_path, output_path)


if __name__ == "__main__":
    main()
