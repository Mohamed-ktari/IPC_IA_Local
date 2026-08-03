# docx_writer.py
# Builds the "mémoire technique" skeleton Word document from the company's
# template, using the structure RCAgent extracted from the RC (arbitrary
# number of sections/subsections — not the template's original fixed
# 7-chapter list).
#
# Key design point: we EDIT the existing template file (python-docx opens
# it, we delete its old fixed chapters, then append the new ones) — we do
# NOT build a document from scratch. This means the logo, title style,
# fonts, and the "Titre"/"Titre1".."Titre9" style definitions all carry
# over automatically. No separate logo image needs to be supplied unless
# the company wants to swap it for a different one later (in which case:
# replace the embedded image inside the template file itself, not here).

import re
from pathlib import Path

from docx import Document

STRUCTURE_LINE_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(.+?)\s*$")

# The template defines Titre1..Titre9 (python-docx sees these as
# "Heading 1".."Heading 9") — cap at 9 so we never request a style that
# doesn't exist in the template.
MAX_HEADING_LEVEL = 9


def parse_structure_text(structure_text: str) -> list[dict]:
    """
    Parses the numbered plan produced by RCAgent's structure-extraction step,
    e.g.:
        1. Introduction
           1.1 Objectifs du projet
        2. Méthodologie
           2.1 Étude préalable
           2.2 Choix techniques
        ...
        8. Annexes

    into a flat list of {"level": int, "title": str}.

    Level is derived from the DOTTED NUMBER depth ("1" -> 1, "1.1" -> 2,
    "1.1.1" -> 3, ...), not from leading whitespace/indentation — LLM output
    can't be trusted to indent consistently, but it reliably keeps the
    numbering scheme intact since that's what the structure prompt asked for.
    """
    nodes = []
    for line in structure_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = STRUCTURE_LINE_RE.match(line)
        if not m:
            continue  # skip stray prose lines (e.g. an LLM preamble sentence)
        numbering, title = m.groups()
        level = min(numbering.count(".") + 1, MAX_HEADING_LEVEL)
        nodes.append({"level": level, "title": title.strip()})
    return nodes


def generate_memoire_docx(
    template_path: str | Path,
    structure_text: str,
    output_path: str | Path,
    keep_leading_paragraphs: int = 3,
) -> Path:
    """
    template_path: the company's .docx template (title + date + logo +
        fixed chapters — the fixed chapters get replaced).
    structure_text: RCAgent's extracted structure (RCAgentResponse.structure
        ["structure"]).
    output_path: where to save the generated .docx.
    keep_leading_paragraphs: how many paragraphs at the top of the template
        to preserve as-is before the old fixed chapters start. In the
        current template this is 3: the "Titre" paragraph, the date line,
        and the paragraph holding the embedded logo image. If the company
        changes the template's front matter, this number needs updating.
    """
    template_path = Path(template_path)
    output_path = Path(output_path)

    doc = Document(template_path)

    # python-docx has no paragraph.delete() — remove the underlying XML
    # element directly to drop the template's old fixed chapters.
    for p in doc.paragraphs[keep_leading_paragraphs:]:
        p._element.getparent().remove(p._element)

    nodes = parse_structure_text(structure_text)
    if not nodes:
        raise ValueError(
            "No headings parsed from structure_text — check it matches the "
            "expected numbered format (e.g. '1. Titre', '1.1 Sous-titre')."
        )

    for node in nodes:
        doc.add_heading(node["title"], level=node["level"])
        doc.add_paragraph("")  # blank line — human fills in the content later

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    return output_path




    # --- appended to docx_writer.py ---
# Fills an EXISTING structure doc's sections with generated content, as
# opposed to generate_memoire_docx() which builds the empty skeleton from
# scratch. Used by generation_agent, never by rc_agent.
#
# Matching is done by (level, title, occurrence_index) rather than pure
# title string, so two sections that happen to share a title (rare, but
# possible after user edits) each still get their own generated content
# rather than the first match winning twice.

def fill_memoire_docx(
    structure_docx_path: str | Path,
    filled_sections: list[dict],
    output_path: str | Path,
) -> Path:
    structure_docx_path = Path(structure_docx_path)
    output_path = Path(output_path)

    doc = Document(structure_docx_path)

    # Cache paragraphs and heading positions ONCE instead of re-querying
    # doc.paragraphs repeatedly inside the loop — doc.paragraphs rebuilds
    # from the XML tree on every access, which gets progressively slower
    # as more paragraphs get inserted per section.
    all_paragraphs = list(doc.paragraphs)
    heading_positions = [
        idx for idx, p in enumerate(all_paragraphs)
        if _heading_level_for_writer(p) is not None
    ]

    if len(heading_positions) != len(filled_sections):
        raise ValueError(
            f"Structure mismatch: found {len(heading_positions)} headings "
            f"in the document but received {len(filled_sections)} filled "
            f"sections. The document may have been edited between parsing "
            f"and filling — re-parse before filling."
        )

    for pos, section in zip(reversed(heading_positions), reversed(filled_sections)):
        heading_para = all_paragraphs[pos]

        next_pos = next((p for p in heading_positions if p > pos), None)
        end = next_pos if next_pos is not None else len(all_paragraphs)

        for p in all_paragraphs[pos + 1:end]:
            p._element.getparent().remove(p._element)

        _insert_markdown_content(heading_para, section["content"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    return output_path

def _heading_level_for_writer(paragraph) -> int | None:
    # Same detection logic as docx_parser._heading_level — duplicated
    # (not imported) to keep docx_writer.py's existing zero-dependency-
    # on-docx_parser property intact, since rc_agent must never be
    # affected by anything generation_agent needs.
    import re
    style_name = paragraph.style.name if paragraph.style else ""
    m = re.match(r"^Heading (\d+)$", style_name)
    if m:
        return int(m.group(1))
    m2 = re.match(r"^Titre\s*(\d+)$", style_name, re.IGNORECASE)
    if m2:
        return int(m2.group(1))
    return None

# --- addition to docx_writer.py ---
# Converts Markdown-ish LLM output into real docx formatting instead of
# dumping raw '**'/'-'/'#' characters into a plain paragraph. Applied only
# at write time (fill_memoire_docx) — generation/retrieval code never sees
# or needs to know about this.

import re

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_STAR_RE = re.compile(r"\*([^*\n]+?)\*")
_ITALIC_UNDERSCORE_RE = re.compile(r"_([^_\n]+?)_")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_BULLET_RE = re.compile(r"^\s*[-*]\s+(.+)$")
_NUMBERED_RE = re.compile(r"^\s*(\d+[.)])\s+(.+)$")


def _add_runs_with_inline_formatting(paragraph, text: str):
    """
    Adds text to `paragraph` as one or more runs, applying bold/italic
    where '**'/'*'/'_' markers are found, and stripping the markers
    themselves so they never appear as literal characters.
    """
    # Bold first (so **_x_** doesn't get mangled by italic regex eating
    # into the ** markers) — process bold spans, and within each bold
    # span's surrounding plain text, still check for italic.
    pos = 0
    for m in _BOLD_RE.finditer(text):
        _add_plain_with_italic(paragraph, text[pos:m.start()])
        run = paragraph.add_run(m.group(1))
        run.bold = True
        pos = m.end()
    _add_plain_with_italic(paragraph, text[pos:])


def _add_plain_with_italic(paragraph, text: str):
    if not text:
        return
    # Merge both marker types into one ordered pass so runs stay in order
    matches = []
    for m in _ITALIC_STAR_RE.finditer(text):
        matches.append((m.start(), m.end(), m.group(1)))
    for m in _ITALIC_UNDERSCORE_RE.finditer(text):
        matches.append((m.start(), m.end(), m.group(1)))
    matches.sort(key=lambda x: x[0])

    pos = 0
    for start, end, content in matches:
        if start < pos:
            continue  # overlapping match, skip (already consumed)
        if start > pos:
            paragraph.add_run(text[pos:start])
        run = paragraph.add_run(content)
        run.italic = True
        pos = end
    if pos < len(text):
        paragraph.add_run(text[pos:])


def _insert_markdown_content(anchor_paragraph, content: str):
    """
    Inserts `content` as one or more real docx paragraphs immediately
    after `anchor_paragraph`, converting Markdown-ish lines (headings,
    bullets, numbered lists, bold/italic) into corresponding docx
    formatting rather than leaving literal Markdown characters visible.

    Returns the last paragraph inserted, so the caller can keep chaining
    `addnext` calls in the right order.
    """
    lines = content.split("\n")
    current_anchor = anchor_paragraph

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            continue  # skip blank lines — avoids stray empty paragraphs

        heading_m = _HEADING_RE.match(line)
        bullet_m = _BULLET_RE.match(line)
        numbered_m = _NUMBERED_RE.match(line)

        new_para = current_anchor.insert_paragraph_before("")
        current_anchor._element.addnext(new_para._element)
        # move current_anchor forward so the NEXT line is inserted after
        # this one, preserving original order
        current_anchor = new_para

        if heading_m:
            _add_runs_with_inline_formatting(new_para, heading_m.group(2))
            for run in new_para.runs:
                run.bold = True
        elif bullet_m:
            new_para.style = "List Paragraph"
            _add_runs_with_inline_formatting(new_para, "•  " + bullet_m.group(1))
        elif numbered_m:
            new_para.style = "List Paragraph"
            marker, rest = numbered_m.group(1), numbered_m.group(2)
            _add_runs_with_inline_formatting(new_para, f"{marker}  {rest}")

    return current_anchor