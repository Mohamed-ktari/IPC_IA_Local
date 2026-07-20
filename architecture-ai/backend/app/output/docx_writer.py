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
    """
    structure_docx_path: the structure .docx the user uploaded (same file
        parse_structure_docx() was called on).
    filled_sections: list of {"level", "title", "content"} in the SAME
        order as parse_structure_docx()'s output — content is the
        generated text (or the user's own existing_content if generation
        was skipped for that section).
    output_path: where to save the filled .docx.
    """
    structure_docx_path = Path(structure_docx_path)
    output_path = Path(output_path)

    doc = Document(structure_docx_path)

    # Build (paragraph_index, level, title) for every heading paragraph,
    # in document order — mirrors parse_structure_docx()'s walk exactly so
    # the i-th heading found here lines up with filled_sections[i].
    heading_positions = []
    for idx, p in enumerate(doc.paragraphs):
        level = _heading_level_for_writer(p)
        if level is not None:
            heading_positions.append(idx)

    if len(heading_positions) != len(filled_sections):
        raise ValueError(
            f"Structure mismatch: found {len(heading_positions)} headings "
            f"in the document but received {len(filled_sections)} filled "
            f"sections. The document may have been edited between parsing "
            f"and filling — re-parse before filling."
        )

    # Walk headings in REVERSE so earlier insertions don't shift the
    # paragraph indices of headings we haven't processed yet.
    for pos, section in zip(reversed(heading_positions), reversed(filled_sections)):
        heading_para = doc.paragraphs[pos]

        # Remove existing content paragraphs between this heading and the
        # next heading (or end of doc) — these are the blank/instruction
        # paragraphs from the original skeleton.
        next_pos = None
        for later in heading_positions:
            if later > pos:
                next_pos = later
                break
        end = next_pos if next_pos is not None else len(doc.paragraphs)

        for p in doc.paragraphs[pos + 1:end]:
            p._element.getparent().remove(p._element)

        # Insert generated content as a new paragraph right after the heading
        new_para = heading_para.insert_paragraph_before("")  # placeholder position trick
        # insert_paragraph_before inserts BEFORE heading_para — we actually
        # want after, so move the heading's XML before our new paragraph.
        heading_para._element.addnext(new_para._element)
        new_para.text = section["content"]

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