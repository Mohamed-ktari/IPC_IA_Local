# section_matcher.py
# Finds a target section inside an RC document based on its markdown headings.
#
# Why not RAG here: the RC is a single document with a predictable structure
# (numbered headings). We need one *complete, contiguous* section, not a
# top-k of semantically similar fragments. A heading match against a known
# vocabulary is deterministic, auditable, and immune to chunk-boundary loss.
#
# Works directly on the markdown produced by PDFParser.parse()["full_text"].

import re
import json
import difflib
from pathlib import Path

SECTIONS_DICT_PATH = Path(__file__).resolve().parent / "dictionnaries" / "rc_sections_dict.json"

# Strips the "<!-- page N -->" markers PDFParser injects between pages —
# they're useful for page-range slicing elsewhere but just noise here.
PAGE_MARKER_RE = re.compile(r"<!--\s*page\s+\d+\s*-->", re.IGNORECASE)

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$", re.MULTILINE)

# Docling assigns heading levels from visual layout cues (font, indent), not
# semantics — a numbered sub-criterion like "1) Valeur technique 65%" can end
# up at the SAME markdown level as the article heading above it. Level alone
# is therefore not reliable for finding where a section actually ends.
#
# These patterns classify headings by their raw title shape (checked BEFORE
# any numbering-stripping normalization) so we can tell "real new division"
# apart from "enumerated sub-item that happens to render at the same level".
ARTICLE_STYLE_RE = re.compile(r"^\(?(article|titre|chapitre|section)\b\s*\d*", re.IGNORECASE)
# Only parenthesis/bullet style counts as "enumerated sub-item" — e.g. "1)",
# "a)", "•". Period-style numbering ("1.", "3.2") is left alone: that's the
# convention legitimate top-level/nested chapters use throughout these docs,
# so treating it as an enum-subitem would wrongly swallow real boundaries.
ENUM_SUBITEM_RE = re.compile(r"^\(?\d+\)|^\(?[a-zA-Z]\)|^[-•▪]\s")


# Headings that are clearly sub-criteria / sub-items of a judgment section.
# They must stay inside the parent section even when they render at the same
# markdown level as the parent heading.
CRITERION_SUBITEM_RE = re.compile(
    r"^(sous[-\s]?critère|critère)\b",
    re.IGNORECASE
)

# Parses a leading hierarchical number sequence — "7.2 - JUGEMENT..." -> (7, 2),
# "7.2.1 – Jugement..." -> (7, 2, 1), "ARTICLE 7" -> (7,). Used to detect
# continuing/nested numbering ("7.2.1" under "7.2") as distinct from
# restarting enumeration ("1)", "2)" under "ARTICLE 7") — two conventions
# real RCs mix, neither of which is caught by the other's heuristic alone.
NUMERIC_PREFIX_RE = re.compile(
    r"^\(?(?:article|titre|chapitre|section)?\s*\.?\s*(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)


def _numeric_prefix(title: str) -> tuple[int, ...] | None:
    m = NUMERIC_PREFIX_RE.match(title)
    if not m:
        return None
    return tuple(int(p) for p in m.group(1).split("."))


def _is_numeric_descendant(nxt_prefix: tuple, matched_prefix: tuple) -> bool:
    if nxt_prefix is None or matched_prefix is None:
        return False
    return len(nxt_prefix) > len(matched_prefix) and nxt_prefix[: len(matched_prefix)] == matched_prefix


# Some RCs explicitly point from one subsection to a sibling, e.g. "...seront
# détaillés dans le paragraphe suivant (7.2.2)". That sibling is correctly
# excluded by the descendancy check above (it's not a child), but the author
# clearly intends the reader to read both together. Follow such references
# rather than stretching the boundary rule to cover it — stretching it would
# risk swallowing unrelated siblings in other documents that don't cross-refer.
CROSS_REF_RE = re.compile(r"\(\s*(\d+(?:\.\d+){1,3})\s*\)")


def _find_heading_index_by_prefix(headings: list[dict], prefix: tuple) -> int | None:
    for i, h in enumerate(headings):
        if _numeric_prefix(h["title"]) == prefix:
            return i
    return None

DEFAULT_THRESHOLD = 0.55

# Generic navigational headings that exist in almost every RC but are never
# real content sections. Excluded outright rather than trusted to score low —
# a bare "SOMMAIRE" heading is a substring of "sommaire attendu du mémoire
# technique" and would otherwise get an inflated containment-boost score.
GENERIC_HEADING_BLOCKLIST = {
    "sommaire",
    "table des matieres",
    "table des matières",
    "index",
    "plan du document",
    "sommaire detaille",
    "sommaire détaillé",
}


def load_sections_dict(path: Path = SECTIONS_DICT_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^\d+([.\d]*)[).\s-]*", "", text)  # strip leading numbering: "3.2 " / "III - "
    text = re.sub(r"\s+", " ", text)
    return text

normalize_label = _normalize

def _extract_headings(markdown_text: str) -> list[dict]:
    headings = []
    for m in HEADING_RE.finditer(markdown_text):
        headings.append({
            "level": len(m.group(1)),
            "title": m.group(2).strip(),
            "line_start": m.start(),
            "body_start": m.end(),
        })
    return headings


def _candidate_pool(sections_dict: dict, target_hint: str | None, default_category: str | None = None) -> list[str]:
    # If the user (or the intent-parsing LLM call) named a specific canonical
    # label, restrict matching to that entry's variants only — tighter match,
    # fewer false positives. Otherwise fall back to `default_category` if the
    # caller knows what kind of section it's looking for (e.g. structure
    # extraction should only ever search structure_memoire, never accidentally
    # match a judgment-criteria heading just because it scores higher overall).
    if target_hint:
        norm_hint = _normalize(target_hint)
        for entry in sections_dict.values():
            all_names = [entry["label"]] + entry["variants"]
            if any(_normalize(n) == norm_hint or norm_hint in _normalize(n) for n in all_names):
                return [entry["label"]] + entry["variants"]
        # Hint didn't match a known canonical entry — use it verbatim as the
        # sole candidate. Covers free-text section names the user typed.
        return [target_hint]

    if default_category and default_category in sections_dict:
        entry = sections_dict[default_category]
        return [entry["label"]] + entry["variants"]

    # No hint and no default category — genuinely don't know what we're
    # looking for, so pool everything. Only used by generic/exploratory
    # callers, never by category-specific extraction methods.
    pool = []
    for entry in sections_dict.values():
        pool.append(entry["label"])
        pool.extend(entry["variants"])
    return pool


def _score(title: str, candidates: list[str]) -> float:
    norm_title = _normalize(title)
    if norm_title in GENERIC_HEADING_BLOCKLIST:
        return 0.0

    best = 0.0
    for c in candidates:
        norm_c = _normalize(c)
        ratio = difflib.SequenceMatcher(None, norm_title, norm_c).ratio()

        if norm_c in norm_title or norm_title in norm_c:
            # Containment is a signal, not proof — a one-word heading that
            # happens to be a substring of a long candidate phrase shouldn't
            # get the same confidence as a heading that covers most of it.
            # Scale the boost by how much of the longer string the shorter
            # one actually covers.
            shorter, longer = sorted([norm_title, norm_c], key=len)
            if len(longer) > 0:
                coverage = len(shorter) / len(longer)
                boosted = 0.5 + 0.45 * coverage  # ranges ~0.5 (weak) to 0.95 (near-full overlap)
                ratio = max(ratio, boosted)

        best = max(best, ratio)
    return best


def _section_text(markdown_text: str, headings: list[dict], matched_index: int) -> str:
    matched = headings[matched_index]
    is_article_style = bool(ARTICLE_STYLE_RE.match(matched["title"]))
    matched_prefix = _numeric_prefix(matched["title"])

    end = len(markdown_text)
    for nxt in headings[matched_index + 1:]:
        if nxt["level"] > matched["level"]:
            continue  # strictly deeper heading — always part of this section

        if ENUM_SUBITEM_RE.match(nxt["title"]):
            continue  # "1)", "a)", "•" — restarting enumeration, not a boundary
        
        # NEW: "Critère 1", "Sous-critère 2", etc. stay inside the parent
        if CRITERION_SUBITEM_RE.match(nxt["title"]):
            continue

        if _is_numeric_descendant(_numeric_prefix(nxt["title"]), matched_prefix):
            continue  # "7.2.1" under "7.2" — continuing hierarchical numbering

        nxt_prefix = _numeric_prefix(nxt["title"])
        if (
            matched_prefix is not None
            and nxt_prefix is not None
            and len(matched_prefix) > 1
            and len(nxt_prefix) == 1
            and nxt_prefix[0] < matched_prefix[0]
        ):
            # Period-style numbering restarting from a small number ("1.",
            # "2.") while we're already deep inside a subsection (e.g. inside
            # 7.2.2) — real chapters advance monotonically, they don't jump
            # backward, so this is a local enumeration restart, not a new
            # top-level chapter. Same visual shape as legitimate chapter
            # numbering, disambiguated only by context.
            continue

        if is_article_style and not ARTICLE_STYLE_RE.match(nxt["title"]):
            continue  # matched used an Article/Titre/Chapitre marker —
            # only another heading of that same style counts as the next section

        end = nxt["line_start"]
        break

    raw = markdown_text[matched["body_start"]:end]
    print("Raw : "+raw)
    return PAGE_MARKER_RE.sub("", raw).strip()


def _expand_cross_references(
    markdown_text: str,
    headings: list[dict],
    section_text: str,
    matched_prefix: tuple | None,
) -> str:
    if matched_prefix is None:
        return section_text

    seen = {matched_prefix}
    appended = []
    for ref_str in CROSS_REF_RE.findall(section_text):
        ref_prefix = tuple(int(p) for p in ref_str.split("."))
        if ref_prefix in seen:
            continue
        # Only follow references to a sibling under the SAME parent (same
        # prefix except the last element) — e.g. 7.2.1 -> 7.2.2. Following
        # references to unrelated numbering (legal article citations like
        # "L.2152-1", or distant clauses) is out of scope and risks pulling
        # in noise the user didn't ask for.
        if len(ref_prefix) != len(matched_prefix) or ref_prefix[:-1] != matched_prefix[:-1]:
            continue
        idx = _find_heading_index_by_prefix(headings, ref_prefix)
        if idx is None:
            continue
        seen.add(ref_prefix)
        ref_text = _section_text(markdown_text, headings, idx)
        appended.append(f"\n\n[Référence croisée — {headings[idx]['title']}]\n{ref_text}")

    return section_text + "".join(appended)


def match_section(
    markdown_text: str,
    target_hint: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    sections_dict: dict | None = None,
    default_category: str | None = None,
) -> dict | None:
    """
    Returns the best-matching section as:
        {"heading": str, "level": int, "score": float, "text": str}
    or None if nothing scored above `threshold`.

    `default_category` scopes the search to one dictionary entry (e.g.
    "structure_memoire") when no explicit target_hint is given — use this
    whenever the caller knows what kind of section it's looking for, to
    avoid matching a heading from an unrelated category just because it
    scores higher across the whole dictionary.
    """
    sections_dict = sections_dict or load_sections_dict()
    headings = _extract_headings(markdown_text)
    if not headings:
        return None

    candidates = _candidate_pool(sections_dict, target_hint, default_category)

    best_idx, best_score = None, 0.0
    for i, h in enumerate(headings):
        s = _score(h["title"], candidates)
        if s > best_score:
            best_idx, best_score = i, s

    if best_idx is None or best_score < threshold:
        return None

    matched_prefix = _numeric_prefix(headings[best_idx]["title"])
    section_text = _section_text(markdown_text, headings, best_idx)
    section_text = _expand_cross_references(markdown_text, headings, section_text, matched_prefix)

    return {
        "heading": headings[best_idx]["title"],
        "level": headings[best_idx]["level"],
        "score": round(best_score, 3),
        "text": section_text,
    }

def is_section_good_enough(section: dict | None, min_words: int = 30) -> bool:
    """Return True only if the match looks complete enough to trust."""
    if section is None:
        return False
    text = section.get("text") or ""
    word_count = len(text.split())
    return word_count >= min_words