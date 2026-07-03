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

DEFAULT_THRESHOLD = 0.55


def load_sections_dict(path: Path = SECTIONS_DICT_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^\d+([.\d]*)[).\s-]*", "", text)  # strip leading numbering: "3.2 " / "III - "
    text = re.sub(r"\s+", " ", text)
    return text


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


def _candidate_pool(sections_dict: dict, target_hint: str | None) -> list[str]:
    # If the user (or the intent-parsing LLM call) named a specific canonical
    # label, restrict matching to that entry's variants only — tighter match,
    # fewer false positives. Otherwise pool every known variant across the dict.
    if target_hint:
        norm_hint = _normalize(target_hint)
        for entry in sections_dict.values():
            all_names = [entry["label"]] + entry["variants"]
            if any(_normalize(n) == norm_hint or norm_hint in _normalize(n) for n in all_names):
                return [entry["label"]] + entry["variants"]
        # Hint didn't match a known canonical entry — use it verbatim as the
        # sole candidate. Covers free-text section names the user typed.
        return [target_hint]

    pool = []
    for entry in sections_dict.values():
        pool.append(entry["label"])
        pool.extend(entry["variants"])
    return pool


def _score(title: str, candidates: list[str]) -> float:
    norm_title = _normalize(title)
    best = 0.0
    for c in candidates:
        norm_c = _normalize(c)
        ratio = difflib.SequenceMatcher(None, norm_title, norm_c).ratio()
        if norm_c in norm_title or norm_title in norm_c:
            ratio = max(ratio, 0.85)  # substring containment is a strong signal
        best = max(best, ratio)
    return best


def _section_text(markdown_text: str, headings: list[dict], matched_index: int) -> str:
    matched = headings[matched_index]
    end = len(markdown_text)
    for nxt in headings[matched_index + 1:]:
        if nxt["level"] <= matched["level"]:
            end = nxt["line_start"]
            break
    raw = markdown_text[matched["body_start"]:end]
    return PAGE_MARKER_RE.sub("", raw).strip()


def match_section(
    markdown_text: str,
    target_hint: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    sections_dict: dict | None = None,
) -> dict | None:
    """
    Returns the best-matching section as:
        {"heading": str, "level": int, "score": float, "text": str}
    or None if nothing scored above `threshold`.
    """
    sections_dict = sections_dict or load_sections_dict()
    headings = _extract_headings(markdown_text)
    if not headings:
        return None

    candidates = _candidate_pool(sections_dict, target_hint)

    best_idx, best_score = None, 0.0
    for i, h in enumerate(headings):
        s = _score(h["title"], candidates)
        if s > best_score:
            best_idx, best_score = i, s

    if best_idx is None or best_score < threshold:
        return None

    return {
        "heading": headings[best_idx]["title"],
        "level": headings[best_idx]["level"],
        "score": round(best_score, 3),
        "text": _section_text(markdown_text, headings, best_idx),
    }
