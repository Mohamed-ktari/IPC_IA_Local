"""
document_cleaner.py
-------------------
Post-parse, pre-agent filter for RAAT and DTA documents.

Receives the full markdown text produced by pdf_parser.py (Docling output)
and returns a reduced string containing only:

  1. The COVER BLOCK  — everything before the "Sommaire" heading,
                        capped at MAX_COVER_WORDS words.
                        Contains: title, dossier number, date, address,
                        operator, lab, client — all header fields.

  2. SECTION 5 BLOCK  — the full "Conclusion détaillé du repérage" section
                        (5, 5.1, 5.2, and any deeper sub-levels).
                        Contains: all material tables with results.

Everything else (Sommaire, sections 1-4, 6, 7, annexes, lab reports,
insurance certificates, safety recommendations) is dropped before the
text reaches the LLM, cutting token usage by ~70 % on a typical
36-page RAAT.

Usage
-----
    from app.documents.parsers.document_cleaner import DocumentCleaner

    cleaner = DocumentCleaner()
    clean_text = cleaner.clean(raw_markdown)

The output is a plain string — drop-in replacement for `full_text` wherever
you currently pass it to ExtractionAgent.run_extraction().
"""

import re


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hard cap on cover-block words (the page before Sommaire is usually < 300
# words; 1 000 gives a generous safety margin for unusual layouts).
MAX_COVER_WORDS = 1000

# Heading patterns that mark the START of Section 5 in Docling markdown.
# Docling renders headings as "## 5. ...", "## 5 -", "## 5.", etc.
# We match any heading whose first token after "##*" is the digit 5.
_SEC5_START = re.compile(
    r'^#{1,4}\s*5[\s.\-–—]',
    re.MULTILINE | re.IGNORECASE,
)

# Heading patterns that mark the END of Section 5 (i.e. start of Section 6).
# We stop as soon as we hit a same-or-higher-level heading for section 6+.
_SEC5_END = re.compile(
    r'^#{1,4}\s*6[\s.\-–—]',
    re.MULTILINE | re.IGNORECASE,
)

# The "Sommaire" heading — we use it as the boundary between cover and body.
_SOMMAIRE = re.compile(
    r'^#{1,4}\s*Sommaire\b',
    re.MULTILINE | re.IGNORECASE,
)

# Fallback: plain text "Sommaire" not in a heading (some templates omit ##)
_SOMMAIRE_PLAIN = re.compile(
    r'(?:^|\n)Sommaire\s*\n',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DocumentCleaner:
    """
    Filters parsed DTA / RAAT markdown to the two blocks needed for
    material extraction.
    """

    def __init__(
        self,
        max_cover_words: int = MAX_COVER_WORDS,
        debug: bool = False,
    ):
        self.max_cover_words = max_cover_words
        self.debug = debug

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, full_text: str) -> str:
        """
        Main entry point.

        Parameters
        ----------
        full_text : str
            Raw markdown returned by pdf_parser.parse()["full_text"].

        Returns
        -------
        str
            Filtered text ready to be passed to ExtractionAgent.
        """
        cover = self._extract_cover(full_text)
        section5 = self._extract_section5(full_text)

        if self.debug:
            print(f"[DocumentCleaner] cover: {len(cover.split())} words")
            print(f"[DocumentCleaner] section5: {len(section5.split())} words")
            print(f"[DocumentCleaner] original: {len(full_text.split())} words")

        parts = []
        if cover:
            parts.append(cover)
        if section5:
            parts.append(section5)

        if not parts:
            # Safety fallback: if parsing failed entirely, return original
            # so the pipeline degrades gracefully rather than crashing.
            print("[DocumentCleaner] Warning: no sections found — returning full text")
            return full_text

        cleaned = "\n\n<!-- section-break -->\n\n".join(parts)

        if self.debug:
            print(f"[DocumentCleaner] cleaned: {len(cleaned.split())} words "
                  f"({100 * len(cleaned) // max(len(full_text), 1)}% of original)")

        return cleaned

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_cover(self, text: str) -> str:
        """
        Returns everything before the Sommaire heading,
        capped at max_cover_words words.
        """
        # Try markdown heading first (## Sommaire)
        m = _SOMMAIRE.search(text)

        # Fallback to plain "Sommaire\n" line if no heading found
        if not m:
            m = _SOMMAIRE_PLAIN.search(text)

        if m:
            cover_text = text[: m.start()].strip()
        else:
            # No Sommaire found — take the first max_cover_words words
            # from the whole document (unusual but safe)
            cover_text = text.strip()

        # Cap at max_cover_words
        words = cover_text.split()
        if len(words) > self.max_cover_words:
            cover_text = " ".join(words[: self.max_cover_words])

        return cover_text.strip()

    def _extract_section5(self, text: str) -> str:
        """
        Returns the full content of Section 5 (all sub-sections included).
        Stops at the first Section 6+ heading or end of document.
        """
        start_match = _SEC5_START.search(text)
        if not start_match:
            print("[DocumentCleaner] Warning: Section 5 heading not found")
            return ""

        start = start_match.start()

        # Look for Section 6 heading AFTER the section 5 start
        end_match = _SEC5_END.search(text, start + 1)
        end = end_match.start() if end_match else len(text)

        return text[start:end].strip()

    # ------------------------------------------------------------------
    # Convenience: stats only (useful for debugging without modifying text)
    # ------------------------------------------------------------------

    def stats(self, full_text: str) -> dict:
        """Returns word counts for each block without assembling output."""
        cover = self._extract_cover(full_text)
        section5 = self._extract_section5(full_text)
        total = len(full_text.split())
        kept = len(cover.split()) + len(section5.split())
        return {
            "total_words": total,
            "cover_words": len(cover.split()),
            "section5_words": len(section5.split()),
            "kept_words": kept,
            "reduction_pct": round(100 * (1 - kept / max(total, 1))),
        }