"""
document_cleaner.py
-------------------
Post-parse, pre-agent filter for RAAT and DTA documents.

Receives the full markdown text produced by pdf_parser.py (Docling output)
and returns a reduced string containing only:

  1. The COVER BLOCK       — everything before the "Sommaire" heading,
                             capped at MAX_COVER_WORDS words.

  2. MATERIALS SECTION     — detected by one of two strategies:
                             a) Page-range slicing (when human config provides
                                page_start / page_end) — uses <!-- page N -->
                                markers injected by the updated pdf_parser.py
                             b) Sommaire-based auto-detection (fallback when
                                no config is provided) — parses the Sommaire,
                                matches section titles against a vocabulary,
                                falls back to trying sections 5->4->3->2.

Adding support for a new client whose section has an unseen title:
    → add one line to MATERIALS_SECTION_TITLES below, no other change needed.

Usage (without config — auto-detection):
    cleaner = DocumentCleaner()
    clean_text = cleaner.clean(full_text)

Usage (with human-provided config):
    from app.documents.parsers.config_reader import ConfigReader
    config = ConfigReader().read("config.xlsx")
    clean_text = cleaner.clean(full_text, config=config)
"""

import re
import unicodedata


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_COVER_WORDS = 1000

MATERIALS_SECTION_TITLES = [
    # DTA variants
    "liste des materiaux ou produits",
    "liste des materiaux et produits",
    "liste des produits et materiaux",
    "liste des produits ou materiaux",
    "conclusion detaille du reperage",
    "conclusion detaillee du reperage",
    "conclusions du rapport",
    "conclusion du reperage",
    "liste des materiaux",
    "resultats du reperage",
    "bilan du reperage",
    "reperage des materiaux",
    "materiaux reperes",
    # RAAT variants
    "conclusion detaille",
    "resultats",
    "recapitulatif zone par zone",
    # Pre-rapport / other client variants — add here as discovered
]

FALLBACK_SECTION_NUMBERS = [5, 4, 3, 2]

_RE_MATERIAL_ID = re.compile(r'\bM\d{3,4}\b')

_SOMMAIRE = re.compile(
    r'^#{1,4}\s*Sommaire\b',
    re.MULTILINE | re.IGNORECASE,
)
_SOMMAIRE_PLAIN = re.compile(
    r'(?:^|\n)Sommaire\s*\n',
    re.IGNORECASE,
)
_SOMMAIRE_ENTRY = re.compile(
    r'^[-\s]*(\d+)[\s.\-\u2013\u2014]+(.+)$',
    re.MULTILINE,
)

# Page marker injected by the updated pdf_parser.py
_PAGE_MARKER = re.compile(r'<!--\s*page\s+(\d+)\s*-->', re.IGNORECASE)


def _normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


class DocumentCleaner:

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

    def clean(self, full_text: str, config: dict | None = None) -> str:
        """
        Parameters
        ----------
        full_text : str
            Raw markdown from pdf_parser.parse()["full_text"].
            Must contain <!-- page N --> markers for page-range slicing.

        config : dict | None
            Optional dict from ConfigReader.read(). When page_start and
            page_end are present, page-range slicing is used instead of
            Sommaire-based auto-detection.
            Expected keys (all optional):
              page_start         : int
              page_end           : int
              columns_to_extract : list[str]   — passed through to agent
        """
        cover = self._extract_cover(full_text)

        # ── Materials block: config-driven or auto-detected ───────────────
        page_start = (config or {}).get("page_start")
        page_end   = (config or {}).get("page_end")

        if page_start is not None and page_end is not None:
            if self.debug:
                print(f"[DocumentCleaner] Config page range: {page_start} → {page_end}")
            materials = self._extract_by_page_range(full_text, page_start, page_end)
            if not materials:
                print(
                    f"[DocumentCleaner] Warning: page range {page_start}-{page_end} "
                    f"returned empty — falling back to auto-detection"
                )
                materials = self._extract_materials_section(full_text)
        else:
            materials = self._extract_materials_section(full_text)

        if self.debug:
            print(f"[DocumentCleaner] cover: {len(cover.split())} words")
            print(f"[DocumentCleaner] materials section: {len(materials.split())} words")
            print(f"[DocumentCleaner] original: {len(full_text.split())} words")

        parts = []
        if cover:
            parts.append(cover)
        if materials:
            parts.append(materials)

        if not parts:
            print("[DocumentCleaner] Warning: no sections found — returning full text")
            return full_text

        cleaned = "\n\n<!-- section-break -->\n\n".join(parts)

        if self.debug:
            print(
                f"[DocumentCleaner] cleaned: {len(cleaned.split())} words "
                f"({100 * len(cleaned) // max(len(full_text), 1)}% of original)"
            )

        return cleaned

    # ------------------------------------------------------------------
    # Private — cover block
    # ------------------------------------------------------------------

    def _extract_cover(self, text: str) -> str:
        m = _SOMMAIRE.search(text)
        if not m:
            m = _SOMMAIRE_PLAIN.search(text)

        cover_text = text[: m.start()].strip() if m else text.strip()

        words = cover_text.split()
        if len(words) > self.max_cover_words:
            cover_text = " ".join(words[: self.max_cover_words])

        return cover_text.strip()

    # ------------------------------------------------------------------
    # Private — page-range slicing (config-driven)
    # ------------------------------------------------------------------

    def _extract_by_page_range(
        self,
        text: str,
        page_start: int,
        page_end: int,
    ) -> str:
        """
        Slices full_text to keep only pages page_start..page_end inclusive.

        Relies on <!-- page N --> markers injected by pdf_parser.py.
        Returns empty string if no markers are found (old parser output).
        """
        markers = list(_PAGE_MARKER.finditer(text))

        if not markers:
            print(
                "[DocumentCleaner] Warning: no <!-- page N --> markers found — "
                "re-parse the PDF with the updated pdf_parser.py"
            )
            return ""

        # Build index: page_no → start char position in text
        page_positions: dict[int, int] = {}
        for m in markers:
            page_no = int(m.group(1))
            page_positions[page_no] = m.start()

        # Clamp to available pages
        available = sorted(page_positions.keys())
        first_page = available[0]
        last_page  = available[-1]

        clamped_start = max(page_start, first_page)
        clamped_end   = min(page_end,   last_page)

        if clamped_start > clamped_end:
            print(
                f"[DocumentCleaner] Warning: requested pages {page_start}-{page_end} "
                f"outside available range {first_page}-{last_page}"
            )
            return ""

        if clamped_start != page_start or clamped_end != page_end:
            print(
                f"[DocumentCleaner] Page range clamped: "
                f"{page_start}-{page_end} → {clamped_start}-{clamped_end}"
            )

        # Slice start = position of the page_start marker
        slice_start = page_positions[clamped_start]

        # Slice end = position of the marker AFTER page_end (or end of text)
        next_page = clamped_end + 1
        if next_page in page_positions:
            slice_end = page_positions[next_page]
        else:
            slice_end = len(text)

        result = text[slice_start:slice_end].strip()

        if self.debug:
            print(
                f"[DocumentCleaner] Page range slice: chars "
                f"{slice_start}-{slice_end} "
                f"({len(result.split())} words)"
            )

        return result

    # ------------------------------------------------------------------
    # Private — Sommaire-based auto-detection
    # ------------------------------------------------------------------

    def _extract_materials_section(self, text: str) -> str:
        sommaire_map = self._parse_sommaire(text)
        if self.debug:
            print(f"[DocumentCleaner] Sommaire map: {sommaire_map}")

        section_number = self._find_materials_section_number(sommaire_map)
        if section_number is not None:
            if self.debug:
                print(
                    f"[DocumentCleaner] Matched section {section_number} "
                    f"via Sommaire: '{sommaire_map.get(section_number)}'"
                )
            result = self._extract_section_by_number(text, section_number)
            if result:
                return result

        if self.debug:
            print("[DocumentCleaner] Sommaire match failed — trying fallback numbers")
        for num in FALLBACK_SECTION_NUMBERS:
            candidate = self._extract_section_by_number(text, num)
            if candidate and _RE_MATERIAL_ID.search(candidate):
                if self.debug:
                    print(
                        f"[DocumentCleaner] Fallback: using section {num} "
                        f"(contains M-identifiers)"
                    )
                return candidate

        print("[DocumentCleaner] Warning: materials section not found")
        return ""

    def _parse_sommaire(self, text: str) -> dict[str, str]:
        m = _SOMMAIRE.search(text) or _SOMMAIRE_PLAIN.search(text)
        if not m:
            return {}

        sommaire_start = m.end()
        next_heading = re.search(r'^#{1,4}\s', text[sommaire_start:], re.MULTILINE)
        sommaire_end = (
            sommaire_start + next_heading.start()
            if next_heading
            else sommaire_start + 2000
        )

        sommaire_text = text[sommaire_start:sommaire_end]
        result = {}
        for entry in _SOMMAIRE_ENTRY.finditer(sommaire_text):
            num   = entry.group(1).strip()
            title = entry.group(2).strip()
            if num not in result:
                result[num] = title

        return result

    def _find_materials_section_number(self, sommaire_map: dict[str, str]) -> str | None:
        for num, title in sommaire_map.items():
            normalized_title = _normalize(title)
            for vocab_entry in MATERIALS_SECTION_TITLES:
                if vocab_entry in normalized_title:
                    return num
        return None

    def _extract_section_by_number(self, text: str, section_number: int | str) -> str:
        num = str(section_number)

        start_pattern = re.compile(
            rf'^#{{1,4}}\s*{re.escape(num)}[\s.\-\u2013\u2014]',
            re.MULTILINE | re.IGNORECASE,
        )
        start_match = start_pattern.search(text)
        if not start_match:
            return ""

        start = start_match.start()

        next_section_pattern = re.compile(
            rf'^#{{1,4}}\s*(?!{re.escape(num)}[\.\d])(\d+)[\s.\-\u2013\u2014]',
            re.MULTILINE | re.IGNORECASE,
        )
        end_match = next_section_pattern.search(text, start + 1)

        if end_match:
            found_num = int(end_match.group(1))
            if found_num > int(num):
                end = end_match.start()
            else:
                end = len(text)
                for m in next_section_pattern.finditer(text, start + 1):
                    if int(m.group(1)) > int(num):
                        end = m.start()
                        break
        else:
            end = len(text)

        return text[start:end].strip()

    # ------------------------------------------------------------------
    # Convenience: stats only
    # ------------------------------------------------------------------

    def stats(self, full_text: str, config: dict | None = None) -> dict:
        cover     = self._extract_cover(full_text)
        page_start = (config or {}).get("page_start")
        page_end   = (config or {}).get("page_end")

        if page_start is not None and page_end is not None:
            materials = self._extract_by_page_range(full_text, page_start, page_end)
            if not materials:
                materials = self._extract_materials_section(full_text)
        else:
            materials = self._extract_materials_section(full_text)

        total = len(full_text.split())
        kept  = len(cover.split()) + len(materials.split())
        return {
            "total_words":     total,
            "cover_words":     len(cover.split()),
            "materials_words": len(materials.split()),
            "kept_words":      kept,
            "reduction_pct":   round(100 * (1 - kept / max(total, 1))),
        }