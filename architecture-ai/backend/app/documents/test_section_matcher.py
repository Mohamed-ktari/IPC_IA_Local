# test_section_matcher.py
#
# Regression suite for section_matcher.py.
#
# Every test here corresponds to a REAL bug found on a real RC document
# during development, not a hypothetical edge case. Do not delete or weaken
# an assertion just because it's inconvenient after a future change — if a
# change legitimately requires updating one of these, that's a deliberate
# decision to make explicitly, not something to fix by loosening the test.
#
# Run with:
#   pytest test_section_matcher.py -v
#
# Import strategy: tries the real package path first (once this file lives
# at app/documents/test_section_matcher.py next to section_matcher.py and
# is run via pytest from the project root). Falls back to loading the
# module directly by file path so this also runs standalone before it's
# wired into the app package.

import importlib.util
from pathlib import Path

import pytest

try:
    from app.documents.section_matcher import match_section, _numeric_prefix
except ImportError:
    _module_path = Path(__file__).resolve().parent / "section_matcher.py"
    _spec = importlib.util.spec_from_file_location("section_matcher", _module_path)
    _sm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_sm)
    match_section = _sm.match_section
    _numeric_prefix = _sm._numeric_prefix


# ----------------------------------------------------------------------
# Bug #1 — "SOMMAIRE" (table of contents heading) scored 0.85 via a flat
# substring-containment boost, beating the real target heading.
# ----------------------------------------------------------------------

def test_generic_toc_heading_is_not_matched():
    md = (
        "# SOMMAIRE\n1. Objet\n\n"
        "## ARTICLE 7 CRITERES DE JUGEMENT DES OFFRES\n"
        "texte pertinent ici.\n"
    )
    result = match_section(md)
    assert result is not None
    assert result["heading"].startswith("ARTICLE 7")


# ----------------------------------------------------------------------
# Bug #2 — a broad parent heading ("TITRE II...") containing the same
# keywords as the real target must not outscore the specific child heading.
# ----------------------------------------------------------------------

def test_specific_child_heading_preferred_over_broad_parent():
    md = (
        "# TITRE II - SELECTION DES CANDIDATURES ET JUGEMENT DES OFFRES\n"
        "intro.\n\n"
        "## ARTICLE 7 CRITERES DE JUGEMENT DES OFFRES\n"
        "contenu.\n\n"
        "## ARTICLE 8 AUTRE CHOSE\n"
        "hors sujet\n"
    )
    result = match_section(md)
    assert result["heading"].startswith("ARTICLE 7")
    assert "contenu." in result["text"]
    assert "hors sujet" not in result["text"]


# ----------------------------------------------------------------------
# Bug #3 — enumerated sub-items ("1)", "2)") rendered by docling at the
# SAME markdown level as the parent "ARTICLE 7" heading, truncating
# extraction right after the intro sentence.
# ----------------------------------------------------------------------

def test_parenthesis_style_subitems_included_not_treated_as_boundary():
    md = (
        "## ARTICLE 7 CRITERES DE JUGEMENT DES OFFRES\n\n"
        "intro.\n\n"
        "## 1) Valeur technique 65 %\n\n"
        "sc1.\n\n"
        "## 2) Prix de l'offre 35 %\n\n"
        "sc2.\n\n"
        "## ARTICLE 8 CONDITIONS D'ATTRIBUTION\n\n"
        "hors sujet.\n"
    )
    result = match_section(md)
    assert "sc1." in result["text"]
    assert "sc2." in result["text"]
    assert "hors sujet." not in result["text"]


# ----------------------------------------------------------------------
# Regression guard — period-style numbering ("3.", "4.") is a LEGITIMATE
# top-level chapter marker elsewhere in these documents and must still work
# as a real boundary when encountered at the top level (not to be confused
# with bug #5 below, which is the same *shape* used differently in context).
# ----------------------------------------------------------------------

def test_period_numbered_chapters_with_nested_subsections():
    md = (
        "## 3. Structure du mémoire technique\n"
        "Le mémoire devra comporter :\n\n"
        "### 3.1 Présentation\ntexte.\n\n"
        "### 3.2 Méthodologie\ntexte.\n\n"
        "## 4. Autre chapitre\nhors sujet\n"
    )
    result = match_section(md, target_hint="structure du mémoire")
    assert result["heading"].startswith("3.")
    assert "texte." in result["text"]
    assert "hors sujet" not in result["text"]


# ----------------------------------------------------------------------
# Bug #4 — continuing hierarchical numbering (7.2 -> 7.2.1 -> 7.2.2), all
# rendered at the same markdown level, must be recognized as
# parent/descendant rather than siblings.
# ----------------------------------------------------------------------

def test_hierarchical_continuing_numbering_included_as_descendants():
    md = (
        "## 7.1 - CRITERES DE SELECTION\nhors sujet.\n\n"
        "## 7.2 - JUGEMENT DES OFFRES\nintro.\n\n"
        "## 7.2.1 – Jugement des offres\nprincipal.\n\n"
        "## 7.2.2 – Détail des critères de sélection\ndetail.\n\n"
        "## 8. AUTRE CHAPITRE\nhors sujet2.\n"
    )
    result = match_section(md, target_hint="critère de jugement")
    assert "principal." in result["text"]
    assert "detail." in result["text"]
    assert "hors sujet2." not in result["text"]
    assert "hors sujet." not in result["text"]  # the 7.1 sibling must be excluded


# ----------------------------------------------------------------------
# Bug #5 (part A) — explicit cross-reference: "...détaillés dans le
# paragraphe suivant (7.2.2)" should pull in the referenced sibling.
# ----------------------------------------------------------------------

def test_explicit_cross_reference_to_sibling_is_followed():
    md = (
        "## 7.2.1 - Jugement des offres\n"
        "Les critères seront détaillés dans le paragraphe suivant (7.2.2).\n\n"
        "## 7.2.2 - Détail des critères de sélection\n"
        "contenu detail reference.\n\n"
        "## 7.3 - AUTRE\nhors sujet.\n"
    )
    result = match_section(md, target_hint="critère de jugement")
    assert "contenu detail reference." in result["text"]
    assert "hors sujet." not in result["text"]


# ----------------------------------------------------------------------
# Bug #5 (part B) — cross-reference following must NOT fire when there is
# no explicit reference — a sibling with a similar-sounding title should
# stay excluded by default.
# ----------------------------------------------------------------------

def test_sibling_not_pulled_in_without_explicit_reference():
    md = (
        "## 7.2.1 - Jugement des offres\n"
        "contenu sans reference explicite.\n\n"
        "## 7.2.2 - Détail des critères de sélection\n"
        "contenu detail non lie.\n\n"
        "## 7.3 - AUTRE\nhors sujet.\n"
    )
    result = match_section(md, target_hint="critère de jugement")
    assert "contenu detail non lie." not in result["text"]


# ----------------------------------------------------------------------
# Bug #6 — period-style numbering RESTARTING from a small number ("1.",
# "2.", "3.") while nested deep inside a subsection (e.g. under 7.2.2) must
# be treated as a local enumeration restart, not a new top-level chapter —
# even though the same "N." shape is a legitimate chapter marker elsewhere
# (see test_period_numbered_chapters_with_nested_subsections above).
# ----------------------------------------------------------------------

def test_period_style_restart_numbering_nested_under_deep_subsection():
    md = (
        "## 7.2.1 - Jugement des offres\n"
        "Les critères seront détaillés dans le paragraphe suivant (7.2.2).\n\n"
        "## 7.2.2 - Détail des critères de sélection\n"
        "Détail des critères ci-dessous.\n\n"
        "## 1. Le prix des prestations complètes du candidat (60 pts)\n"
        "Barème dégressif standard.\n\n"
        "## 2. Présentation de l'équipe + Références (10 pts)\n"
        "CV et références exigées.\n\n"
        "## 3. Méthodologie et moyens pour le suivi des travaux (30 pts)\n"
        "Plan d'action détaillé.\n\n"
        "## 8. AUTRE CHAPITRE\n"
        "hors sujet, ne doit pas être inclus.\n"
    )
    result = match_section(md, target_hint="critère de jugement")
    assert "Détail des critères ci-dessous." in result["text"]
    assert "Barème dégressif standard." in result["text"]
    assert "CV et références exigées." in result["text"]
    assert "Plan d'action détaillé." in result["text"]
    assert "ne doit pas être inclus." not in result["text"]


# ----------------------------------------------------------------------
# Sanity checks — target_hint resolution and graceful "not found"
# ----------------------------------------------------------------------

def test_no_hint_falls_back_to_full_dictionary_pool():
    md = (
        "## ARTICLE 6 CRITERES DE SELECTION DES CANDIDATURES\n"
        "contenu selection.\n\n"
        "## ARTICLE 7 CRITERES DE JUGEMENT DES OFFRES\n"
        "contenu jugement.\n"
    )
    result = match_section(md)  # no hint — must still find something sensible
    assert result is not None
    assert result["score"] >= 0.55


def test_free_text_hint_not_in_dictionary_still_tries_literal_match():
    md = "## Eléments constitutifs de l'offre\ncontenu ici.\n"
    result = match_section(md, target_hint="Eléments constitutifs de l'offre")
    assert result is not None
    assert "contenu ici." in result["text"]


def test_no_match_returns_none_rather_than_a_weak_guess():
    md = "## Une rubrique complètement hors sujet\ntexte sans rapport.\n"
    result = match_section(md, target_hint="critères de jugement des offres")
    # "critères de jugement des offres" bears no resemblance to this heading —
    # must return None, not a low-confidence guess.
    assert result is None


# ----------------------------------------------------------------------
# Unit-level check on the numeric prefix parser itself, since several
# behaviors above depend on it directly.
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "title,expected",
    [
        ("ARTICLE 7 CRITERES DE JUGEMENT DES OFFRES", (7,)),
        ("7.2 - JUGEMENT DES OFFRES", (7, 2)),
        ("7.2.1 – Jugement des offres", (7, 2, 1)),
        ("1) Valeur technique 65 %", (1,)),
        ("SOMMAIRE", None),
    ],
)
def test_numeric_prefix_parsing(title, expected):
    assert _numeric_prefix(title) == expected


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))