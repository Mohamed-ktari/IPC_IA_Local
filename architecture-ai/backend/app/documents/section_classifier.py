# section_classifier.py
# Classifies a mémoire section heading into a retrieval lane:
#   "boilerplate"       — needs_retrieval: false, skip RAG entirely
#   "memoire"           — generic/recurring, search past mémoires
#   "programme"         — project-specific, search this project's programme
#
# Pipeline (cheapest/most-deterministic check first):
#   1. Small boilerplate dict — exact/fuzzy match (signature, mentions légales)
#   2. Embedding similarity to two centroids (generic vs project-specific),
#      built from real labeled headings extracted from past mémoires
#   3. If ambiguous, one cheap LLM classification call
#
# Standalone module — no dependency on section_matcher.py, so either can
# change independently. Some small helpers are duplicated on purpose.

import json
import re
import difflib
from pathlib import Path

from app.documents.embedder import get_embedder
from app.llm.factory import get_llm
from app.llm.base import Message, Role

BOILERPLATE_DICT_PATH = (
    Path(__file__).resolve().parent / "dictionnaries" / "memoire_boilerplate_dict.json"
)

BOILERPLATE_THRESHOLD = 0.6

# Margin between the two centroid similarities below which we consider the
# result ambiguous and fall back to an LLM call. Needs empirical tuning
# once tested on real headings — this is a starting guess.
AMBIGUITY_MARGIN = 0.05

# --- exemplar headings, drawn from real past-mémoire extraction, labeled
# with your responsable's review. "unsure" and "boilerplate" entries are
# excluded — this list only feeds the two RAG-lane centroids. ---
GENERIC_EXEMPLARS = [
    "Méthode de diagnostic",
    "Organisation et présentation de l'équipe",
    "Une organisation structurée",
    "Le management de la validation",
    "Méthodologie",
    "Approche développement durable",
    "Lutte contre les ilots de chaleur et confort d'été",
    "Utilisation de matériaux de réemploi et de matériaux biosourcés",
    "Organisation du groupement",
    "Répartition des tâches entre les membres de l'équipe",
    "Présentation du groupement",
    "Moyens humains",
    "Moyens matériels",
    "Présentation de l'équipe",
    "Références en milieu occupé",
    "Méthodologie de travail mise en place",
    "Qualité documentaire de la réalisation (DOE)",
    "RSE et développement durable",
    "Démarche d'économie circulaire",
    "Gestion et protection des données",
]

PROJECT_SPECIFIC_EXEMPLARS = [
    "Compréhension du projet",
    "Site",
    "Le bâti",
    "Calendrier prévisionnel",
    "Présentation du site et organisation du bâtiment",
    "Diagnostic visuel de l'existant et pistes d'intervention",
    "Cohérence avec le programme et le budget",
    "Justification des choix techniques en regard du budget alloué",
    "Enjeux de la mission",
    "Analyse du site",
    "Analyse du programme",
    "Analyse structure",
    "Estimatif travaux",
    "Caractéristiques des bâtiments",
    "Compréhension de l'opération",
    "Les enjeux techniques",
    "Planning détaillé",
    "Analyse financière détaillée du budget de la maitrise d'ouvrage",
    "Critique du programme",
    "Prise en compte des locataires",
]


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^\d+([.\d]*)[).\s-]*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _load_boilerplate_dict() -> dict:
    if not BOILERPLATE_DICT_PATH.exists():
        return {}
    return json.loads(BOILERPLATE_DICT_PATH.read_text(encoding="utf-8"))


def _matches_boilerplate(title: str, boilerplate_dict: dict) -> bool:
    norm_title = _normalize(title)
    for entry in boilerplate_dict.values():
        variants = [entry["label"]] + entry.get("variants", [])
        for v in variants:
            ratio = difflib.SequenceMatcher(None, norm_title, _normalize(v)).ratio()
            if ratio >= BOILERPLATE_THRESHOLD:
                return True
    return False


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _average(vectors: list[list[float]]) -> list[float]:
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(dim)]


class SectionClassifier:

    def __init__(self):
        self.embedder = get_embedder()
        self.boilerplate_dict = _load_boilerplate_dict()
        self._generic_centroid: list[float] | None = None
        self._programme_centroid: list[float] | None = None

    def _ensure_centroids(self):
        if self._generic_centroid is not None:
            return
        generic_vecs = self.embedder.embed_batch(GENERIC_EXEMPLARS)
        programme_vecs = self.embedder.embed_batch(PROJECT_SPECIFIC_EXEMPLARS)
        self._generic_centroid = _average(generic_vecs)
        self._programme_centroid = _average(programme_vecs)

    def _llm_classify(self, title: str) -> str:
        llm = get_llm("classification")
        messages = [
            Message(
                role=Role.system,
                content=(
                    "Tu classes des titres de sections de mémoires techniques "
                    "d'architecture/BTP en une seule catégorie : 'memoire' si "
                    "le contenu est générique et réutilisable d'un projet à "
                    "l'autre (équipe, méthodologie générale, moyens, "
                    "références), ou 'programme' si le contenu dépend "
                    "spécifiquement de CE projet (analyse du site, enjeux du "
                    "projet, chiffrage, contraintes spécifiques). Réponds "
                    "uniquement par un seul mot : memoire ou programme."
                ),
            ),
            Message(role=Role.user, content=title),
        ]
        response = llm.chat(messages, temperature=0.0, max_tokens=5)
        answer = response.content.strip().lower()
        return "programme" if "programme" in answer else "memoire"

    def classify(self, title: str) -> dict:
        """
        Returns {"lane": "boilerplate"|"memoire"|"programme", "method": str,
        "confidence": float|None} — method/confidence kept for audit
        traceability, not just the final decision.
        """
        if _matches_boilerplate(title, self.boilerplate_dict):
            return {"lane": "boilerplate", "method": "dict", "confidence": None}

        self._ensure_centroids()
        vec = self.embedder.embed(title)
        sim_generic = _cosine(vec, self._generic_centroid)
        sim_programme = _cosine(vec, self._programme_centroid)

        if abs(sim_generic - sim_programme) < AMBIGUITY_MARGIN:
            lane = self._llm_classify(title)
            return {"lane": lane, "method": "llm_fallback", "confidence": None}

        lane = "memoire" if sim_generic > sim_programme else "programme"
        confidence = abs(sim_generic - sim_programme)
        return {"lane": lane, "method": "embedding", "confidence": round(confidence, 3)}


_classifier: SectionClassifier | None = None

def get_classifier() -> SectionClassifier:
    global _classifier
    if _classifier is None:
        _classifier = SectionClassifier()
    return _classifier