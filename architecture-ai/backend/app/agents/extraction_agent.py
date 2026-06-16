# extraction_agent.py
import json
import time
import re
from pathlib import Path

from app.agents.base_agent import BaseAgent, AgentResponse
from app.config import settings

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


class ExtractionAgent(BaseAgent):

    agent_type = "extraction"
    description = (
        "Extracts structured data from technical diagnostic reports "
        "into machine-readable JSON following company schemas"
    )

    def run_extraction(
        self,
        document_text: str,
        template_name: str = "amiante",
    ) -> AgentResponse:
        start = time.time()
        template = self._load_template(template_name)

        print(f"[extraction_agent] Starting extraction with template: {template['name']}")
        print(f"[extraction_agent] Document: {len(document_text.split())} words")

        print(f"[extraction_agent] Step 1: extracting header...")
        header = self._extract_header(document_text, template)
        print(f"[extraction_agent] Header: {header}")

        groups = self._split_into_groups(document_text)
        print(f"[extraction_agent] Step 2: {len(groups)} page groups to process")

        # No ID-based dedup. The same material description (e.g. "Enduits projetés")
        # legitimately repeats across different locations/apartments in the building,
        # so every occurrence found in every group is kept as-is. A human reviews the
        # final list afterward, so duplicates (including ones caused purely by the
        # overlap window between adjacent groups) are an acceptable cost here.
        all_materiaux = []

        for i, group_text in enumerate(groups):
            print(f"[extraction_agent] Processing group {i+1}/{len(groups)}...")
            materiaux = self._extract_materiaux_from_group(
                group_text, template, group_index=i
            )
            all_materiaux.extend(materiaux)
            print(f"[extraction_agent] Group {i+1}: found {len(materiaux)} matériaux "
                  f"(total so far: {len(all_materiaux)})")

        result = {
            "template": template["name"],
            "header": header,
            "materiaux": all_materiaux,
            "stats": {
                "total_materiaux": len(all_materiaux),
                "presence_amiante": sum(
                    1 for m in all_materiaux
                    if "PRÉSENCE" in self._get_resultat(m).upper()
                ),
                "absence_amiante": sum(
                    1 for m in all_materiaux
                    if "ABSENCE" in self._get_resultat(m).upper()
                ),
            }
        }

        duration = round(time.time() - start, 2)
        print(f"[extraction_agent] Done in {duration}s — {len(all_materiaux)} extracted")

        return AgentResponse(
            content=json.dumps(result, ensure_ascii=False, indent=2),
            agent_type=self.agent_type,
            model=self.llm.model,
            duration_seconds=duration,
        )

    def _split_into_groups(self, text: str) -> list[str]:
        words = text.split()
        group_size = settings.EXTRACTION_WORDS_PER_PAGE * settings.EXTRACTION_PAGES_PER_GROUP
        overlap = settings.EXTRACTION_WORDS_PER_PAGE

        groups = []
        i = 0
        while i < len(words):
            group_words = words[i:i + group_size]
            groups.append(" ".join(group_words))
            i += group_size - overlap

        return groups

    def _extract_header(self, text: str, template: dict) -> dict:
        first_section = " ".join(text.split()[:1000])
        schema = json.dumps(template["header_fields"], ensure_ascii=False, indent=2)

        prompt = f"""Tu es un expert en extraction de données de documents techniques.

Extrait les informations d'en-tête de ce début de rapport.

SCHÉMA JSON À REMPLIR (réponds UNIQUEMENT avec le JSON, sans texte autour) :
{schema}

RÈGLES :
- Si une information est absente, mets null
- Pour la date, format DD/MM/YYYY
- Réponds uniquement avec un objet JSON valide

TEXTE :
{first_section}"""

        response = self.chat(
            user_message=prompt,
            temperature=0.0,
            max_tokens=500,
            json_mode=True,
        )
        return self._safe_parse_json(response.content, default={})

    def _extract_materiaux_from_group(
        self,
        group_text: str,
        template: dict,
        group_index: int,
    ) -> list[dict]:
        materiau_schema = template["materiau_schema"]
        empty_item = {k: None for k in materiau_schema.keys()}

        prompt = f"""Tu es un expert en extraction de données de rapports DTA (Dossier Technique Amiante).

Extrait UNIQUEMENT les lignes décrivant des matériaux ou produits physiques repérés dans le bâtiment.

UN MATÉRIAU VALIDE SE RECONNAÎT PAR :
- Une description physique d'un matériau de construction (ex: "Flocages", "Enduits projetés", 
  "Volets coupe-feu", "Dalles de sol", "Calorifugeages", "Panneaux collés", "Joints")
- Une localisation dans le bâtiment (étage, local, zone)
- Un résultat d'analyse (présence ou absence d'amiante)

NE PAS EXTRAIRE — ces lignes ne sont PAS des matériaux :
- Lignes de catégorie (LISTE A, LISTE B, Flocages/Calorifugeages...)
- Lignes de suivi ou d'évaluation périodique sans matériau identifié
- Données administratives (références de dossier, contacts, dates seules)
- Lignes dont la description est vide ou ne correspond pas à un matériau physique

SCHÉMA D'UN ÉLÉMENT (utilise EXACTEMENT ces clés) :
{json.dumps(empty_item, ensure_ascii=False, indent=2)}

DESCRIPTION DES CHAMPS :
{json.dumps(materiau_schema, ensure_ascii=False, indent=2)}

RÈGLES :
- Réponds UNIQUEMENT avec un tableau JSON
- Si aucun matériau valide n'est présent, réponds avec : []
- Pour les champs absents, utilise null
- Ne génère pas de données fictives

TEXTE :
{group_text}"""
        response = self.chat(
            user_message=prompt,
            temperature=0.0,
            max_tokens=settings.DEFAULT_MAX_TOKENS,
            json_mode=True,
        )

        parsed = self._safe_parse_json(response.content, default=[])
        return self._to_list(parsed, expected_keys=set(materiau_schema.keys()))

    def _to_list(self, parsed, expected_keys: set) -> list[dict]:
        if isinstance(parsed, list):
            return [
                item for item in parsed
                if isinstance(item, dict)
                and any(k in item for k in expected_keys)
            ]

        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list) and len(v) > 0:
                    candidates = [
                        item for item in v
                        if isinstance(item, dict)
                        and any(k in item for k in expected_keys)
                    ]
                    if candidates:
                        return candidates
            return []

        return []

    def _get_resultat(self, m: dict) -> str:
        # Canonical path: the prompt now forces "resultat" to be exactly
        # "presence" or "absence". Everything below is just a safety net for
        # older templates or any single entry that still drifts from the schema.
        raw = str(m.get("resultat") or "").strip().lower()

        if raw in ("presence", "présence"):
            return "PRÉSENCE AMIANTE"
        if raw == "absence":
            return "Absence d'amiante"

        fallback = (
            m.get("conclusion") or
            m.get("presence_of_asbestos") or
            m.get("conclusion_justification") or
            m.get("analysis_result") or
            ""
        )
        blob = (raw + " " + str(fallback)).lower()
        if "présence" in blob or "presence" in blob:
            return "PRÉSENCE AMIANTE"
        if "absence" in blob:
            return "Absence d'amiante"
        if re.search(r'\b(ep|ac1|ac2)\b', blob):
            return "PRÉSENCE AMIANTE"
        return raw

    def _safe_parse_json(self, text: str, default):
        text = text.strip()
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        array_match = re.search(r'\[.*\]', text, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group())
            except json.JSONDecodeError:
                pass

        obj_match = re.search(r'\{.*\}', text, re.DOTALL)
        if obj_match:
            try:
                parsed = json.loads(obj_match.group())
                if isinstance(parsed, dict):
                    for v in parsed.values():
                        if isinstance(v, list):
                            return v
            except json.JSONDecodeError:
                pass

        print(f"[extraction_agent] Warning: could not parse JSON from: {text[:200]}")
        return default

    def _load_template(self, template_name: str) -> dict:
        template_file = TEMPLATES_DIR / f"{template_name}.json"
        if not template_file.exists():
            available = [f.stem for f in TEMPLATES_DIR.glob("*.json")]
            raise ValueError(
                f"Template '{template_name}' not found. "
                f"Available: {', '.join(available)}"
            )
        return json.loads(template_file.read_text(encoding="utf-8"))

    def run(self, *args, **kwargs) -> AgentResponse:
        return self.run_extraction(*args, **kwargs)