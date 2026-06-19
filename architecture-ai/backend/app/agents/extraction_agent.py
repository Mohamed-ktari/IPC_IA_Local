import json
import time
import re
from pathlib import Path

from app.agents.base_agent import BaseAgent, AgentResponse
from app.config import settings
from app.documents.parsers.document_cleaner import DocumentCleaner

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

DEBUG_DIR = Path("debug_output")


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
        debug: bool = False,
    ) -> AgentResponse:
        start = time.time()
        template = self._load_template(template_name)

        # ── Clean: keep only cover block + section 5 ────────────────────
        cleaner = DocumentCleaner(debug=debug)
        if debug:
            print(f"[extraction_agent] Cleaner stats: {cleaner.stats(document_text)}")
        document_text = cleaner.clean(document_text)

        print(f"[extraction_agent] Starting extraction — template: {template['name']}")
        print(f"[extraction_agent] Document (cleaned): {len(document_text.split())} words")

        # ── Header ──────────────────────────────────────────────────────
        print(f"[extraction_agent] Step 1: extracting header...")
        header = self._extract_header(document_text, template)
        print(f"[extraction_agent] Header: {header}")

        # ── Split ───────────────────────────────────────────────────────
        groups = self._split_into_groups(document_text)
        print(f"[extraction_agent] Step 2: {len(groups)} groups to process")

        # ── Pass 1 — free-form extraction ───────────────────────────────
        raw_materiaux: list[dict] = []
        for i, group_text in enumerate(groups):
            print(f"[extraction_agent] Pass-1 group {i+1}/{len(groups)}...")
            raw = self._extract_raw_from_group(group_text, group_index=i)
            raw_materiaux.extend(raw)
            print(f"[extraction_agent]   +{len(raw)} rows (total raw: {len(raw_materiaux)})")

        if debug:
            self._write_debug("pass1_raw.json", {
                "description": "Pass-1 output: verbatim rows before any processing.",
                "total_rows": len(raw_materiaux),
                "rows": raw_materiaux,
            })

        # ── Pre-processing: regex extraction + deduplication ─────────────
        # 1. Parse deterministic fields (id, refs, resultat) from
        #    the embedded Description string — faster and more reliable
        #    than asking the LLM to do it.
        # 2. Deduplicate by material ID — section 5 produces 2-3 rows
        #    per material (one from 5.1 detail table, one from 5.2 recap).
        #    We keep the richest row per ID.
        print(f"[extraction_agent] Step 3: pre-processing {len(raw_materiaux)} raw rows...")
        remapped = self._remap_col_rows(raw_materiaux)
        enriched = [self._parse_known_fields(row) for row in remapped]
        deduped  = self._deduplicate_raw(enriched)
        print(f"[extraction_agent] After dedup: {len(deduped)} unique materials")

        if debug:
            self._write_debug("pass1_enriched_deduped.json", {
                "description": (
                    "After regex extraction of id/refs/resultat "
                    "and deduplication by material ID."
                ),
                "total_rows": len(deduped),
                "rows": deduped,
            })

        # ── Pass 2 — normalisation ───────────────────────────────────────
        print(f"[extraction_agent] Step 4: normalising {len(deduped)} rows...")
        all_materiaux = self._normalise_batch(deduped, template)
        print(f"[extraction_agent] Normalised: {len(all_materiaux)} items")

        if debug:
            self._write_debug("pass2_normalised.json", {
                "description": "Pass-2 output: canonical schema with others field.",
                "total_rows": len(all_materiaux),
                "rows": all_materiaux,
            })

        # ── Stats ────────────────────────────────────────────────────────
        result = {
            "template": template["name"],
            "header": header,
            "materiaux": all_materiaux,
            "stats": self._compute_stats(all_materiaux),
        }

        duration = round(time.time() - start, 2)
        print(f"[extraction_agent] Done in {duration}s — "
              f"{result['stats']['total_materiaux']} items | "
              f"présence: {result['stats']['presence_amiante']} | "
              f"absence: {result['stats']['absence_amiante']}")

        return AgentResponse(
            content=json.dumps(result, ensure_ascii=False, indent=2),
            agent_type=self.agent_type,
            model=self.llm.model,
            duration_seconds=duration,
        )

    # ────────────────────────────────────────────────────────────────────────
    # Pre-processing — regex extraction of deterministic fields
    # ────────────────────────────────────────────────────────────────────────

    # Regex patterns — compiled once at class level for performance
    _RE_ID = re.compile(
        r'(?:Identifiant\s*[:\s]+|Zone\s*[:\s]+)?\b(M\d{3,4})\b',
        re.IGNORECASE,
    )
    _RE_REF_ECH = re.compile(
        r'(?:R[ée]f\.?\s*[ée]chantillon\s*[:\s]+|[ée]chantillon(?:s)?\s*[:\s]+)([\w/\-\.]+)',
        re.IGNORECASE,
    )
    _RE_REF_LABO = re.compile(
        r'R[ée]f\.?\s*(?:de\s+)?laboratoire\s*[:\s]+([\w/\-\.]+)',
        re.IGNORECASE,
    )
    _RE_RESULTAT_PRESENCE = re.compile(
        r'pr[ée]sence\s+d\'?amiante|PRÉSENCE|presence|\bEP\b|\bAC[12]\b',
        re.IGNORECASE,
    )
    _RE_RESULTAT_ABSENCE = re.compile(
        r'absence\s+d\'?amiante|non\s+d[ée]tect[ée]|ABSENCE',
        re.IGNORECASE,
    )

    def _parse_known_fields(self, row: dict) -> dict:
        """
        Runs regex over all string values in a raw row to extract
        deterministic fields: id, reference_echantillon, reference_labo,
        resultat.

        These are set directly on the row so Pass 2 normalisation
        receives pre-filled canonical keys and only needs to handle
        the semantic/variable fields (localisation, composant, others...).

        The original keys are preserved so Pass 2 still has full context.
        """
        # Concatenate all string values for a single-pass search
        blob = " ".join(str(v) for v in row.values() if v)

        row = dict(row)  # shallow copy — don't mutate caller's data

        # ── id ──────────────────────────────────────────────────────────
        if not row.get("id"):
            # Prefer 'Zone' key (5.2 rows) — most reliable source
            zone_val = row.get("Zone") or row.get("zone")
            if zone_val and re.match(r'^M\d{3,4}$', str(zone_val).strip()):
                row["id"] = str(zone_val).strip()
            else:
                m = self._RE_ID.search(blob)
                if m:
                    row["id"] = m.group(1).upper()

        # ── reference_echantillon ────────────────────────────────────────
        if not row.get("reference_echantillon"):
            # First try explicit "N° Echantillon" column (RAAT format)
            nech = row.get("N° Echantillon") or row.get("N° echantillon")
            if nech and str(nech).strip():
                row["reference_echantillon"] = str(nech).strip()
            else:
                m = self._RE_REF_ECH.search(blob)
                if m:
                    ref = m.group(1).strip().rstrip(".,;")
                    # Sanity check: skip generic words like "Oui", "Non"
                    if ref.lower() not in ("oui", "non", "aucun"):
                        row["reference_echantillon"] = ref

        # ── reference_labo ───────────────────────────────────────────────
        if not row.get("reference_labo"):
            m = self._RE_REF_LABO.search(blob)
            if m:
                row["reference_labo"] = m.group(1).strip().rstrip(".,;")

        # ── resultat ─────────────────────────────────────────────────────
        # Only set if not already a canonical value from a previous pass
        current_res = str(row.get("resultat") or "").strip().lower()
        if current_res not in ("presence", "absence"):
            if self._RE_RESULTAT_PRESENCE.search(blob):
                row["resultat"] = "presence"
            elif self._RE_RESULTAT_ABSENCE.search(blob):
                row["resultat"] = "absence"
            # else: leave as-is, Pass 2 will try to resolve

        return row

    # ────────────────────────────────────────────────────────────────────────
    # Pre-processing — deduplication by material ID
    # ────────────────────────────────────────────────────────────────────────

    def _deduplicate_raw(self, rows: list[dict]) -> list[dict]:
        """
        Keeps the single richest row per material ID.

        Scoring (higher = preferred):
          +3  row has a 'Zone' key  (5.2 recap — cleanest structure)
          +2  row has named columns like 'Localisation', 'Description'
          +1  row has col_N columns (5.1 detail — more fields but noisier)
          +N  one point per non-empty value (rewards information density)

        Rows without any detectable ID are kept as-is (rare edge cases).
        """
        by_id: dict[str, dict] = {}
        no_id: list[dict] = []

        for row in rows:
            mat_id = row.get("id")
            if not mat_id:
                no_id.append(row)
                continue

            score = self._row_score(row)

            if mat_id not in by_id:
                row["_score"] = score
                by_id[mat_id] = row
            else:
                existing_score = by_id[mat_id].get("_score", 0)
                if score > existing_score:
                    row["_score"] = score
                    by_id[mat_id] = row

        # Strip internal scoring key before returning
        result = []
        for row in by_id.values():
            clean = {k: v for k, v in row.items() if k != "_score"}
            result.append(clean)

        # Sort by ID so output is deterministic and easy to read
        result.sort(key=lambda r: r.get("id", ""))

        if no_id:
            print(f"[extraction_agent] {len(no_id)} rows had no detectable ID "
                  f"and were dropped during deduplication")

        return result

    def _row_score(self, row: dict) -> int:
        """Assigns a quality score to a raw row for deduplication."""
        score = 0
        keys = set(row.keys())

        if "Zone" in keys:
            score += 3
        elif "Localisation" in keys or "localisation" in keys:
            score += 2
        elif any(k.startswith("col_") for k in keys):
            score += 1

        # Reward information density
        score += sum(1 for v in row.values() if v and str(v).strip())

        return score

    # ────────────────────────────────────────────────────────────────────────
    # Splitting
    # ────────────────────────────────────────────────────────────────────────

    def _split_into_groups(self, text: str) -> list[str]:
        """
        Prefer docling page markers (\f or <!-- page N -->).
        Fallback: word-window with overlap.
        Each group is prefixed with the last table-header row seen so far.
        """
        page_splits = re.split(r'\f|<!-- page \d+ -->', text)
        if len(page_splits) > 1:
            pages = [p.strip() for p in page_splits if p.strip()]
        else:
            words = text.split()
            group_size = settings.EXTRACTION_WORDS_PER_PAGE * settings.EXTRACTION_PAGES_PER_GROUP
            overlap    = settings.EXTRACTION_WORDS_PER_PAGE
            pages = []
            i = 0
            while i < len(words):
                pages.append(" ".join(words[i:i + group_size]))
                i += group_size - overlap
            return pages

        pgs_per_group = getattr(settings, "EXTRACTION_PAGES_PER_GROUP", 3)
        groups = []
        last_header = ""
        for i in range(0, len(pages), pgs_per_group):
            chunk = "\n\n".join(pages[i:i + pgs_per_group])
            m = re.search(r'^(?:[^\n|]*\|){3,}[^\n]*$', chunk, re.MULTILINE)
            if m:
                last_header = m.group(0).strip()
            prefix = (f"[CONTEXTE — EN-TÊTE DE TABLEAU DE LA PAGE PRÉCÉDENTE] : "
                      f"{last_header}\n\n") if last_header else ""
            groups.append(prefix + chunk)

        return groups

    # ────────────────────────────────────────────────────────────────────────
    # Pass 1 — free-form extraction
    # ────────────────────────────────────────────────────────────────────────

    def _extract_raw_from_group(
        self,
        group_text: str,
        group_index: int,
    ) -> list[dict]:
        """
        No schema imposed. Model uses source column names verbatim.
        Pure recall: better to over-extract than miss rows.
        """
        prompt = f"""Tu es un expert en lecture de rapports DTA (Dossier Technique Amiante) et RAAT (Repérage Amiante Avant Travaux).

MISSION : extraire TOUTES les lignes qui décrivent un matériau ou produit physique
repéré dans le bâtiment. Privilégie le RAPPEL : mieux vaut extraire trop que manquer
une ligne.

INCLURE — si la ligne contient AU MOINS 2 de ces éléments :
  • description d'un matériau de construction
    (flocage, dalle, joint, enduit, volet coupe-feu, calorifugeage, panneau, conduit…)
  • localisation dans le bâtiment (étage, local, couloir, cage d'escalier…)
  • résultat d'analyse (présence, absence, EP, AC1, AC2…)
  • référence d'échantillon ou de laboratoire
  • identifiant de type M001, M002…

EXCLURE — ne PAS extraire :
  • titres de sections / catégories pures sans localisation ni résultat
  • lignes 100 % administratives sans matériau identifié
  • lignes totalement vides
  • la liste normative NF X 46-020 (liste de composants génériques sans résultat)

FORMAT DE SORTIE :
  • Tableau JSON uniquement ([] si rien à extraire)
  • Conserve les clés EXACTES du tableau source (ne traduis pas, ne renomme pas)
  • Si une colonne n'a pas de libellé clair → "col_1", "col_2"…
  • Ne génère aucune donnée inventée

TEXTE :
{group_text}"""

        response = self.chat(
            user_message=prompt,
            temperature=0.0,
            max_tokens=settings.DEFAULT_MAX_TOKENS,
            json_mode=True,
        )
        parsed = self._safe_parse_json(response.content, default=[])
        return self._to_list(parsed, expected_keys=set())

    # ────────────────────────────────────────────────────────────────────────
    # Pass 2 — normalisation
    # ────────────────────────────────────────────────────────────────────────

    def _normalise_batch(
    self,
    raw_items: list[dict],
    template: dict,
) -> list[dict]:
        if not raw_items:
            return []

        schema     = template["materiau_schema"]
        empty_item = {k: None for k in schema.keys()}
        return self._normalise_per_item(raw_items, schema, empty_item)

    def _normalise_single_batch(
        self,
        batch: list[dict],
        schema: dict,
        empty_item: dict,
    ) -> list[dict]:
        prompt = f"""Tu es un expert en normalisation de données de rapports DTA et RAAT.

Tu reçois {len(batch)} items déjà pré-traités : les champs "id", "resultat",
"reference_echantillon" et "reference_labo" ont déjà été extraits par regex
quand ils étaient présents — NE LES MODIFIE PAS s'ils sont déjà remplis.

MISSION : convertir CHAQUE item vers le schéma cible. Tu dois retourner
exactement {len(batch)} items.

SCHÉMA CIBLE (utilise EXACTEMENT ces clés) :
{json.dumps(empty_item, ensure_ascii=False, indent=2)}

DESCRIPTION DE CHAQUE CHAMP :
{json.dumps(schema, ensure_ascii=False, indent=2)}

RÈGLES DE MAPPING :

1. "id" → si déjà rempli (ex: "M001"), CONSERVER tel quel sans modification.
   Sinon chercher un identifiant de type M001 dans les données source.

2. "resultat" → si déjà rempli ("presence" ou "absence"), CONSERVER tel quel.
   Sinon mapper OBLIGATOIREMENT vers l'une des deux valeurs exactes :
     "presence"  si : présence / PRÉSENCE / EP / AC1 / AC2 / Sur décision / positif
     "absence"   si : absence / ABSENCE / non détecté / aucune fibre
     null        si vraiment aucune information disponible

3. "reference_echantillon" → si déjà rempli, CONSERVER.
   Sinon extraire la référence d'échantillon (ex: P001, 24/ABD/15322/ROC/M001-P001).

4. "reference_labo" → si déjà rempli, CONSERVER.
   Sinon extraire la référence laboratoire.

5. "localisation" → la localisation dans le bâtiment.
   Ex: "Sous-Sol -2 - ORIENT - Couloir/Dégag", "Entrée", "Salle d'eau".
   Pour les 5.2 (Zone rows), chercher dans "Identifiant + Description".

6. "description_materiau" → la description physique du matériau uniquement.
   Ex: "Flocages", "Enduit à base de ciment + peinture (mur de circulation)".
   NE PAS inclure la localisation ou le résultat dans ce champ.

7. "composant" → la catégorie NF X 46-020 si présente.
   Ex: "Flocages, Calorifugeages, Faux plafonds", "Clapets / volets coupe-feu".
   null si absent.

8. "etat_conservation" → code brut (EP, AC1, AC2, "-") ou null.

9. "preconisation" → normalise vers l'une de ces valeurs si possible :
     "Aucune action requise"
     "Évaluation périodique"
     "Action corrective 1er niveau"
     "Action corrective 2nd niveau"
   null si absent.

10. "others" → objet JSON contenant TOUS les champs présents dans la source
    qui ne correspondent à aucun des champs du schéma ci-dessus.
    EXEMPLES de champs qui vont dans others :
      - "Partie à sonder", "Sondage", "Localisation sur croquis"
      - "n° de sondage", "Echantillon" (la colonne Oui/Non pas la référence)
      - "tour", "bâtiment", tout champ spécifique à l'entreprise
    null si vraiment rien à capturer.
    IMPORTANT : ne jamais perdre d'information — si elle n'a pas de champ
    dédié dans le schéma, elle va dans others.

DONNÉES SOURCE :
{json.dumps(batch, ensure_ascii=False, indent=2)}

Réponds UNIQUEMENT avec un tableau JSON de {len(batch)} objets normalisés.
Pas de texte avant ou après. Pas de markdown."""

        response = self.chat(
        user_message=prompt,
        temperature=0.0,
        max_tokens=settings.DEFAULT_MAX_TOKENS,
        json_mode=True,
        )
        parsed = self._safe_parse_json(response.content, default=[])
        result = self._to_list(parsed, expected_keys=set(empty_item.keys()))

        if not result:
            print(f"[extraction_agent] Warning: normalisation failed for item — keeping raw")
            return batch

        return result





    def _normalise_per_item(self, batch: list[dict], schema: dict, empty_item: dict) -> list[dict]:
        results = []
        for item in batch:
            single_result = self._normalise_single_batch([item], schema, empty_item)
            results.extend(single_result if single_result else [item])
        return results

    # ────────────────────────────────────────────────────────────────────────
    # Header
    # ────────────────────────────────────────────────────────────────────────

    def _extract_header(self, text: str, template: dict) -> dict:
        first_section = " ".join(text.split()[:1000])
        schema = json.dumps(template["header_fields"], ensure_ascii=False, indent=2)

        prompt = f"""Tu es un expert en extraction de données de documents techniques.

Extrait les informations d'en-tête de ce début de rapport.

SCHÉMA JSON À REMPLIR :
{schema}

RÈGLES :
- Si une information est absente, mets null
- Pour la date, format DD/MM/YYYY
- Réponds uniquement avec un objet JSON valide, sans texte autour

TEXTE :
{first_section}"""

        response = self.chat(
            user_message=prompt,
            temperature=0.0,
            max_tokens=500,
            json_mode=True,
        )
        return self._safe_parse_json(response.content, default={})

    # ────────────────────────────────────────────────────────────────────────
    # Stats
    # ────────────────────────────────────────────────────────────────────────

    def _compute_stats(self, materiaux: list[dict]) -> dict:
        presence = 0
        absence  = 0
        unknown  = 0
        for m in materiaux:
            r = self._get_resultat(m)
            if "PRÉSENCE" in r.upper():
                presence += 1
            elif "ABSENCE" in r.upper():
                absence += 1
            else:
                unknown += 1
        return {
            "total_materiaux":  len(materiaux),
            "presence_amiante": presence,
            "absence_amiante":  absence,
            "resultat_inconnu": unknown,
        }

    # ────────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────────

    def _get_resultat(self, m: dict) -> str:
        raw = str(m.get("resultat") or "").strip().lower()

        if raw == "presence":
            return "PRÉSENCE AMIANTE"
        if raw == "absence":
            return "Absence d'amiante"

        # Fallback for rows where normalisation failed
        fallback = " ".join(filter(None, [
            str(m.get("conclusion") or ""),
            str(m.get("resultat_analyse") or ""),
            str(m.get("Conclusion (justification)") or ""),
            str(m.get("analysis_result") or ""),
        ])).lower()
        blob = raw + " " + fallback
        if "présence" in blob or "presence" in blob:
            return "PRÉSENCE AMIANTE"
        if "absence" in blob:
            return "Absence d'amiante"
        if re.search(r'\b(ep|ac1|ac2)\b', blob):
            return "PRÉSENCE AMIANTE"
        return raw

    def _to_list(self, parsed, expected_keys: set) -> list[dict]:
        if isinstance(parsed, dict) and ("id" in parsed or any(k in parsed for k in expected_keys)):
            return [parsed]
        if isinstance(parsed, list):
            if not expected_keys:
                return [item for item in parsed if isinstance(item, dict)]
            return [
                item for item in parsed
                if isinstance(item, dict) and any(k in item for k in expected_keys)
            ]
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    candidates = [
                        i for i in v
                        if isinstance(i, dict)
                        and (not expected_keys or any(k in i for k in expected_keys))
                    ]
                    if candidates:
                        return candidates
        return []

    def _safe_parse_json(self, text: str, default):
        text = re.sub(r'```json\s*', '', text.strip())
        text = re.sub(r'```\s*', '', text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        for pattern in (r'\[.*\]', r'\{.*\}'):
            m = re.search(pattern, text, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                    if isinstance(parsed, dict):
                        for v in parsed.values():
                            if isinstance(v, list):
                                return v
                    return parsed
                except json.JSONDecodeError:
                    pass
        print(f"[extraction_agent] Warning: could not parse JSON: {text[:200]}")
        return default


    def _remap_col_rows(self, rows: list[dict]) -> list[dict]:
        result = []
        for row in rows:
            col_keys = [k for k in row if re.match(r'^col_\d+$', k)]
            if not col_keys:
                result.append(row)
                continue
            result.append(self._infer_col_mapping(row, col_keys))
        return result

    def _infer_col_mapping(self, row: dict, col_keys: list) -> dict:
        # Sort col_1, col_2... in order
        sorted_cols = sorted(col_keys, key=lambda k: int(k.split('_')[1]))
        values = [row[k] for k in sorted_cols]
        
        mapped = {}
        for val in values:
            s = str(val).strip()
            # M001 pattern → id
            if re.match(r'^M\d{3,4}$', s):
                mapped['id'] = s
            # "Partie à inspecter" → composant hint, skip
            elif s.startswith('Partie à inspecter'):
                mapped['_partie_inspecter'] = s
            # Pxxx pattern → reference_echantillon
            elif re.match(r'^P\d{3,4}$', s):
                mapped['reference_echantillon'] = s
            # "Identifiant :Mxxx..." → the description blob
            elif 'Identifiant' in s or 'Résultat' in s:
                mapped['_description_blob'] = s
            # "Oui ..." or "Aucun prélèvement" → echantillon flag
            elif s.startswith('Oui') or 'prélèvement' in s.lower():
                mapped['_echantillon_flag'] = s
            # Pure digit → n° de sondage
            elif re.match(r'^\d+$', s):
                mapped['n° de sondage'] = s
            # Remaining string → localisation candidate
            elif s:
                mapped.setdefault('Localisation', s)
        
        # Preserve originals for Pass 2 context
        for k in col_keys:
            mapped[k] = row[k]
        
        return mapped

    def _write_debug(self, filename: str, data: dict) -> None:
        DEBUG_DIR.mkdir(exist_ok=True)
        path = DEBUG_DIR / filename
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[extraction_agent] Debug file written: {path}")

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