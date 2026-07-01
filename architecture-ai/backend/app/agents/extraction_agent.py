import json
import time
import re
from pathlib import Path

from app.agents.base_agent import BaseAgent, AgentResponse
from app.config import settings
from app.documents.parsers.document_cleaner import DocumentCleaner

DEBUG_DIR = Path("debug_output")

# Generic header fields — not client-specific, safe as a fixed default.
DEFAULT_HEADER_FIELDS = {
    "adresse_immeuble":   "Adresse complète du bâtiment diagnostiqué",
    "reference_dossier":  "Référence ou numéro du dossier/rapport",
    "date_reperage":      "Date du repérage, format DD/MM/YYYY",
    "operateur":          "Nom et qualification de l'opérateur de repérage",
    "laboratoire":        "Nom du laboratoire d'analyse, si mentionné",
}


class ExtractionAgent(BaseAgent):

    agent_type = "extraction"
    description = (
        "Extracts structured data from technical diagnostic reports "
        "into machine-readable JSON, driven entirely by a per-document "
        "column mapping (page range + source→target field mapping) "
        "provided via the team's Excel config."
    )

    def run_extraction(
        self,
        document_text: str,
        debug: bool = False,
        config: dict | None = None,
    ) -> AgentResponse:
        start = time.time()
        config = config or {}

        column_mapping = config.get("column_mapping")
        if not column_mapping:
            raise ValueError(
                "config['column_mapping'] is required — read it from the "
                "Excel via ConfigReader before calling run_extraction()."
            )

        source_columns = list(column_mapping.keys())
        target_fields  = list(column_mapping.values())

        # ── Clean: keep only cover block + configured page range ─────────
        cleaner = DocumentCleaner(debug=debug)
        if debug:
            print(f"[extraction_agent] Cleaner stats: {cleaner.stats(document_text, config=config)}")
        document_text = cleaner.clean(document_text, config=config)
        print(document_text)
        print(f"[extraction_agent] Starting extraction")
        print(f"[extraction_agent] Document (cleaned): {len(document_text.split())} words")
        print(f"[extraction_agent] Column mapping: {column_mapping}")

        # ── Header (generic, client-agnostic) ─────────────────────────────
        print(f"[extraction_agent] Step 1: extracting header...")
        header = self._extract_header(document_text)
        print(f"[extraction_agent] Header: {header}")

        # ── Split ───────────────────────────────────────────────────────
        groups = self._split_into_groups(document_text)
        print(f"[extraction_agent] Step 2: {len(groups)} groups to process")

        # ── Single pass: extract directly into the target schema ─────────
        raw_rows: list[dict] = []
        for i, group_text in enumerate(groups):
            print(f"[extraction_agent] Group {i+1}/{len(groups)}...")
            rows = self._extract_rows_from_group(group_text, source_columns, target_fields)
            raw_rows.extend(rows)
            print(f"[extraction_agent]   +{len(rows)} rows (total: {len(raw_rows)})")

        if debug:
            self._write_debug("pass1_raw.json", {
                "description": "Raw rows, already in target schema, before dedup.",
                "total_rows": len(raw_rows),
                "rows": raw_rows,
            })

        # ── Dedup: exact/near-exact duplicates from group overlap ────────
        # No ID regex, no client-specific logic — duplicates only happen
        # because _split_into_groups() overlaps windows on purpose.
        deduped = self._dedupe_exact(raw_rows)
        print(f"[extraction_agent] After dedup: {len(deduped)} unique rows")

        if debug:
            self._write_debug("pass2_deduped.json", {
                "description": "After exact-duplicate removal — final output.",
                "total_rows": len(deduped),
                "rows": deduped,
            })

        result = {
            "header": header,
            "materiaux": deduped,
            "stats": self._compute_stats(deduped),
            "extraction_config": config,
        }

        duration = round(time.time() - start, 2)
        print(f"[extraction_agent] Done in {duration}s — {result['stats']['total_materiaux']} items")

        return AgentResponse(
            content=json.dumps(result, ensure_ascii=False, indent=2),
            agent_type=self.agent_type,
            model=self.llm.model,
            duration_seconds=duration,
        )

    # ────────────────────────────────────────────────────────────────────
    # Single-pass extraction, output directly in client's target schema
    # ────────────────────────────────────────────────────────────────────

    def _extract_rows_from_group(
        self,
        group_text: str,
        source_columns: list[str],
        target_fields: list[str],
    ) -> list[dict]:
        fields_block = "\n".join(f'  - "{t}"' for t in target_fields)
        source_hint = ", ".join(f'"{c}"' for c in source_columns)
        min_expected = self._estimate_min_rows(group_text)
        schema = {
            "type": "array",
            "minItems": min_expected,
            "items": {
                "type": "object",
                "properties": {f: {"type": ["string", "null"]} for f in target_fields},
                "required": target_fields,
                "additionalProperties": False,
            },
        }
        example_obj_1 = {f: f"<valeur_{i+1}>" for i, f in enumerate(target_fields)}
        example_obj_2 = {f: f"<autre_valeur_{i+1}>" for i, f in enumerate(target_fields)}
        example_array = json.dumps([example_obj_1, example_obj_2], ensure_ascii=False)

        bad_example = {f: [f"<valeur_{i+1}>", f"<autre_valeur_{i+1}>"] for i, f in enumerate(target_fields)}
        bad_example_str = json.dumps(bad_example, ensure_ascii=False)

        prompt = f"""Tu es un expert en lecture de rapports DTA (Dossier Technique Amiante) et RAAT.

MISSION : extraire TOUTES les lignes qui décrivent un matériau ou produit
physique repéré dans le bâtiment. Privilégie le RAPPEL : mieux vaut extraire
trop que manquer une ligne.

Pour chaque ligne trouvée, cherche les informations correspondant à ces
colonnes source (telles qu'elles peuvent apparaître dans le document,
sous un intitulé identique ou très proche) :
  {source_hint}

Et restitue-les UNIQUEMENT sous ces noms de champs exacts en sortie :
{fields_block}

N'ajoute AUCUN champ en dehors de cette liste, même si tu vois d'autres
colonnes dans le document (ex: photo, numéro de page...) — ignore-les.
Si une colonne source est absente pour une ligne donnée, mets sa valeur à null.
N'invente aucune donnée.

RÈGLE DE FORMAT — TRÈS IMPORTANT :
Réponds avec un TABLEAU JSON D'OBJETS — un objet par ligne/matériau trouvé.

  ✅ CORRECT (un objet par ligne) :
  {example_array}

  ❌ INCORRECT (colonnes groupées en listes séparées — NE JAMAIS FAIRE CECI) :
  {bad_example_str}

  ✅ Si un seul résultat : retourne quand même un tableau avec un objet.
  ✅ Si rien trouvé : []

TEXTE :
{group_text}"""
        for attempt in range(2):
            response = self.chat(
                user_message=prompt,
                temperature=0.0,
                max_tokens=settings.DEFAULT_MAX_TOKENS,
                json_schema=schema,
            )
            parsed = self._safe_parse_json(response.content, default=[])
            rows = self._to_list(parsed, expected_keys=set(target_fields))
            if len(rows) >= min_expected:
                print(f"[extraction_agent] RAW response group {response.content[:1500]}")
                return rows
            print(f"[extraction_agent] Got {len(rows)}/{min_expected} expected rows, retrying...")
            print(f"[extraction_agent] RAW response group {response.content[:1500]}")
        
        print(f"[extraction_agent] Warning: only {len(rows)}/{min_expected} rows after retry — keeping partial result")
        return rows

    # ────────────────────────────────────────────────────────────────────
    # Dedup — content-based, no client-specific ID assumptions
    # ────────────────────────────────────────────────────────────────────

    def _dedupe_exact(self, rows: list[dict]) -> list[dict]:
        seen = set()
        out = []
        for row in rows:
            key = tuple(sorted(
                (k, str(v).strip().lower())
                for k, v in row.items()
                if v not in (None, "", "null")
            ))
            if not key:
                continue  # fully empty row, skip
            if key not in seen:
                seen.add(key)
                out.append(row)
        return out

    # ────────────────────────────────────────────────────────────────────
    # Splitting (unchanged — page-marker aware, with table-header context)
    # ────────────────────────────────────────────────────────────────────

    def _split_into_groups(self, text: str) -> list[str]:
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

    # ────────────────────────────────────────────────────────────────────
    # Header — generic schema, client-agnostic
    # ────────────────────────────────────────────────────────────────────

    def _extract_header(self, text: str) -> dict:
        first_section = " ".join(text.split()[:1000])
        schema = json.dumps(DEFAULT_HEADER_FIELDS, ensure_ascii=False, indent=2)

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

    # ────────────────────────────────────────────────────────────────────
    # Stats — only counts presence/absence if a "resultat" field exists
    # in the client's mapping; otherwise just reports totals.
    # ────────────────────────────────────────────────────────────────────

    def _compute_stats(self, materiaux: list[dict]) -> dict:
        if not materiaux or "resultat" not in materiaux[0]:
            return {"total_materiaux": len(materiaux)}

        presence = absence = unknown = 0
        for m in materiaux:
            val = str(m.get("resultat") or "").lower()
            if "présence" in val or "presence" in val or re.search(r'\b(ep|ac1|ac2)\b', val):
                presence += 1
            elif "absence" in val or "non détecté" in val or "non detecte" in val:
                absence += 1
            else:
                unknown += 1
        return {
            "total_materiaux":  len(materiaux),
            "presence_amiante": presence,
            "absence_amiante":  absence,
            "resultat_inconnu": unknown,
        }

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────

    def _to_list(self, parsed, expected_keys: set) -> list[dict]:
        if isinstance(parsed, list):
            if not expected_keys:
                return [item for item in parsed if isinstance(item, dict)]
            return [
                item for item in parsed
                if isinstance(item, dict) and any(k in item for k in expected_keys)
            ]

        if isinstance(parsed, dict):
            # Case A: flat dict, no nested values = a single row object
            if not any(isinstance(v, (list, dict)) for v in parsed.values()):
                return [parsed]

            # Case B: columnar/transposed dict — {"field1": [...], "field2": [...]}
            # All values are lists of scalars (not list of dicts) and roughly
            # the same length. This is what Qwen just gave you.
            list_values = {k: v for k, v in parsed.items() if isinstance(v, list)}
            if list_values and all(
                all(not isinstance(item, (list, dict)) for item in v)
                for v in list_values.values()
            ):
                lengths = {len(v) for v in list_values.values()}
                if len(lengths) == 1:  # all columns same length — safe to transpose
                    n = lengths.pop()
                    rows = []
                    for i in range(n):
                        row = {k: v[i] for k, v in list_values.items()}
                        rows.append(row)
                    print(f"[extraction_agent] Transposed columnar response into {n} rows")
                    return rows
                else:
                    print(
                        f"[extraction_agent] Warning: columnar response has mismatched "
                        f"column lengths {[(k, len(v)) for k, v in list_values.items()]} "
                        f"— cannot safely transpose, dropping"
                    )
                    return []

            # Case C: wrapper like {"rows": [...]}
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

    def _write_debug(self, filename: str, data: dict) -> None:
        DEBUG_DIR.mkdir(exist_ok=True)
        path = DEBUG_DIR / filename
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[extraction_agent] Debug file written: {path}")



    def _estimate_min_rows(self, group_text: str) -> int:
        # Conservative floor: count unique row-identifier-looking tokens.
        # Works for any client where rows are tagged Mxxx / ZPSO-xxx / etc,
        # falls back to 1 if nothing recognizable.
        ids = set(re.findall(r'\b[A-Z]{1,5}-?\d{2,5}\b', group_text))
        return max(len(ids), 1)



    def run(self, *args, **kwargs) -> AgentResponse:
        return self.run_extraction(*args, **kwargs)