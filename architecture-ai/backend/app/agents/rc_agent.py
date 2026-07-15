# rc_agent.py
# Feature — RC (règlement de consultation) analysis.
#
# Given the parsed markdown of an RC and a free-text user prompt, produces:
#   1. A general summary of the whole document (map-reduce over ~15 pages),
#      optionally weighted toward whatever the user asked to emphasize.
#   2. The extracted "mémoire technique" structure — found either via a
#      section name the user named explicitly, or via a fuzzy match against
#      a dictionary of known section-name variants (structure de mémoire,
#      critères de jugement, critères de sélection, ...).
#
# No RAG involved: the summary needs the whole document, and the structure
# is a single structural element found via heading match, not semantic
# retrieval. See section_matcher.py for why.
#
# This agent returns structured data only — it does NOT write the Word
# file itself. That's the caller's/service layer's job.

import json
import re
import time
from pathlib import Path

from app.agents.base_agent import BaseAgent, AgentResponse
from app.documents.chunker import Chunker
from app.documents.section_matcher import match_section, load_sections_dict

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts" / "RC"
PAGE_MARKER_RE = re.compile(r"<!--\s*page\s+\d+\s*-->", re.IGNORECASE)


class RCAgentResponse:
    # Structured result — the API route/service layer decides what to do
    # with it (e.g. hand `structure` off to docx_writer.py).
    def __init__(
        self,
        summary: str,
        structure: dict,
        intent: dict,
        agent_type: str,
        model: str,
        duration_seconds: float,
    ):
        self.summary = summary
        self.structure = structure
        self.intent = intent
        self.agent_type = agent_type
        self.model = model
        self.duration_seconds = round(duration_seconds, 2)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "structure": self.structure,
            "intent": self.intent,
            "agent_type": self.agent_type,
            "model": self.model,
            "duration_seconds": self.duration_seconds,
        }


class RCAgent(BaseAgent):

    agent_type = "rc_analysis"
    description = (
        "Analyse un règlement de consultation (RC) : génère un résumé "
        "général du document et extrait la structure du mémoire technique "
        "attendue"
    )

    def __init__(self):
        super().__init__()
        self.sections_dict = load_sections_dict()

    # ----------------------------------------------------------------
    # Orchestrator
    # ----------------------------------------------------------------

    def run(self, document_markdown: str, user_prompt: str = "") -> RCAgentResponse:
        start = time.time()

        clean_text = PAGE_MARKER_RE.sub("", document_markdown)

        intent = (
            self._parse_intent(user_prompt)
            if user_prompt.strip()
            else {"target_section_label": None, "emphasis_instructions": None}
        )

        # Independent steps — structure failing doesn't block the summary,
        # and vice versa.
        structure_result = self._extract_structure(
            document_markdown, intent["target_section_label"]
        )
        summary = self._summarize_document(clean_text, intent["emphasis_instructions"])

        return RCAgentResponse(
            summary=summary,
            structure=structure_result,
            intent=intent,
            agent_type=self.agent_type,
            model=self.llm.model,
            duration_seconds=time.time() - start,
        )

    # ----------------------------------------------------------------
    # Step 1 — Intent parsing (target section + emphasis, from one call)
    # ----------------------------------------------------------------

    def _parse_intent(self, user_prompt: str) -> dict:
        labels = [entry["label"] for entry in self.sections_dict.values()]
        labels_text = "\n".join(f"- {label}" for label in labels)

        user_message = (
            f"Demande de l'utilisateur :\n\"{user_prompt}\"\n\n"
            f"Libellés canoniques disponibles :\n{labels_text}\n"
        )

        response = self.chat(
            user_message=user_message,
            temperature=0.0,
            max_tokens=300,
            system_prompt_override=self._load_prompt("rc_intent"),
            json_mode=True,
        )

        try:
            data = json.loads(response.content)
        except json.JSONDecodeError:
            data = {}

        return {
            "target_section_label": data.get("target_section_label") or None,
            "emphasis_instructions": data.get("emphasis_instructions") or None,
        }

    # ----------------------------------------------------------------
    # Step 2 — Structure extraction (heading match, no LLM until found)
    # ----------------------------------------------------------------

    def _extract_structure(
        self, document_markdown: str, target_section_label: str | None
    ) -> dict:
        match = match_section(
            document_markdown,
            target_hint=target_section_label,
            sections_dict=self.sections_dict,
        )

        if match is None:
            return {
                "found": False,
                "heading": None,
                "match_score": None,
                "raw_text": None,
                "structure": None,
            }
        user_message = (
            f"Voici le texte de la section « {match['heading']} » extraite "
            f"du règlement de consultation :\n\n{match['text']}\n\n"
            "Génère la structure du mémoire technique attendue."
        )

        response = self.chat(
            user_message=user_message,
            temperature=0.0,
            max_tokens=1500,
            system_prompt_override=self._load_prompt("rc_structure"),
        )

        structure_text = self._ensure_annex_chapter(response.content)

        return {
            "found": True,
            "heading": match["heading"],
            "match_score": match["score"],
            "raw_text": match["text"],
            "structure": structure_text,
        }

    # ----------------------------------------------------------------
    # Step 3 — Summary (map-reduce, emphasis threaded through both steps)
    # ----------------------------------------------------------------

    def _summarize_document(self, document_text: str, emphasis: str | None) -> str:
        chunker = Chunker()
        chunks = chunker.chunk_for_summarization(document_text, chunk_size=2000)

        if len(chunks) <= 1:
            return self._reduce_summary([document_text], emphasis)

        map_prompt = self._load_prompt("rc_summary_map")
        chunk_summaries = []
        for chunk in chunks:
            message = self._build_map_message(chunk["text"], emphasis)
            response = self.chat(
                user_message=message,
                temperature=0.2,
                max_tokens=600,
                system_prompt_override=map_prompt,
            )
            chunk_summaries.append(response.content)

        return self._reduce_summary(chunk_summaries, emphasis)

    def _build_map_message(self, chunk_text: str, emphasis: str | None) -> str:
        message = f"Extrait du document :\n\n{chunk_text}\n"
        if emphasis:
            message += (
                f"\nSi cet extrait contient des informations liées à : "
                f"{emphasis}, conserve-les avec plus de détail que le reste.\n"
            )
        return message

    def _reduce_summary(self, summaries: list[str], emphasis: str | None) -> str:
        combined = "\n\n---\n\n".join(
            f"Extrait {i + 1}:\n{s}" for i, s in enumerate(summaries)
        )
        message = f"Résumés partiels du document :\n\n{combined}\n"
        if emphasis:
            message += (
                f"\nProduis une synthèse finale du document en insistant "
                f"particulièrement sur : {emphasis}\n"
            )
        else:
            message += "\nProduis une synthèse finale claire et complète du document.\n"

        response = self.chat(
            user_message=message,
            temperature=0.2,
            max_tokens=1800,
            system_prompt_override=self._load_prompt("rc_summary_reduce"),
        )
        return response.content

    # ----------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------

    def _ensure_annex_chapter(self, structure_text: str) -> str:
        # Belt-and-braces: the prompt already asks for a final "Annexes"
        # chapter, but a 14B quantized model won't follow fixed formatting
        # rules 100% of the time. Enforce it deterministically instead of
        # trusting instruction-following alone.
        top_level_re = re.compile(r"^\s*(\d+)\.\s+(.+)$", re.MULTILINE)
        matches = list(top_level_re.finditer(structure_text))

        if any("annexe" in m.group(2).lower() for m in matches):
            return structure_text

        next_num = int(matches[-1].group(1)) + 1 if matches else 1
        return structure_text.rstrip() + f"\n{next_num}. Annexes\n"

    def _load_prompt(self, name: str) -> str:
        prompt_file = PROMPTS_DIR / f"{name}.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Missing prompt file: {prompt_file}")
        return prompt_file.read_text(encoding="utf-8").strip()