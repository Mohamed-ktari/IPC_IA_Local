# analysis_agent.py
# Feature 1 — Document analysis and structured synthesis.
#
# Two modes of operation:
#
#   1. DIRECT mode — document text passed directly (small docs, testing)
#      agent.run(document_text=text, template_name="amiante")
#
#   2. RAG mode — agent retrieves relevant chunks from ChromaDB
#      agent.run_rag(query=query, doc_id=doc_id, template_name="amiante")
#
# RAG mode is what runs in production — the user uploads a document,
# it gets ingested, then the agent queries ChromaDB to find relevant
# sections and synthesizes a structured response.

import json
from pathlib import Path

from app.agents.base_agent import BaseAgent, AgentResponse
from app.documents.retrieval import get_retriever

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


class AnalysisAgent(BaseAgent):

    agent_type = "analysis"
    description = (
        "Analyses documents and generates structured summaries "
        "following company templates"
    )

    # ----------------------------------------------------------------
    # MODE 1 — Direct analysis (document text passed directly)
    # ----------------------------------------------------------------

    def run(
        self,
        document_text: str,
        template_name: str = "amiante",
        additional_instructions: str = "",
    ) -> AgentResponse:
        template = self._load_template(template_name)
        system_prompt = self._build_analysis_prompt(template)
        user_message = self._build_user_message(
            document_text, template, additional_instructions
        )
        return self.chat(
            user_message=user_message,
            temperature=0.1,
            max_tokens=3000,
            system_prompt_override=system_prompt,
        )

    # ----------------------------------------------------------------
    # MODE 2 — RAG mode (retrieves from ChromaDB)
    # ----------------------------------------------------------------

    def run_rag(
        self,
        doc_id: str,
        template_name: str = "amiante",
        additional_instructions: str = "",
        top_k: int = 8,
    ) -> AgentResponse:
        # RAG mode — retrieves relevant chunks then synthesizes.
        #
        # Strategy for structured templates:
        # Instead of one generic query, we query once per template section.
        # This ensures every section has relevant context, not just the
        # sections that happen to match a single query embedding.

        template = self._load_template(template_name)
        retriever = get_retriever()

        # Query once per section — collect all relevant chunks
        all_context_parts = []
        seen_chunks = set()  # avoid duplicate chunks across sections

        for section in template["sections"]:
            # Build a targeted query for this section
            section_query = (
                f"{section['label']} : {section['description']}"
            )

            results = retriever.retrieve(
                query=section_query,
                top_k=3,  # 3 chunks per section
                doc_id=doc_id,
            )

            for result in results:
                chunk_key = f"{result.doc_id}_{result.chunk_index}"
                if chunk_key not in seen_chunks and result.hybrid_score > 0.2:
                    seen_chunks.add(chunk_key)
                    all_context_parts.append(
                        f"[Section: {section['label']} — "
                        f"pertinence: {result.hybrid_score:.0%}]\n"
                        f"{result.text}"
                    )

        if not all_context_parts:
            return AgentResponse(
                content=(
                    "Aucun contenu pertinent trouvé dans ce document "
                    "pour générer la synthèse demandée."
                ),
                agent_type=self.agent_type,
                model="none",
                duration_seconds=0,
            )

        # Combine all retrieved context
        context = "\n\n---\n\n".join(all_context_parts)

        print(
            f"[analysis_agent] RAG retrieved {len(all_context_parts)} "
            f"unique chunks for {len(template['sections'])} sections"
        )

        # Build prompt and run
        system_prompt = self._build_analysis_prompt(template)
        user_message = self._build_user_message(
            context, template, additional_instructions
        )

        return self.chat(
            user_message=user_message,
            temperature=0.1,
            max_tokens=3000,
            system_prompt_override=system_prompt,
        )

    # ----------------------------------------------------------------
    # MODE 3 — Long document map-reduce
    # ----------------------------------------------------------------

    def run_on_long_document(
        self,
        document_text: str,
        template_name: str,
        chunk_size: int = 2000,
    ) -> AgentResponse:
        import time
        from app.documents.chunker import Chunker

        start = time.time()
        chunker = Chunker()
        chunks = chunker.chunk_for_summarization(
            document_text, chunk_size=chunk_size
        )

        if len(chunks) <= 1:
            return self.run(document_text, template_name)

        print(f"[analysis_agent] Long doc: {len(chunks)} chunks — map-reduce")

        # MAP — summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"  Summarizing chunk {i+1}/{len(chunks)}...")
            summary = self.chat(
                user_message=(
                    f"Résume les informations clés de cet extrait "
                    f"de document technique :\n\n{chunk['text']}"
                ),
                temperature=0.1,
                max_tokens=500,
            )
            chunk_summaries.append(summary.content)

        # REDUCE — combine into final synthesis
        combined = "\n\n---\n\n".join([
            f"Extrait {i+1}:\n{s}"
            for i, s in enumerate(chunk_summaries)
        ])

        template = self._load_template(template_name)
        system_prompt = self._build_analysis_prompt(template)

        final = self.chat(
            user_message=self._build_user_message(
                combined, template,
                "Ce texte est une combinaison de résumés d'extraits du document."
            ),
            temperature=0.1,
            max_tokens=3000,
            system_prompt_override=system_prompt,
        )

        final.duration_seconds = round(time.time() - start, 2)
        return final

    # ----------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------

    def _load_template(self, template_name: str) -> dict:
        template_file = TEMPLATES_DIR / f"{template_name}.json"
        if not template_file.exists():
            available = [f.stem for f in TEMPLATES_DIR.glob("*.json")]
            raise ValueError(
                f"Template '{template_name}' not found. "
                f"Available: {', '.join(available)}"
            )
        return json.loads(template_file.read_text(encoding="utf-8"))

    def _build_analysis_prompt(self, template: dict) -> str:
        sections_text = "\n".join([
            f"## {s['label']}\n{s['description']}"
            for s in template["sections"]
        ])
        return f"""Tu es un assistant spécialisé dans l'analyse de documents techniques \
pour un bureau d'études en architecture et ingénierie.

Ta tâche est d'analyser le contenu fourni et de produire une synthèse structurée \
selon ce modèle : {template['name']}.

Tu DOIS structurer ta réponse avec les sections suivantes dans l'ordre :
{sections_text}

Règles :
- Réponds en français
- Sois fidèle au contenu source — n'invente jamais d'informations absentes du document
- Si une section ne peut pas être renseignée, écris : "Information non disponible"
- Sois concis et professionnel
- Utilise des listes à puces pour les éléments multiples
- Cite toujours la source quand tu extrais une information précise"""

    def _build_user_message(
        self,
        document_text: str,
        template: dict,
        additional_instructions: str = "",
    ) -> str:
        message = (
            f"Analyse le contenu suivant et produis une synthèse structurée "
            f"selon le modèle {template['name']} :\n\n"
            f"CONTENU :\n{document_text}\n"
        )
        if additional_instructions:
            message += f"\nINSTRUCTIONS SUPPLÉMENTAIRES :\n{additional_instructions}\n"
        return message