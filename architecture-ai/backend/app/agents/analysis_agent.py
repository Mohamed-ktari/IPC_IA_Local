# analysis_agent.py
# Feature 1 — Document analysis and structured synthesis.
#
# What it does:
#   1. Receives raw document text + a template name
#   2. Loads the template structure from /templates/<template_name>.json
#   3. Builds a dynamic system prompt that instructs the LLM to fill each section
#   4. Returns a structured synthesis the user can export to Word/PDF
#
# This agent does NOT do RAG yet — it works on a document passed directly.
# RAG (querying the vector DB) will be added in the retrieval layer (week 2).

import json
from pathlib import Path
from app.agents.base_agent import BaseAgent, AgentResponse

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


class AnalysisAgent(BaseAgent):

    agent_type = "analysis"
    description = "Analyses documents and generates structured summaries following company templates"

    def run(
        self,
        document_text: str,
        template_name: str = "amiante",
        additional_instructions: str = "",
    ) -> AgentResponse:
        # Load the template
        template = self._load_template(template_name)

        # Build a dynamic system prompt that includes the template structure
        system_prompt = self._build_analysis_prompt(template)

        # Build the user message — document + any extra instructions
        user_message = self._build_user_message(
            document_text, template, additional_instructions
        )

        return self.chat(
            user_message=user_message,
            temperature=0.1,    # very low — we want deterministic extraction
            max_tokens=3000,
            system_prompt_override=system_prompt,
        )

    def _load_template(self, template_name: str) -> dict:
        template_file = TEMPLATES_DIR / f"{template_name}.json"
        if not template_file.exists():
            available = [f.stem for f in TEMPLATES_DIR.glob("*.json")]
            raise ValueError(
                f"Template '{template_name}' not found. "
                f"Available templates: {', '.join(available)}"
            )
        return json.loads(template_file.read_text(encoding="utf-8"))

    def _build_analysis_prompt(self, template: dict) -> str:
        # Builds the system prompt dynamically from the template structure
        sections_text = "\n".join([
            f"## {s['label']}\n{s['description']}"
            for s in template["sections"]
        ])
        return f"""You are a specialized document analysis assistant for an architecture and engineering firm.

Your task is to analyze the provided document and produce a structured synthesis following this exact template: {template['name']}.

You MUST structure your response with the following sections in order:
{sections_text}

Rules:
- Respond in the same language as the document (French if the document is in French)
- Be faithful to the source — never invent information not present in the document
- If information for a section is not found in the document, write: "Information non disponible"
- Be concise and professional
- Use bullet points inside sections when listing multiple items"""

    def _build_user_message(
        self,
        document_text: str,
        template: dict,
        additional_instructions: str,
    ) -> str:
        message = f"""Please analyze the following document and produce a structured synthesis following the {template['name']} template.

DOCUMENT:
{document_text}
"""
        if additional_instructions:
            message += f"\nADDITIONAL INSTRUCTIONS:\n{additional_instructions}\n"

        return message



def run_on_long_document(
    self,
    document_text: str,
    template_name: str,
    chunk_size: int = 2000,
) -> AgentResponse:
    # For documents too long to fit in one LLM context window.
    # Strategy: summarize each chunk → combine summaries → final synthesis
    import time
    from app.documents.chunker import Chunker

    start = time.time()
    chunker = Chunker()
    chunks = chunker.chunk_for_summarization(document_text, chunk_size=chunk_size)

    # If document fits in one call — use regular run()
    if len(chunks) <= 1:
        return self.run(document_text, template_name)

    print(f"Long document: {len(chunks)} chunks — using map-reduce")

    # MAP — summarize each chunk independently
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"  Summarizing chunk {i+1}/{len(chunks)}...")
        summary_response = self.chat(
            user_message=f"Summarize the key information in this document excerpt:\n\n{chunk['text']}",
            temperature=0.1,
            max_tokens=500,
        )
        chunk_summaries.append(summary_response.content)

    # REDUCE — combine all summaries into final structured synthesis
    combined_summaries = "\n\n---\n\n".join([
        f"Excerpt {i+1}:\n{summary}"
        for i, summary in enumerate(chunk_summaries)
    ])

    template = self._load_template(template_name)
    system_prompt = self._build_analysis_prompt(template)

    final_response = self.chat(
        user_message=self._build_user_message(
            combined_summaries, template,
            "This text is a combination of summaries from different parts of the document."
        ),
        temperature=0.1,
        max_tokens=3000,
        system_prompt_override=system_prompt,
    )

    duration = time.time() - start
    final_response.duration_seconds = round(duration, 2)
    return final_response