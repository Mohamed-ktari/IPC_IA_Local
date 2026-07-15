# qa_agent.py
# Conversational Q&A over an ingested document (any type — PDF today,
# DOCX/Excel once their parsers exist). RAG-based: unlike RCAgent (bounded
# ~15 pages, full-context), a Q&A document can be arbitrarily long, so this
# always retrieves per-turn rather than resending the whole document.
#
# Two steps per turn:
#   1. Query rewrite — condense (recent history + new question) into a
#      standalone search query. Needed because a raw follow-up like
#      "et pour le prix ?" has no meaningful embedding on its own — the
#      pronoun/ellipsis needs resolving before retrieval, not after.
#   2. Retrieve (hybrid, doc-scoped via retrieval.py) + answer, grounded
#      only in what was retrieved.
#
# Conversation history lives in Redis (conversation_store.py) keyed by
# conversation_id — a document can have several separate conversations.
# CRITICAL: only the raw {question, answer} pair is persisted to history —
# never the retrieval-augmented prompt sent to the LLM for a given turn.
# Storing the augmented version would make stored history balloon with
# redundant retrieved chunks on every turn.

from pathlib import Path

from app.agents.base_agent import BaseAgent, AgentResponse
from app.llm.base import Message, Role
from app.documents.retrieval import get_retriever
from app.conversations import store as conversation_store
from app.config import settings

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"
REWRITE_HISTORY_TURNS = 6  # last N messages fed to the query-rewrite step


class QAAgent(BaseAgent):

    agent_type = "qa"
    description = (
        "Répond aux questions de l'utilisateur sur un document déjà "
        "ingéré, en conversation multi-tours (RAG)."
    )

    def start_conversation(self, doc_id: str) -> str:
        return conversation_store.create_conversation(doc_id)

    def ask(self, conversation_id: str, question: str) -> dict:
        conversation = conversation_store.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation '{conversation_id}' not found")

        doc_id = conversation["doc_id"]
        raw_history = conversation["messages"]  # [{"role", "content"}, ...]

        history_messages = [
            Message(role=Role(m["role"]), content=m["content"])
            for m in raw_history
        ]

        # Step 1 — query rewrite. Skip the extra LLM call on the first turn:
        # there's no history to disambiguate against yet.
        standalone_query = (
            self._rewrite_query(raw_history, question) if raw_history else question
        )

        # Step 2 — retrieve, scoped to this document only
        retriever = get_retriever()
        context, results = retriever.retrieve_and_format(
            query=standalone_query,
            top_k=settings.QA_RETRIEVAL_TOP_K,
            doc_id=doc_id,
            min_score=settings.QA_MIN_SCORE,
        )

        user_message = f"Contexte extrait du document :\n\n{context}\n\nQuestion : {question}"

        response: AgentResponse = self.chat(
            user_message=user_message,
            history=history_messages,
            temperature=0.2,
            max_tokens=1200,
        )

        # Persist the RAW question/answer only — see module docstring.
        conversation_store.append_turn(conversation_id, question, response.content)

        # Only report sources that actually cleared the relevance threshold
        # used in format_context — otherwise the frontend would show sources
        # the LLM never actually saw in its prompt.
        used_results = [r for r in results if r.hybrid_score >= settings.QA_MIN_SCORE]

        return {
            "conversation_id": conversation_id,
            "answer": response.content,
            "standalone_query": standalone_query,
            "sources": [
                {
                    "chunk_index": r.chunk_index,
                    "file_name": r.file_name,
                    "hybrid_score": round(r.hybrid_score, 3),
                }
                for r in used_results
            ],
            "model": response.model,
            "duration_seconds": response.duration_seconds,
        }

    def _rewrite_query(self, raw_history: list[dict], question: str) -> str:
        history_text = "\n".join(
            f"{'Utilisateur' if m['role'] == 'user' else 'Assistant'} : {m['content']}"
            for m in raw_history[-REWRITE_HISTORY_TURNS:]
        )
        user_message = (
            f"Historique de la conversation :\n{history_text}\n\n"
            f"Nouvelle question : {question}"
        )
        response = self.chat(
            user_message=user_message,
            temperature=0.0,
            max_tokens=150,
            system_prompt_override=self._load_prompt("qa_query_rewrite"),
        )
        return response.content.strip()

    def _load_prompt(self, name: str) -> str:
        prompt_file = PROMPTS_DIR / f"{name}.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Missing prompt file: {prompt_file}")
        return prompt_file.read_text(encoding="utf-8").strip()

    def run(self, *args, **kwargs):
        # BaseAgent requires this abstract method. QAAgent doesn't fit the
        # single-shot run(document, ...) shape the other agents use — its
        # real entry points are start_conversation()/ask() instead.
        raise NotImplementedError(
            "QAAgent uses start_conversation()/ask(), not run()."
        )