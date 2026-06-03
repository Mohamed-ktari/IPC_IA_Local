# retrieval.py
# Retrieves relevant document chunks from ChromaDB for a given query.
#
# This is the "R" in RAG (Retrieval-Augmented Generation).
# It is called by agents that need to answer questions about ingested documents.
#
# Flow:
#   1. Embed the user query using the same model used at ingestion time
#   2. Search ChromaDB for the most similar chunks (cosine similarity)
#   3. Optionally filter by specific document ID
#   4. Return chunks formatted and ready for LLM context injection
#
# Critical rule: the embedding model here MUST be identical to the one
# used in embedder.py at ingestion time. If you change the model,
# you must re-ingest all documents.

import chromadb
from app.config import settings
from app.documents.embedder import get_embedder


class RetrievalResult:
    # Represents a single retrieved chunk with its metadata
    def __init__(
        self,
        text: str,
        doc_id: str,
        file_name: str,
        chunk_index: int,
        similarity_score: float,
    ):
        self.text = text
        self.doc_id = doc_id
        self.file_name = file_name
        self.chunk_index = chunk_index
        self.similarity_score = similarity_score

    def __repr__(self):
        return (
            f"RetrievalResult("
            f"file='{self.file_name}', "
            f"chunk={self.chunk_index}, "
            f"score={self.similarity_score:.3f}, "
            f"words={len(self.text.split())})"
        )


class Retriever:

    def __init__(self):
        self.embedder = get_embedder()
        self.client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
        )

    def _get_collection(self):
        return self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        doc_id: str | None = None,
    ) -> list[RetrievalResult]:
        # Main entry point.
        # query   : the user's question in natural language
        # top_k   : how many chunks to return (defaults to config value)
        # doc_id  : if provided, only search within that specific document
        #           if None, search across ALL ingested documents

        top_k = top_k or settings.RETRIEVAL_TOP_K
        collection = self._get_collection()

        # Embed the query — same model as ingestion
        query_embedding = self.embedder.embed(query)

        # Build filter — optionally restrict to one document
        where_filter = {"doc_id": doc_id} if doc_id else None

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Parse results into RetrievalResult objects
        retrieved = []
        if not results["ids"] or not results["ids"][0]:
            return retrieved

        for i, chunk_id in enumerate(results["ids"][0]):
            text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]

            # ChromaDB returns cosine distance (0=identical, 2=opposite)
            # Convert to similarity score (1=identical, 0=unrelated)
            similarity = 1 - (distance / 2)

            retrieved.append(RetrievalResult(
                text=text,
                doc_id=metadata.get("doc_id", ""),
                file_name=metadata.get("file_name", ""),
                chunk_index=metadata.get("chunk_index", i),
                similarity_score=similarity,
            ))

        # Sort by similarity descending — best match first
        retrieved.sort(key=lambda r: r.similarity_score, reverse=True)
        return retrieved

    def format_context(
        self,
        results: list[RetrievalResult],
        min_score: float = 0.3,
    ) -> str:
        # Formats retrieved chunks into a single context string
        # ready to be injected into an LLM prompt.
        #
        # min_score: chunks below this similarity threshold are excluded
        # 0.3 is a reasonable default — below this the chunk is likely
        # not relevant to the query at all

        filtered = [r for r in results if r.similarity_score >= min_score]

        if not filtered:
            return "Aucun document pertinent trouvé pour cette question."

        parts = []
        for i, result in enumerate(filtered, start=1):
            parts.append(
                f"[Source {i} — {result.file_name}, "
                f"section {result.chunk_index}, "
                f"pertinence: {result.similarity_score:.0%}]\n"
                f"{result.text}"
            )

        return "\n\n---\n\n".join(parts)

    def retrieve_and_format(
        self,
        query: str,
        top_k: int | None = None,
        doc_id: str | None = None,
        min_score: float = 0.3,
    ) -> tuple[str, list[RetrievalResult]]:
        # Convenience method — retrieves and formats in one call.
        # Returns both the formatted context string AND the raw results
        # so the agent can log sources separately.
        results = self.retrieve(query, top_k=top_k, doc_id=doc_id)
        context = self.format_context(results, min_score=min_score)
        return context, results


# Single instance — reuse across requests
_retriever: Retriever | None = None

def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever