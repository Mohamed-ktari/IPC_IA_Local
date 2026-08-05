# retrieval.py
# Hybrid retrieval — combines semantic search (ChromaDB) and
# keyword search (BM25) for best results on technical documents.
#
# Why hybrid?
# - Semantic search finds conceptually similar chunks (good for prose)
# - BM25 finds exact keyword matches (good for tables, form fields)
# - Architecture diagnostics are 80% tables → BM25 is essential here
#
# The final score for each chunk is:
#   hybrid_score = (semantic_weight * semantic_score)
#                + (bm25_weight * bm25_score)
#
# Default weights: 40% semantic, 60% BM25
# Adjust in config based on document type.

import json
import math
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from app.config import settings
from app.documents.embedder import get_embedder


class RetrievalResult:
    def __init__(
        self,
        text: str,
        doc_id: str,
        file_name: str,
        chunk_index: int,
        similarity_score: float,
        bm25_score: float = 0.0,
        hybrid_score: float = 0.0,
        source: str = "hybrid",
        original_file_name: str | None = None,
    ):
        self.text = text
        self.doc_id = doc_id
        self.file_name = file_name
        self.chunk_index = chunk_index
        self.similarity_score = similarity_score
        self.original_file_name = original_file_name
        self.bm25_score = bm25_score
        self.hybrid_score = hybrid_score
        self.source = source  # "semantic" | "bm25" | "hybrid"

    def __repr__(self):
        return (
            f"RetrievalResult("
            f"chunk={self.chunk_index}, "
            f"hybrid={self.hybrid_score:.3f}, "
            f"semantic={self.similarity_score:.3f}, "
            f"bm25={self.bm25_score:.3f}, "
            f"words={len(self.text.split())})"
        )


class Retriever:

    # Weight balance between semantic and BM25
    # 0.4/0.6 favors keyword matching — good for table-heavy docs
    # Change to 0.7/0.3 for prose-heavy docs
    SEMANTIC_WEIGHT = 0.3
    BM25_WEIGHT = 0.7

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

    def _load_bm25_chunks(self, doc_id: str) -> list[dict]:
        # Loads the BM25 chunk file saved at ingestion time
        bm25_path = settings.upload_path / doc_id / "bm25_chunks.json"
        if not bm25_path.exists():
            return []
        data = json.loads(bm25_path.read_text(encoding="utf-8"))
        return data.get("chunks", [])

    def _build_bm25_index(self, chunks: list[dict]) -> BM25Okapi:
        # Tokenizes chunks and builds BM25 index
        # Simple whitespace tokenization — works well for French technical docs
        tokenized = [
            chunk["text"].lower().split()
            for chunk in chunks
        ]
        return BM25Okapi(tokenized)

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        # Normalizes a list of scores to [0, 1] range
        # Needed to make semantic and BM25 scores comparable
        if not scores:
            return scores
        max_score = max(scores)
        min_score = min(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [
            (s - min_score) / (max_score - min_score)
            for s in scores
        ]

    def retrieve(
    self,
    query: str,
    top_k: int | None = None,
    doc_id: list[str] | None = None,
    where: dict | None = None,   # NEW — e.g. {"doc_type": "memoire"}
                                   # or {"$and": [{"doc_type": "programme"},
                                   #              {"project_id": "..."}]}
) -> list[RetrievalResult]:
        top_k = top_k or settings.RETRIEVAL_TOP_K

        semantic_results = self._semantic_search(query, top_k * 2, doc_id, where)
        bm25_results = self._bm25_search(query, top_k * 2, doc_id, where)

        merged = self._merge_results(
            semantic_results,
            bm25_results,
            top_k,
        )

        return merged

    def _semantic_search(
    self,
    query: str,
    top_k: int,
    doc_id: list[str] | None,
    where: dict | None = None,   # NEW
) -> list[RetrievalResult]:
        collection = self._get_collection()
        query_embedding = self.embedder.embed(query)

        # doc_id (single-document scoping, e.g. QA agent on one uploaded doc)
        # and where (category/project scoping, e.g. this new agent's lanes)
        # are two different filtering needs — combine them if both given.
        if isinstance(doc_id, list):
            doc_filter = {"doc_id": {"$in": doc_id}} if doc_id else None
        elif doc_id:
            doc_filter = {"doc_id": doc_id}
        else:
            doc_filter = None

        if doc_filter and where:
            where_filter = {"$and": [doc_filter, where]}
        elif doc_filter:
            where_filter = doc_filter
        elif where:
            where_filter = where
        else:
            where_filter = None

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        retrieved = []
        if not results["ids"] or not results["ids"][0]:
            return retrieved

        for i in range(len(results["ids"][0])):
            text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            similarity = 1 - (distance / 2)

            retrieved.append(RetrievalResult(
                text=text,
                doc_id=metadata.get("doc_id", ""),
                file_name=metadata.get("file_name", ""),
                original_file_name=metadata.get("original_file_name", ""),
                chunk_index=metadata.get("chunk_index", i),
                similarity_score=similarity,
                source="semantic",
            ))

        return retrieved
        
    def _bm25_search(
    self,
    query: str,
    top_k: int,
    doc_id: list[str] | None,
    where: dict | None = None,   # NEW
    ) -> list[RetrievalResult]:
        # Determine which documents to search
        if isinstance(doc_id, list):
            doc_ids = doc_id
        elif doc_id:
            doc_ids = [doc_id]
        else:
            from app.documents.ingestion import list_documents
            docs = list_documents()
            if where:
                docs = [d for d in docs if _matches_where(d, where)]
            doc_ids = [d["doc_id"] for d in docs]

        all_results = []

        for did in doc_ids:
            chunks = self._load_bm25_chunks(did)
            if not chunks:
                continue

            chunks = [c for c in chunks if c["word_count"] >= 30]
            if not chunks:
                continue

            bm25 = self._build_bm25_index(chunks)
            tokenized_query = query.lower().split()
            raw_scores = bm25.get_scores(tokenized_query)

            meta_path = settings.upload_path / did / "metadata.json"
            file_name = did
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                file_name = meta.get("file_name", did)
                original_file_name = meta.get("original_file_name", did) 

            for i, score in enumerate(raw_scores):
                if score > 0:
                    all_results.append(RetrievalResult(
                        text=chunks[i]["text"],
                        doc_id=did,
                        file_name=file_name,
                        original_file_name=original_file_name,
                        chunk_index=chunks[i]["index"],
                        similarity_score=0.0,
                        bm25_score=score,
                        source="bm25",
                    ))

        all_results.sort(key=lambda r: r.bm25_score, reverse=True)
        return all_results[:top_k]

    def _merge_results(
        self,
        semantic: list[RetrievalResult],
        bm25: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        # Merge semantic and BM25 results using weighted hybrid scoring
        # Key: chunk_index + doc_id — same chunk from both searches = one result

        # Normalize scores to [0,1] for fair comparison
        semantic_scores = self._normalize_scores(
            [r.similarity_score for r in semantic]
        )
        bm25_scores = self._normalize_scores(
            [r.bm25_score for r in bm25]
        )

        # Build lookup by (doc_id, chunk_index)
        merged: dict[str, RetrievalResult] = {}

        for i, result in enumerate(semantic):
            key = f"{result.doc_id}_{result.chunk_index}"
            result.similarity_score = semantic_scores[i]
            result.hybrid_score = self.SEMANTIC_WEIGHT * semantic_scores[i]
            merged[key] = result

        for i, result in enumerate(bm25):
            key = f"{result.doc_id}_{result.chunk_index}"
            result.bm25_score = bm25_scores[i]
            bm25_contribution = self.BM25_WEIGHT * bm25_scores[i]

            if key in merged:
                # Chunk found by both — combine scores
                merged[key].bm25_score = result.bm25_score
                merged[key].hybrid_score += bm25_contribution
                merged[key].source = "hybrid"
            else:
                # Only found by BM25 — add it
                result.hybrid_score = bm25_contribution
                merged[key] = result

        # Sort by hybrid score and return top_k
        results = list(merged.values())
        results.sort(key=lambda r: r.hybrid_score, reverse=True)
        return results[:top_k]

    def format_context(
        self,
        results: list[RetrievalResult],
        min_score: float = 0.3,
    ) -> str:
        filtered = [r for r in results if r.hybrid_score >= min_score]

        if not filtered:
            return "Aucun document pertinent trouvé pour cette question."

        parts = []
        for i, result in enumerate(filtered, start=1):
            parts.append(
                f"[Source {i} — {result.file_name}, "
                f"section {result.chunk_index}, "
                f"pertinence: {result.hybrid_score:.0%} "
                f"(sémantique: {result.similarity_score:.0%}, "
                f"mots-clés: {result.bm25_score:.0%})]\n"
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
        results = self.retrieve(query, top_k=top_k, doc_id=doc_id)
        context = self.format_context(results, min_score=min_score)
        return context, results
    # ------------------------------------------------------------
    # Ephemeral retrieval — for documents that were never ingested
    # ------------------------------------------------------------
    # Used by callers that need RAG on a document that isn't (and won't
    # be) persisted through ingest_document() — e.g. RCAgent's structure
    # fallback. No ChromaDB write, no bm25_chunks.json on disk. Everything
    # is built and discarded within this call. Reuses the same hybrid
    # merge/normalize logic as the persisted path.

    def retrieve_ephemeral(
        self,
        query: str,
        chunks: list[dict],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        top_k = top_k or settings.RETRIEVAL_TOP_K
        if not chunks:
            return []

        semantic_results = self._semantic_search_ephemeral(query, chunks)
        bm25_results = self._bm25_search_ephemeral(query, chunks)

        return self._merge_results(semantic_results, bm25_results, top_k)

    def _semantic_search_ephemeral(
        self, query: str, chunks: list[dict]
    ) -> list[RetrievalResult]:
        query_embedding = self.embedder.embed(query)
        chunk_embeddings = self.embedder.embed_batch([c["text"] for c in chunks])

        results = []
        for chunk, emb in zip(chunks, chunk_embeddings):
            similarity = self._cosine_similarity(query_embedding, emb)
            results.append(RetrievalResult(
                text=chunk["text"],
                doc_id="ephemeral",
                file_name="ephemeral",
                chunk_index=chunk["index"],
                similarity_score=similarity,
                source="semantic",
            ))
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results

    def _bm25_search_ephemeral(
        self, query: str, chunks: list[dict]
    ) -> list[RetrievalResult]:
        chunks = [c for c in chunks if c["word_count"] >= 30]
        if not chunks:
            return []

        bm25 = self._build_bm25_index(chunks)
        tokenized_query = query.lower().split()
        raw_scores = bm25.get_scores(tokenized_query)

        results = []
        for i, score in enumerate(raw_scores):
            if score > 0:
                results.append(RetrievalResult(
                    text=chunks[i]["text"],
                    doc_id="ephemeral",
                    file_name="ephemeral",
                    chunk_index=chunks[i]["index"],
                    similarity_score=0.0,
                    bm25_score=score,
                    source="bm25",
                ))
        results.sort(key=lambda r: r.bm25_score, reverse=True)
        return results

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


_retriever: Retriever | None = None

def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _matches_where(doc_metadata: dict, where: dict) -> bool:
    # Minimal interpreter for the subset of Chroma's `where` shape this
    # module actually produces: either a flat {"key": "value"} equality
    # dict, or {"$and": [ {...}, {...} ]}. Extend if a caller ever needs
    # $or/$ne here — not needed yet, so not built yet.
    if "$and" in where:
        return all(_matches_where(doc_metadata, clause) for clause in where["$and"])
    return all(doc_metadata.get(k) == v for k, v in where.items())