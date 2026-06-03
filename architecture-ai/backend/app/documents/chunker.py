# chunker.py
# Splits extracted text into chunks for embedding and retrieval.
#
# Two modes:
#   - chunk_for_retrieval()    → small chunks (512 tokens) for RAG vector search
#   - chunk_for_summarization() → larger chunks (2000 tokens) for map-reduce analysis
#
# Why different sizes?
#   RAG needs small precise chunks so retrieval finds the exact relevant passage.
#   Summarization needs large chunks so the LLM has enough context per call.

import re
from app.config import settings


class Chunker:

    def chunk_for_retrieval(
        self,
        text: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> list[dict]:
        # Small chunks with overlap — for embedding into ChromaDB
        size = chunk_size or settings.CHUNK_SIZE
        overlap_size = overlap or settings.CHUNK_OVERLAP
        return self._split_text(text, size, overlap_size)

    def chunk_for_summarization(
        self,
        text: str,
        chunk_size: int = 2000,
    ) -> list[dict]:
        # Larger chunks, no overlap — for sequential summarization
        # No overlap needed because we process them in order
        return self._split_text(text, chunk_size, overlap=0)

    # chunker.py — replace the entire _split_text method

    def _split_text(
            self,
            text: str,
            chunk_size: int,
            overlap: int,
    ) -> list[dict]:
        # Hard character limit — nomic-embed-text breaks above ~8000 chars
        HARD_CHAR_LIMIT = 6000

        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_size = len(sentence.split())

            if current_size + sentence_size > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)

                # If chunk still exceeds hard char limit, split it further
                for sub_chunk in self._hard_split(chunk_text, HARD_CHAR_LIMIT):
                    chunks.append({
                        "index": chunk_index,
                        "text": sub_chunk,
                        "word_count": len(sub_chunk.split()),
                    })
                    chunk_index += 1

                if overlap > 0:
                    overlap_words = " ".join(current_chunk).split()[-overlap:]
                    current_chunk = overlap_words
                    current_size = len(overlap_words)
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size

        # Last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            for sub_chunk in self._hard_split(chunk_text, HARD_CHAR_LIMIT):
                chunks.append({
                    "index": chunk_index,
                    "text": sub_chunk,
                    "word_count": len(sub_chunk.split()),
                })
                chunk_index += 1

        return chunks

    def _hard_split(self, text: str, max_chars: int) -> list[str]:
        # Last resort splitter — cuts at max_chars on whitespace boundary
        # Used when sentence splitting produces chunks that are still too large
        if len(text) <= max_chars:
            return [text]

        parts = []
        while len(text) > max_chars:
            # Find last whitespace before the limit
            split_at = text.rfind(' ', 0, max_chars)
            if split_at == -1:
                split_at = max_chars  # no whitespace found — hard cut
            parts.append(text[:split_at].strip())
            text = text[split_at:].strip()

        if text:
            parts.append(text)

        return parts

    def _split_into_sentences(self, text: str) -> list[str]:
        # Simple sentence splitter that handles French punctuation
        # Splits on . ! ? followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-Ÿ\[])', text)
        # Remove empty strings and very short fragments
        return [s.strip() for s in sentences if len(s.strip()) > 10]