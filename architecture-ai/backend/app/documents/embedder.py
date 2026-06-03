# embedder.py
# Handles text embedding for the RAG pipeline.
#
# Responsibilities:
#   - Converts text chunks into vector embeddings
#   - Uses nomic-embed-text via Ollama — runs fully locally
#   - Same model MUST be used at ingestion time and retrieval time
#     (if you change the model, you must re-ingest all documents)
#
# Why a separate file?
#   Embedding is a distinct concern from chunking and storage.
#   If we want to swap the embedding model later, we change only this file.

import httpx
from app.config import settings


class Embedder:

    def __init__(self):
        self.model = settings.OLLAMA_EMBEDDING_MODEL
        self.base_url = settings.OLLAMA_BASE_URL
        self.timeout = 60  # embedding is faster than generation

    def embed(self, text: str) -> list[float]:
        # Embeds a single text string.
        # Returns a list of floats (the embedding vector).
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": self.model,
                        "input": text,
                    },
                )
                response.raise_for_status()
                return response.json()["embeddings"][0]

        except httpx.TimeoutException:
            raise RuntimeError(
                f"Embedding timed out — Ollama may be busy with generation"
            )
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {str(e)}")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Embeds a list of texts one by one.
        # Ollama doesn't support true batch embedding yet,
        # so we loop — but we do it efficiently with a single client.
        embeddings = []
        with httpx.Client(timeout=self.timeout) as client:
            for i, text in enumerate(texts):
                try:
                    response = client.post(
                        f"{self.base_url}/api/embed",
                        json={
                            "model": self.model,
                            "input": text,
                        },
                    )
                    response.raise_for_status()
                    embeddings.append(response.json()["embeddings"][0])

                    # Progress indicator for large batches
                    if (i + 1) % 10 == 0:
                        print(
                            f"[embedder] Embedded {i + 1}/{len(texts)} chunks..."
                        )

                except Exception as e:
                    raise RuntimeError(
                        f"Embedding failed on chunk {i}: {str(e)}"
                    )
        return embeddings


# Single instance — reuse across calls
_embedder: Embedder | None = None

def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder