# ingestion.py
# Orchestrates the full document ingestion pipeline.
#
# Responsibilities:
#   1. Receives an uploaded file (PDF, DOCX, etc.)
#   2. Saves the original file to disk with a unique ID
#   3. Parses it using the appropriate parser
#   4. Saves the parsed output as JSON
#   5. Chunks the text
#   6. Embeds chunks and stores them in ChromaDB
#   7. Saves document metadata to a registry file
#
# This is the single entry point for ALL document ingestion.
# Agents never call parsers or chunkers directly — they always
# go through the retrieval layer which reads from ChromaDB.
import json
import time
import uuid
import shutil
from pathlib import Path
from datetime import datetime, timezone

from app.config import settings
from app.documents.parsers.pdf_parser import PDFParser
from app.documents.chunker import Chunker
from app.documents.doc_type import DocType
import chromadb


# Registry file — tracks all ingested documents
# One JSON file, one entry per document
REGISTRY_FILE = settings.upload_path / "registry.json"




def _get_chroma_collection():
    # Returns the ChromaDB collection where all document chunks are stored.
    # Creates it if it doesn't exist yet.
    client = chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
    )
    # get_or_create — safe to call multiple times
    return client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},  # cosine similarity for text
    )


def ingest_document(
    file_path: str | Path,
    uploaded_by: str = "unknown",
    max_pages: int | None = None,
    doc_type: DocType = DocType.unspecified,   # NEW — "memoire" | "programme" | "rc" | "unspecified"
    project_id: str | None = None,   # NEW — required in practice for "programme", optional otherwise
) -> dict:
    # Main entry point.
    # file_path: path to the file to ingest (already saved to disk)
    # uploaded_by: user identifier for audit log
    # doc_type: what kind of document this is — drives retrieval lane
    #     filtering later (see retrieval.py's `where` param). Defaults to
    #     "unspecified" so existing call sites (rc_agent, qa_agent) keep
    #     working unmodified — those chunks just won't match any
    #     doc_type-scoped filter, which is correct: they were never meant
    #     to be lane-filtered in the first place.
    # project_id: which project this document belongs to. Required in
    #     practice for "programme" docs (that's the whole point of the
    #     lane), meaningless for "memoire" docs (memoires are intentionally
    #     searched cross-project), optional/None otherwise.
    # Returns: document metadata dict

    file_path = Path(file_path)
    start = time.time()

    # 1. Generate unique document ID
    doc_id = str(uuid.uuid4())
    print(f"[ingestion] Starting ingestion for {file_path.name} — ID: {doc_id}")

    # 2. Create document directory
    doc_dir = settings.upload_path / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    # 3. Copy original file into document directory
    original_dest = doc_dir / file_path.name
    shutil.copy2(file_path, original_dest)
    print(f"[ingestion] Saved original file to {original_dest}")

    # 4. Parse the document
    print(f"[ingestion] Parsing document...")
    parsed = _parse_document(original_dest, max_pages)

    # 5. Save parsed output as JSON
    parsed_path = doc_dir / "parsed.json"
    parsed_path.write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[ingestion] Saved parsed output — {parsed['word_count']} words")

    # 6. Chunk the text
    print(f"[ingestion] Chunking text...")
    chunker = Chunker()
    chunks = chunker.chunk_for_retrieval(parsed["full_text"])
    _store_bm25_index(doc_id, chunks, doc_dir)
    print(f"[ingestion] Created {len(chunks)} chunks")

    # 7. Store chunks in ChromaDB
    print(f"[ingestion] Storing in ChromaDB...")
    _store_in_chromadb(doc_id, chunks, file_path.name, doc_type, project_id)

    # 8. Build and save metadata
    duration = round(time.time() - start, 2)
    metadata = {
        "doc_id": doc_id,
        "file_name": file_path.name,
        "file_path": str(original_dest),
        "uploaded_by": uploaded_by,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "total_pages": parsed["total_pages"],
        "word_count": parsed["word_count"],
        "chunk_count": len(chunks),
        "ingestion_duration_seconds": duration,
        "status": "ready",
        "doc_type": doc_type,        # NEW — persisted so list_documents()/
        "project_id": project_id,    # NEW — registry entries carry this too
    }

    # Save metadata alongside the parsed file
    metadata_path = doc_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 9. Register document in the global registry
    _register_document(metadata)

    print(f"[ingestion] Done in {duration}s — document ready for querying")
    return metadata

def _parse_document(file_path: Path, max_pages: int | None) -> dict:
    # Routes to the correct parser based on file extension.
    # Currently only PDF — DOCX and Excel parsers will plug in here later.
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        parser = PDFParser()
        return parser.parse(file_path)

    # Placeholder for future parsers
    elif suffix in [".docx", ".doc"]:
        from app.documents.parsers.docx_parser import parse_for_ingestion
        return parse_for_ingestion(file_path)

    elif suffix in [".xlsx", ".xls"]:
        raise NotImplementedError(
            "Excel parser not implemented yet — coming in next iteration"
        )
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: .pdf (DOCX and Excel coming soon)"
        )


def _store_in_chromadb(
    doc_id: str,
    chunks: list[dict],
    file_name: str,
    doc_type: DocType = DocType.unspecified,  # NEW
    project_id: str | None = None,    # NEW
):
    from app.documents.embedder import get_embedder

    # Filter out chunks too small to be meaningful
    MIN_CHUNK_WORDS = 30
    chunks = [c for c in chunks if c["word_count"] >= MIN_CHUNK_WORDS]
    print(f"[ingestion] {len(chunks)} chunks after filtering micro-chunks")

    collection = _get_chroma_collection()
    embedder = get_embedder()

    ids = [f"{doc_id}_chunk_{chunk['index']}" for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]

    # Chroma metadata values can't be None — omit project_id entirely
    # when it's None rather than storing a null, so `where` filters that
    # check its presence/absence behave predictably.
    metadatas = []
    for chunk in chunks:
        meta = {
            "doc_id": doc_id,
            "file_name": file_name,
            "chunk_index": chunk["index"],
            "word_count": chunk["word_count"],
            "doc_type": doc_type.value,
        }
        if project_id is not None:
            meta["project_id"] = project_id
        metadatas.append(meta)

    # Embed explicitly — never let ChromaDB embed silently
    print(f"[ingestion] Embedding {len(chunks)} chunks with {settings.OLLAMA_EMBEDDING_MODEL}...")
    embeddings = embedder.embed_batch(documents)

    # Store in batches of 100
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )
    print(f"[ingestion] Stored {len(chunks)} chunks in ChromaDB")

def _store_bm25_index(doc_id: str, chunks: list[dict], doc_dir: Path):
    # Saves chunks as a JSON file for BM25 keyword search.
    # BM25 is not stored in ChromaDB — it works directly on text.
    # This file is loaded at retrieval time to build the BM25 index.
    bm25_data = {
        "doc_id": doc_id,
        "chunks": [
            {
                "index": c["index"],
                "text": c["text"],
                "word_count": c["word_count"],
            }
            for c in chunks
        ]
    }
    bm25_path = doc_dir / "bm25_chunks.json"
    bm25_path.write_text(
        json.dumps(bm25_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[ingestion] Saved BM25 index — {len(chunks)} chunks")


def _register_document(metadata: dict):
    # Appends document metadata to the global registry.
    # Registry is a simple JSON array — one entry per document.
    # This lets us list all ingested documents without querying ChromaDB.
    registry = []
    if REGISTRY_FILE.exists():
        try:
            registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        except Exception:
            registry = []

    registry.append(metadata)
    REGISTRY_FILE.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_documents() -> list[dict]:
    # Returns all ingested documents from the registry.
    # Used by the API route GET /documents
    if not REGISTRY_FILE.exists():
        return []
    try:
        return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def get_document_metadata(doc_id: str) -> dict | None:
    # Returns metadata for a specific document by ID.
    docs = list_documents()
    return next((d for d in docs if d["doc_id"] == doc_id), None)


def get_document_full_text(doc_id: str) -> str | None:
    # Returns the full parsed markdown text for a document (page markers
    # included), read from the parsed.json saved at ingestion time.
    #
    # This is distinct from the chunked/embedded content in ChromaDB —
    # agents that need the WHOLE document (RCAgent's summary + structure
    # extraction, not RAG) go through this instead of the retrieval layer.
    doc_dir = settings.upload_path / doc_id
    parsed_path = doc_dir / "parsed.json"
    if not parsed_path.exists():
        return None
    try:
        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        return parsed.get("full_text")
    except Exception as e:
        print(f"[ingestion] Failed to read parsed text for {doc_id}: {e}")
        return None


def delete_document(doc_id: str) -> bool:
    # Removes a document from disk, ChromaDB, and the registry.
    # RGPD: this is the "right to deletion" implementation.
    try:
        # Remove from ChromaDB
        collection = _get_chroma_collection()
        results = collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            collection.delete(ids=results["ids"])

        # Remove files from disk
        doc_dir = settings.upload_path / doc_id
        if doc_dir.exists():
            shutil.rmtree(doc_dir)

        # Remove from registry
        registry = list_documents()
        registry = [d for d in registry if d["doc_id"] != doc_id]
        REGISTRY_FILE.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return True
    except Exception as e:
        print(f"[ingestion] Delete failed for {doc_id}: {e}")
        return False