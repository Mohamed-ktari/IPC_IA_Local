# documents.py
# HTTP routes for document upload and management.
#
# These routes are intentionally "thin" — they handle HTTP concerns
# (receiving files, returning JSON, error codes) and delegate all
# real logic to app.documents.ingestion.
#
# Endpoints:
#   POST   /documents/upload   — upload and ingest a document
#   GET    /documents          — list all ingested documents
#   GET    /documents/{doc_id} — get metadata for one document
#   DELETE /documents/{doc_id} — delete a document (RGPD right to deletion)

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from app.documents.doc_type import DocType

from app.documents.ingestion import (
    ingest_document,
    list_documents,
    get_document_metadata,
    delete_document,
)

router = APIRouter()


# ----------------------------------------------------------------
# Response models — define the exact shape of JSON returned to clients
# ----------------------------------------------------------------

class DocumentMetadata(BaseModel):
    doc_id: str
    file_name: str
    original_file_name: str | None = None
    uploaded_by: str
    uploaded_at: str
    total_pages: int
    word_count: int
    chunk_count: int
    ingestion_duration_seconds: float
    status: str
    doc_type: DocType


class UploadResponse(BaseModel):
    message: str
    document: DocumentMetadata


class DocumentListResponse(BaseModel):
    count: int
    documents: list[DocumentMetadata]


class DeleteResponse(BaseModel):
    message: str
    doc_id: str


# ----------------------------------------------------------------
# Routes
# ----------------------------------------------------------------

@router.post("/upload", response_model=UploadResponse,)
async def upload_document(file: UploadFile = File(...),doc_type: DocType = Form(...)):
    # Receives a file upload, saves it temporarily, then runs
    # the full ingestion pipeline (parse → chunk → embed → store).
    #
    # Supported types are enforced here at the boundary (fast, clear
    # error) AND in ingestion.py's _parse_document (defense in depth —
    # any other caller of ingest_document() is still protected even if
    # it bypasses this route).
    ALLOWED_SUFFIXES = (".pdf", ".docx", ".doc")

    if not file.filename.lower().endswith(ALLOWED_SUFFIXES):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(ALLOWED_SUFFIXES)}"
        )

    # Save uploaded file to a temporary location first.
    # ingestion.py will copy it into its own managed storage (data/uploads/<doc_id>/)
    suffix = Path(file.filename).suffix.lower()
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")
    
    try:
        metadata = ingest_document(
        file_path=tmp_path,
        original_filename=file.filename,
        uploaded_by="api_user",
        doc_type= doc_type,  # Pass the doc_type to the ingestion function
    )
    except NotImplementedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return UploadResponse(
        message="Document ingested successfully",
        document=DocumentMetadata(**metadata),
    )


@router.get("", response_model=DocumentListResponse)
async def get_documents():
    # Returns all ingested documents from the registry
    docs = list_documents()
    return DocumentListResponse(
        count=len(docs),
        documents=[DocumentMetadata(**d) for d in docs],
    )


@router.get("/{doc_id}", response_model=DocumentMetadata)
async def get_document(doc_id: str):
    # Returns metadata for a single document
    metadata = get_document_metadata(doc_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return DocumentMetadata(**metadata)


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def remove_document(doc_id: str):
    # RGPD: right to deletion — removes file, ChromaDB entries, and registry record
    metadata = get_document_metadata(doc_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    success = delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete document")

    return DeleteResponse(message="Document deleted", doc_id=doc_id)