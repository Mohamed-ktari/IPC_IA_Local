# qa.py
# HTTP routes for conversational Q&A over a single ingested document.
#
# A document can have multiple separate conversations (each with its own
# conversation_id) — that's why "start conversation" is a distinct step
# from "ask a question", rather than passing doc_id on every ask.
#
# Endpoints:
#   POST   /qa/conversations                    — start a new conversation for a doc
#   POST   /qa/conversations/{conversation_id}/ask — ask a question in it
#   GET    /qa/conversations/{conversation_id}   — fetch full history
#   GET    /qa/documents/{doc_id}/conversations  — list conversation_ids for a doc
#   DELETE /qa/conversations/{conversation_id}   — delete a conversation

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agents.qa_agent import QAAgent
from app.documents.ingestion import get_document_metadata
from app.conversations import store as conversation_store

router = APIRouter()


# ----------------------------------------------------------------
# Request / response models
# ----------------------------------------------------------------

class StartConversationRequest(BaseModel):
    doc_ids: list[str]


class StartConversationResponse(BaseModel):
    conversation_id: str
    doc_ids: list[str]


class AskRequest(BaseModel):
    question: str


class SourceInfo(BaseModel):
    file_name: str
    chunk_index: int
    hybrid_score: float


class AskResponse(BaseModel):
    conversation_id: str
    answer: str
    standalone_query: str
    sources: list[SourceInfo]
    model: str
    duration_seconds: float


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    doc_ids: list[str]
    created_at: float
    messages: list[dict]


# ----------------------------------------------------------------
# Routes
# ----------------------------------------------------------------

@router.post("/conversations", response_model=StartConversationResponse)
async def start_conversation(request: StartConversationRequest):
    for doc_id in request.doc_ids:
        if get_document_metadata(doc_id) is None:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    agent = QAAgent()
    conversation_id = agent.start_conversation(request.doc_ids)

    return StartConversationResponse(
        conversation_id=conversation_id,
        doc_ids=request.doc_ids,
    )


@router.post("/conversations/{conversation_id}/ask", response_model=AskResponse)
async def ask(conversation_id: str, request: AskRequest):
    agent = QAAgent()
    try:
        result = agent.ask(conversation_id=conversation_id, question=request.question)
    except ValueError as e:
        # conversation not found / expired
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Q&A failed: {e}")

    return AskResponse(**result)


@router.get("/conversations/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(conversation_id: str):
    conversation = conversation_store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation '{conversation_id}' not found or expired."
        )
    return ConversationHistoryResponse(**conversation)


@router.get("/documents/{doc_id}/conversations")
async def list_conversations(doc_id: str):
    metadata = get_document_metadata(doc_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    conversation_ids = conversation_store.list_conversations_for_doc(doc_id)
    return {"doc_id": doc_id, "conversation_ids": conversation_ids}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    deleted = conversation_store.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation '{conversation_id}' not found."
        )
    return {"deleted": True, "conversation_id": conversation_id}


# NEW — add a document to an existing conversation
class AddDocumentRequest(BaseModel):
    doc_id: str

@router.post("/conversations/{conversation_id}/documents")
async def add_document(conversation_id: str, request: AddDocumentRequest):
    if get_document_metadata(request.doc_id) is None:
        raise HTTPException(status_code=404, detail=f"Document '{request.doc_id}' not found.")
    try:
        conversation_store.add_document(conversation_id, request.doc_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"conversation_id": conversation_id, "added": request.doc_id}