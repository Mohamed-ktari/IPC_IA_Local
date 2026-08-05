# rc.py
# HTTP routes for RC (règlement de consultation) analysis.
#
# Distinct from agents.py's /analyze endpoint: that one runs template-driven
# RAG synthesis over ingested chunks. This one needs the document's FULL
# text (map-reduce summary + heading-based structure extraction — see
# rc_agent.py and section_matcher.py for why RAG isn't used here).
#
# Endpoint:
#   POST /rc/analyze — summary + mémoire structure extraction + generated
#                       .docx skeleton, all in one call.
#
# Docx generation is bundled directly into /analyze rather than a separate
# editable-structure-text step: the target users aren't technical, and
# letting them hand-edit raw structure text before generation is a good way
# to introduce parsing errors. Simpler and safer: generate the file
# immediately, let them fix anything wrong directly in Word afterward —
# a tool they already know.

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.agents.rc_agent import RCAgent
from app.documents.ingestion import get_document_metadata, get_document_full_text
from app.output.docx_writer import generate_memoire_docx
from app.config import settings

router = APIRouter()

# Company template — title/date/logo + Titre1..9 styles, fixed chapters
# get replaced dynamically. Update this path if the template moves.
TEMPLATE_PATH = Path(__file__).resolve().parent.parent.parent / "output" / "templates" / "memoire_template.docx"


# ----------------------------------------------------------------
# Request / response models
# ----------------------------------------------------------------

class RCAnalyzeRequest(BaseModel):
    doc_id: str
    user_prompt: str = ""


class StructureResult(BaseModel):
    found: bool
    heading: str | None
    match_score: float | None
    raw_text: str | None
    structure: str | None
    source: str | None = None


class IntentResult(BaseModel):
    target_section_label: str | None
    emphasis_instructions: str | None


class RCAnalyzeResponse(BaseModel):
    doc_id: str
    agent_type: str
    model: str
    duration_seconds: float
    summary: str
    structure: StructureResult
    intent: IntentResult
    docx_path: str | None  # None if generation failed or structure wasn't found
    docx_error: str | None  # populated only if generation was attempted and failed


# ----------------------------------------------------------------
# Routes
# ----------------------------------------------------------------

@router.post("/analyze", response_model=RCAnalyzeResponse)
async def analyze_rc(request: RCAnalyzeRequest):
    # Runs RCAgent on an already-ingested RC document: general summary
    # (map-reduce, optionally weighted by the user's prompt), mémoire
    # structure extraction (heading match, no RAG), and the generated
    # .docx skeleton — all in one response.

    metadata = get_document_metadata(request.doc_id)
    if metadata is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.doc_id}' not found. Upload it first via /documents/upload."
        )

    document_markdown = get_document_full_text(request.doc_id)
    if not document_markdown:
        raise HTTPException(
            status_code=422,
            detail=f"No parsed text available for document '{request.doc_id}'."
        )

    try:
        agent = RCAgent()
        response = agent.run(
            document_markdown=document_markdown,
            user_prompt=request.user_prompt,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RC analysis failed: {e}")

    result = response.to_dict()

    # Docx generation is best-effort: if the structure section wasn't found,
    # or generation fails for some other reason, the summary/structure the
    # LLM already produced is still returned — the whole request shouldn't
    # fail just because the Word file couldn't be built.
    docx_path = None
    docx_error = None
    structure_text = result["structure"].get("structure")

    if structure_text:
        if not TEMPLATE_PATH.exists():
            docx_error = (
                f"Mémoire template not found at {TEMPLATE_PATH} — "
                f"copy the company template there (see docx_writer.py)."
            )
        else:
            try:
                doc_dir = settings.upload_path / request.doc_id
                output_path = doc_dir / "memoire_skeleton.docx"
                generate_memoire_docx(
                    template_path=TEMPLATE_PATH,
                    structure_text=structure_text,
                    output_path=output_path,
                )
                docx_path = str(output_path)
                if result["structure"].get("source") == "rag_fallback":
                    docx_error = (
                        "Structure générée par recherche approximative (le titre de "
                        "section n'a pas été trouvé directement) — vérifiez le "
                        "contenu généré avant envoi."
                    )
            except Exception as e:
                docx_error = f"Docx generation failed: {e}"
    else:
        docx_error = "No structure section was found in the RC — skeleton not generated."

    return RCAnalyzeResponse(
        doc_id=request.doc_id,
        agent_type=result["agent_type"],
        model=result["model"],
        duration_seconds=result["duration_seconds"],
        summary=result["summary"],
        structure=StructureResult(**result["structure"]),
        intent=IntentResult(**result["intent"]),
        docx_path=docx_path,
        docx_error=docx_error,
    )


@router.get("/download/{doc_id}")
async def download_memoire_docx(doc_id: str):
    # Serves the mémoire skeleton generated by /analyze for this doc_id.
    # Frontend: point a download button/link at GET /rc/download/{doc_id}
    # once RCAnalyzeResponse.docx_path is non-null.

    metadata = get_document_metadata(doc_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    output_path = settings.upload_path / doc_id / "memoire_skeleton.docx"
    if not output_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No generated mémoire found for '{doc_id}'. Run /rc/analyze first."
        )

    return FileResponse(
        path=output_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename="memoire_skeleton.docx",
    )