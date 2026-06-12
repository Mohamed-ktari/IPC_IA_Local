# agents.py
# HTTP routes for running AI agents on ingested documents.
#
# Currently only the analysis_agent is wired (Feature 1).
# Other agents (generation, proofreading, etc.) will follow
# the exact same pattern: thin route → agent.run_xxx()
#
# Endpoints:
#   GET  /agents              — list available agents and templates
#   POST /agents/analyze      — run document analysis with a template

from pathlib import Path
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agents.analysis_agent import AnalysisAgent
from app.documents.ingestion import get_document_metadata

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"


# ----------------------------------------------------------------
# Request / response models
# ----------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    doc_id: str
    template_name: str = "amiante"
    additional_instructions: str = ""


class AnalyzeResponse(BaseModel):
    doc_id: str
    template_name: str
    agent_type: str
    model: str
    duration_seconds: float
    content: str


class AgentInfo(BaseModel):
    agent_type: str
    description: str


class TemplateInfo(BaseModel):
    name: str
    file_name: str


class AgentsListResponse(BaseModel):
    agents: list[AgentInfo]
    available_templates: list[TemplateInfo]


# ----------------------------------------------------------------
# Routes
# ----------------------------------------------------------------

@router.get("", response_model=AgentsListResponse)
async def get_agents():
    # Returns available agents and templates — used by the frontend
    # to build the agent/template selection UI.

    agents = [
        AgentInfo(
            agent_type=AnalysisAgent.agent_type,
            description=AnalysisAgent.description,
        ),
        # Future agents added here as they're built:
        # AgentInfo(agent_type="generation", description="..."),
        # AgentInfo(agent_type="proofreading", description="..."),
    ]

    templates = []
    for f in TEMPLATES_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            templates.append(TemplateInfo(name=data.get("name", f.stem), file_name=f.stem))
        except Exception:
            continue

    return AgentsListResponse(agents=agents, available_templates=templates)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_document(request: AnalyzeRequest):
    # Runs the analysis agent in RAG mode on an already-ingested document.

    # Verify document exists
    metadata = get_document_metadata(request.doc_id)
    if metadata is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.doc_id}' not found. Upload it first via /documents/upload."
        )

    # Verify template exists
    template_path = TEMPLATES_DIR / f"{request.template_name}.json"
    if not template_path.exists():
        available = [f.stem for f in TEMPLATES_DIR.glob("*.json")]
        raise HTTPException(
            status_code=400,
            detail=f"Template '{request.template_name}' not found. Available: {available}"
        )

    try:
        agent = AnalysisAgent()
        response = agent.run_rag(
            doc_id=request.doc_id,
            template_name=request.template_name,
            additional_instructions=request.additional_instructions,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return AnalyzeResponse(
        doc_id=request.doc_id,
        template_name=request.template_name,
        agent_type=response.agent_type,
        model=response.model,
        duration_seconds=response.duration_seconds,
        content=response.content,
    )