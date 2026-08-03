# generation.py
# POST /generation/upload-structure  — upload the structure .docx to fill
# POST /generation/run               — enqueue filling as an RQ job, returns job_id
# GET  /generation/status/{job_id}   — poll progress (Redis key, set by the agent itself)
# GET  /generation/result/{job_id}   — download the filled .docx once done

import uuid
import shutil
import json as jsonlib
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import get_db
from app.models.project import Project
from app.agents.generation_agent import get_job_status, run_generation_job
from app.queue import generation_queue

router = APIRouter()

GENERATION_DIR = settings.upload_path / "_generation_jobs"


class RunGenerationRequest(BaseModel):
    job_id: str
    project_id: str
    global_prompt: str = ""


@router.post("/upload-structure")
def upload_structure(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".docx", ".doc")):
        raise HTTPException(status_code=400, detail="Expected a .docx file")

    job_id = str(uuid.uuid4())
    job_dir = GENERATION_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    structure_path = job_dir / "structure.docx"
    with open(structure_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"job_id": job_id, "structure_path": str(structure_path)}


@router.post("/run")
def run_generation(payload: RunGenerationRequest, db: Session = Depends(get_db)):
    job_dir = GENERATION_DIR / payload.job_id
    structure_path = job_dir / "structure.docx"

    if not structure_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No uploaded structure found for this job_id — call /upload-structure first",
        )

    # Validate project_id exists before enqueueing — fail fast rather than
    # discovering it's invalid partway through a background job.
    try:
        project_uuid = uuid.UUID(payload.project_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="project_id is not a valid UUID")

    project = db.query(Project).filter(Project.project_id == project_uuid).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    output_path = job_dir / "filled.docx"

    generation_queue.enqueue(
        run_generation_job,
        kwargs={
            "structure_docx_path": str(structure_path),
            "output_path": str(output_path),
            "project_id": payload.project_id,
            "global_prompt": payload.global_prompt,
            "job_id": payload.job_id,
        },
        job_id=payload.job_id,
        job_timeout="30m",  # generation over many sections can be slow on a 14B model
    )

    return {"job_id": payload.job_id, "status": "queued"}


@router.get("/status/{job_id}")
def get_status(job_id: str):
    status = get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Unknown job_id, or job expired")
    return status


@router.get("/result/{job_id}")
def get_result(job_id: str):
    status = get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Unknown job_id, or job expired")
    if status.get("status") != "done":
        raise HTTPException(status_code=409, detail=f"Job not finished yet — status: {status.get('status')}")

    output_path = Path(status["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=410, detail="Output file no longer available")

    return FileResponse(
        output_path,
        filename="memoire_rempli.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )