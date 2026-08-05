# projects.py
# POST   /projects                          — create a project
# GET    /projects                          — list all projects
# GET    /projects/{project_id}              — get one project (with its documents)
# PATCH  /projects/{project_id}              — update name/client_name/status
# POST   /projects/{project_id}/documents     — attach a document to a project
# DELETE /projects/{project_id}/documents/{doc_id} — detach a document
#
# Minimal on purpose — no auth/ownership fields yet, matches the rest of
# the API's current scope.

import uuid
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.session import get_db
from app.models.project import Project, ProjectDocument
from app.documents.doc_type import DocType

router = APIRouter()


class ProjectCreate(BaseModel):
    name: str
    client_name: str | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    client_name: str | None = None
    status: str | None = None


class AddDocumentRequest(BaseModel):
    doc_id: str
    original_file_name: str | None = None
    doc_type: DocType


@router.post("")
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)):
    project = Project(name=payload.name, client_name=payload.client_name)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project.to_dict()


@router.get("")
def list_projects(db: Session = Depends(get_db)):
    projects = db.query(Project).order_by(Project.created_at.desc()).all()
    return [p.to_dict() for p in projects]


@router.get("/{project_id}")
def get_project(project_id: uuid.UUID, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.project_id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project.to_dict()


@router.patch("/{project_id}")
def update_project(
    project_id: uuid.UUID, payload: ProjectUpdate, db: Session = Depends(get_db)
):
    project = db.query(Project).filter(Project.project_id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(project, field, value)

    db.commit()
    db.refresh(project)
    return project.to_dict()


from fastapi import UploadFile, File, Form
from app.documents.ingestion import ingest_document
import tempfile
import shutil
from pathlib import Path


@router.post("/{project_id}/documents")
async def add_document(
    project_id: uuid.UUID,
    file: UploadFile = File(...),
    doc_type: DocType = Form(...),
    db: Session = Depends(get_db),
):
    # 1. Check project exists
    project = db.query(Project).filter(Project.project_id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # 2. Validate file type
    ALLOWED_SUFFIXES = (".pdf", ".docx", ".doc")
    if not file.filename.lower().endswith(ALLOWED_SUFFIXES):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(ALLOWED_SUFFIXES)}"
        )

    # 3. Save temporarily and ingest
    suffix = Path(file.filename).suffix.lower()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    try:
        metadata = ingest_document(
            file_path=tmp_path,
            original_filename=file.filename,
            uploaded_by="api_user",
            doc_type=doc_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    # 4. Attach to project
    doc = ProjectDocument(
        project_id=project_id,
        doc_id=metadata["doc_id"],
        doc_type=doc_type.value,
        original_file_name=file.filename,   # or metadata.get("original_file_name")
    )
    db.add(doc)
    db.commit()
    db.refresh(project)

    return project.to_dict()


@router.delete("/{project_id}/documents/{doc_id}")
def remove_document(project_id: uuid.UUID, doc_id: str, db: Session = Depends(get_db)):
    doc = (
        db.query(ProjectDocument)
        .filter(ProjectDocument.project_id == project_id, ProjectDocument.doc_id == doc_id)
        .first()
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not attached to this project")

    db.delete(doc)
    db.commit()
    return {"deleted": True, "project_id": str(project_id), "doc_id": doc_id}


@router.delete("/{project_id}")
def delete_project(project_id: uuid.UUID, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.project_id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    db.delete(project)
    db.commit()

    return {
        "deleted": True,
        "project_id": str(project_id),
        "message": "Project deleted successfully"
    }