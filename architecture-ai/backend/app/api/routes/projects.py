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


@router.post("/{project_id}/documents")
def add_document(
    project_id: uuid.UUID, payload: AddDocumentRequest, db: Session = Depends(get_db)
):
    project = db.query(Project).filter(Project.project_id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    doc = ProjectDocument(
        project_id=project_id,
        doc_id=payload.doc_id,
        doc_type=payload.doc_type.value,
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