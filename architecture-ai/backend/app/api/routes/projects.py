# projects.py
# POST /projects        — create a project
# GET  /projects         — list all projects
# GET  /projects/{id}    — get one project
# PATCH /projects/{id}   — update programme_doc_id / status / name
#
# Minimal on purpose — no auth/ownership fields yet, matches the rest of
# the API's current scope. Extend when multi-user access matters.

import uuid
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.session import get_db
from app.models.project import Project

router = APIRouter()


class ProjectCreate(BaseModel):
    name: str
    client_name: str | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    client_name: str | None = None
    programme_doc_id: str | None = None
    status: str | None = None


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