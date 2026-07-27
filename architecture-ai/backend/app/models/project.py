# project.py
# Generic Project — the base entity every project type shares (name,
# client, status, timestamps). Project TYPES (mémoire, and whatever comes
# later) are modeled as separate subtype tables via joined-table
# inheritance, discriminated by `project_type`.
#
# A generic Project can hold MANY documents (ProjectDocument, one row per
# doc_id) — e.g. a future project type might ingest 10 reference PDFs.
# Mémoire projects specifically only ever need ONE programme doc, which is
# why that's its own scalar column on MemoireProjectDetail, not modeled
# through the many-documents table — see that class's docstring.

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base import Base


class Project(Base):
    __tablename__ = "projects"

    project_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_type = Column(String, nullable=False)  # discriminator — "memoire", future types...

    name = Column(String, nullable=False)
    client_name = Column(String, nullable=True)
    status = Column(String, nullable=False, default="active")  # active | archived

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    documents = relationship(
        "ProjectDocument", back_populates="project", cascade="all, delete-orphan"
    )

    __mapper_args__ = {
        "polymorphic_identity": "project",
        "polymorphic_on": project_type,
    }

    def to_dict(self) -> dict:
        return {
            "project_id": str(self.project_id),
            "project_type": self.project_type,
            "name": self.name,
            "client_name": self.client_name,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "doc_ids": [d.doc_id for d in self.documents],
        }


class ProjectDocument(Base):
    # Generic many-documents-per-project association — any project type
    # can attach an arbitrary number of ingested Chroma doc_ids here.
    __tablename__ = "project_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.project_id"), nullable=False)
    doc_id = Column(String, nullable=False)  # Chroma doc_id — not a real FK, lives outside Postgres
    doc_role = Column(String, nullable=True)  # optional free-form tag, e.g. "reference", "annexe"
    added_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    project = relationship("Project", back_populates="documents")


class MemoireProjectDetail(Base):
    # Mémoire-specific extension of Project — exactly ONE programme
    # document, modeled as a scalar column (not through ProjectDocument)
    # precisely because "at most one" is a real constraint for this
    # project type, not an arbitrary collection. Future project types with
    # different single-valued or multi-valued needs get their own
    # ProjectXDetail table alongside this one.
    __tablename__ = "memoire_project_details"

    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.project_id"), primary_key=True)
    programme_doc_id = Column(String, nullable=True)  # nullable — project can predate its programme upload

    project = relationship("Project", backref="memoire_detail")