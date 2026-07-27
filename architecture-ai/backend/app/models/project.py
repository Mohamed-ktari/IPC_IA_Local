# project.py
# Flat project model — a Project is just an identity (name, client, status,
# timestamps) that can hold an arbitrary number of documents, each tagged
# with a doc_type. No project-type subclassing: a "mémoire project" is
# just a Project whose documents happen to include one doc_type="programme"
# row — nothing structurally special is needed to support that, and future
# project needs (multiple programmes, reference docs, whatever) are just
# more ProjectDocument rows with a different doc_type, no schema change.

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base import Base


class Project(Base):
    __tablename__ = "projects"

    project_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
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

    def to_dict(self) -> dict:
        return {
            "project_id": str(self.project_id),
            "name": self.name,
            "client_name": self.client_name,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "documents": [d.to_dict() for d in self.documents],
        }


class ProjectDocument(Base):
    # One row per document attached to a project. doc_type mirrors the
    # DocType enum used at ingestion time (documents/doc_type.py) — kept as
    # a plain string column here (not a Postgres ENUM) so new doc_types can
    # be introduced without a migration; validate against the DocType enum
    # at the API layer instead.
    __tablename__ = "project_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.project_id"), nullable=False)
    doc_id = Column(String, nullable=False)   # Chroma doc_id — not a real FK, lives outside Postgres
    doc_type = Column(String, nullable=False)  # "programme" | "memoire" | "rc" | "unspecified" | ...
    added_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    project = relationship("Project", back_populates="documents")

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "added_at": self.added_at.isoformat() if self.added_at else None,
        }