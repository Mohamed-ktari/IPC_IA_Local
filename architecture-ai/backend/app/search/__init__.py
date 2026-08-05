# session.py
# Sync SQLAlchemy engine + session factory.
# Mirrors the sync style already used throughout the app (OllamaLLM's
# httpx.Client, ingestion.py, retrieval.py) — no async/sync mixing.

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    # FastAPI dependency — one session per request, always closed after.
    # Usage in a route: db: Session = Depends(get_db)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all_tables():
    # Dev-only schema setup — creates tables that don't exist yet.
    # Does NOT alter existing tables on schema changes (no migrations).
    # Move to Alembic once the schema stabilizes beyond active dev.
    from app.db.base import Base
    # Import every model module here so Base.metadata knows about it
    # before create_all() runs — SQLAlchemy only creates tables for
    # classes that have actually been imported/registered.
    from app.models import project  # noqa: F401

    Base.metadata.create_all(bind=engine)