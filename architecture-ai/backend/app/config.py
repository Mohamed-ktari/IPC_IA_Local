# config.py
# Single source of truth for all settings.
# Every other file imports from here — never import os.environ directly elsewhere.

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

# Base directory of the project — used to build absolute paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):

    # ------------------------------------------------------------------
    # APP
    # ------------------------------------------------------------------
    APP_NAME: str = "Architecture AI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # --------------------------------------s----------------------------
    # LLM — change LLM_PROVIDER to switch between backends
    # "ollama" = local server (production)
    # "mistral" = Mistral API (fallback)
    # ------------------------------------------------------------------
    LLM_PROVIDER: str = "ollama"

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBEDDING_MODEL: str = "bge-m3"
    #OLLAMA_MODEL: str = "qwen2.5:14b"
    OLLAMA_MODEL: str = "qwen2.5:14b-instruct-q4_K_M"
    OLLAMA_VISION_MODEL: str = "qwen2.5vl:7b"  # multimodal — for vision agent
    OLLAMA_TIMEOUT: int = 120  # seconds — 14B can be slow on first token
    OLLAMA_NUM_CTX: int =8192

    # Mistral (fallback only)
    MISTRAL_API_KEY: str = ""
    MISTRAL_MODEL: str = "mistral-small-latest"

    # ------------------------------------------------------------------
    # VECTOR DATABASE — ChromaDB
    # ------------------------------------------------------------------
    CHROMA_PATH: str = ""  # set in .env as absolute path on the server
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001

    @property
    def chroma_path(self) -> Path:
        if self.CHROMA_PATH:
            return Path(self.CHROMA_PATH)
        return BASE_DIR / "backend" / "data" / "chroma"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # How long a Q&A conversation's history is kept in Redis before expiring.
    # Refreshed on every new message (see conversations/store.py) — so an
    # active conversation never expires mid-use, only ones abandoned for
    # this long. 7 days is a starting guess, not a validated number.
    CONVERSATION_TTL_SECONDS: int = 60 * 60 * 24 * 7

    # Postgres
    POSTGRES_USER: str = "archai"
    POSTGRES_PASSWORD: str = "changeme"
    POSTGRES_DB: str = "architecture_ai"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD}@localhost/"
            f"{self.POSTGRES_DB}"
        )


    # ------------------------------------------------------------------
    # DOCUMENT STORAGE
    # ------------------------------------------------------------------
    UPLOAD_PATH: str = ""

    @property
    def upload_path(self) -> Path:
        if self.UPLOAD_PATH:
            return Path(self.UPLOAD_PATH)
        return BASE_DIR / "backend" / "data" / "uploads"

    # ------------------------------------------------------------------
    # CHUNKING — tweak these when tuning RAG quality
    # ------------------------------------------------------------------
    CHUNK_SIZE: int = 512      # tokens per chunk
    CHUNK_OVERLAP: int = 64    # overlap between chunks to preserve context
    RETRIEVAL_TOP_K: int = 10   # how many chunks to retrieve per query (AnalysisAgent)

    # Q&A conversational retrieval — deliberately smaller than
    # RETRIEVAL_TOP_K above. AnalysisAgent wants comprehensive coverage for
    # a structured synthesis; Q&A wants a few focused chunks for a direct
    # answer. Starting value, not yet validated against real usage.
    QA_RETRIEVAL_TOP_K: int = 5
    QA_MIN_SCORE: float = 0.3

    # Generation defaults
    DEFAULT_TEMPERATURE: float = 0.1
    DEFAULT_MAX_TOKENS: int = 4096
    # ------------------------------------------------------------------
    # EXTERNAL SEARCH (regulatory agent only)
    # ------------------------------------------------------------------
    BRAVE_API_KEY: str = ""
    SEARCH_ENABLED: bool = False  # disabled until search gateway is built
    SEARCH_MAX_RESULTS: int = 5


    EXTRACTION_PAGES_PER_GROUP: int = 5
    EXTRACTION_WORDS_PER_PAGE: int = 300

    # ------------------------------------------------------------------
    # AUDIT / RGPD
    # ------------------------------------------------------------------
    AUDIT_LOG_PATH: str = ""

    @property
    def audit_log_path(self) -> Path:
        if self.AUDIT_LOG_PATH:
            return Path(self.AUDIT_LOG_PATH)
        return BASE_DIR / "backend" / "data" / "logs"

    # ------------------------------------------------------------------
    # SECURITY
    # ------------------------------------------------------------------
    SECRET_KEY: str = "change-this-in-production"
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]  # frontend URL

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"  # silently ignore unknown keys in .env


# Single instance imported everywhere
# Usage in any file: from app.config import settings
settings = Settings()


# Ensure required directories exist on startup
def ensure_directories():
    for path in [
        settings.chroma_path,
        settings.upload_path,
        settings.audit_log_path,
    ]:
        path.mkdir(parents=True, exist_ok=True)



if __name__ == "__main__":
    print("OLLAMA_MODEL =", settings.OLLAMA_MODEL)