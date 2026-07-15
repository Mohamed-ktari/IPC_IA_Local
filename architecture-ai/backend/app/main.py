# main.py
# FastAPI application factory.
# This file wires everything together — routers, middleware, startup/shutdown events.
# It does NOT contain any business logic.

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware




from app.config  import settings, ensure_directories
from app.api.routes import health, documents, agents, chat, search, rc, qa


# ------------------------------------------------------------------
# Lifespan — runs on startup and shutdown
# Everything that needs to be initialized before accepting requests
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"LLM provider : {settings.LLM_PROVIDER}")
    print(f"Model        : {settings.OLLAMA_MODEL}")

    # Create data directories if they don't exist
    ensure_directories()

    # TODO: initialize ChromaDB connection here (week 2)
    # TODO: warm up Ollama model with a dummy request (avoids cold start)

    yield  # app is running — handle requests

    # --- SHUTDOWN ---
    print("Shutting down cleanly...")
    # TODO: close DB connections here


# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.DEBUG else None,  # hide swagger in prod
        redoc_url=None,
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # CORS — allow the Next.js frontend to talk to this API
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Routers — each feature area has its own router
    # prefix groups all routes under a common path
    # ------------------------------------------------------------------
    app.include_router(health.router,     prefix="/health",    tags=["health"])
    app.include_router(documents.router,  prefix="/documents", tags=["documents"])
    app.include_router(agents.router,     prefix="/agents",    tags=["agents"])
    app.include_router(chat.router,       prefix="/chat",      tags=["chat"])
    app.include_router(search.router,     prefix="/search",    tags=["search"])
    app.include_router(rc.router,         prefix="/rc",        tags=["rc"])
    app.include_router(qa.router,         prefix="/qa",        tags=["qa"])

    return app


# Create the app instance
# Uvicorn imports this: uvicorn app.main:app
app = create_app()