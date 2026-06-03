# health.py
# GET /health — checks that all critical services are reachable.
# Used by Docker healthcheck and by the frontend to show system status.

import httpx
from fastapi import APIRouter
from pydantic import BaseModel
from app.config import settings

router = APIRouter()


class ServiceStatus(BaseModel):
    status: str        # "ok" or "error"
    detail: str = ""


class HealthResponse(BaseModel):
    app: str
    version: str
    ollama: ServiceStatus
    chromadb: ServiceStatus


async def check_ollama() -> ServiceStatus:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                return ServiceStatus(status="ok", detail=f"models: {', '.join(models)}")
            return ServiceStatus(status="error", detail=f"status {r.status_code}")
    except Exception as e:
        return ServiceStatus(status="error", detail=str(e))


async def check_chromadb() -> ServiceStatus:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(
                f"http://{settings.CHROMA_HOST}:{settings.CHROMA_PORT}/api/v2/heartbeat"
            )
            if r.status_code == 200:
                # Also check collections endpoint
                r2 = await client.get(
                    f"http://{settings.CHROMA_HOST}:{settings.CHROMA_PORT}"
                    f"/api/v2/tenants/default_tenant/databases/default_database/collections"
                )
                count = len(r2.json()) if r2.status_code == 200 else "?"
                return ServiceStatus(
                    status="ok",
                    detail=f"chromadb reachable — {count} collection(s)"
                )
            return ServiceStatus(status="error", detail=f"status {r.status_code}")
    except Exception as e:
        return ServiceStatus(status="error", detail=str(e))


@router.get("", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        app=settings.APP_NAME,
        version=settings.APP_VERSION,
        ollama=await check_ollama(),
        chromadb=await check_chromadb(),
    )