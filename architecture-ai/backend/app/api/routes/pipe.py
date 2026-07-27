"""
title: Architecture AI — Q&A Agent
author: you
description: Uploads files via the backend's ingestion pipeline and answers
  questions grounded in them, supporting multiple documents per conversation.
version: 0.2.0
"""

import base64
import httpx
import redis.asyncio as aioredis
from pydantic import BaseModel, Field


class Pipe:

    class Valves(BaseModel):
        BACKEND_BASE_URL: str = Field(
            default="http://backend:8000",
            description="Base URL of the FastAPI backend.",
        )
        REDIS_URL: str = Field(
            default="redis://redis:6379",
            description="Redis instance used to map Open WebUI chat_id -> backend conversation_id + doc_ids.",
        )
        REQUEST_TIMEOUT: int = Field(default=120)

    def __init__(self):
        self.valves = self.Valves()

    def _redis(self):
        return aioredis.from_url(self.valves.REDIS_URL, decode_responses=True)

    async def _get_conversation_id(self, chat_id: str) -> str | None:
        r = self._redis()
        return await r.get(f"openwebui_chat:{chat_id}:conversation_id")

    async def _set_conversation_id(self, chat_id: str, conversation_id: str):
        r = self._redis()
        await r.set(f"openwebui_chat:{chat_id}:conversation_id", conversation_id)

    async def _extract_files(self, body: dict) -> list[dict]:
        """
        Returns a list of {"filename": str, "content": bytes} for any files
        attached to this turn.

        UNVERIFIED — Open WebUI's exact file payload shape in `body` needs
        confirming against a real request (log body.keys() / body["files"]
        on a test upload) before trusting this in production. Checked here,
        in order of plausibility, are:
          - body["files"]: list of {"name"/"filename", "content"/"data" (base64)}
          - body["messages"][-1]["files"]: same shape, attached to the message
        Adjust once the real shape is confirmed.
        """
        files = body.get("files") or body.get("messages", [{}])[-1].get("files") or []
        extracted = []
        for f in files:
            name = f.get("filename") or f.get("name") or "upload.bin"
            raw = f.get("content") or f.get("data")
            if raw is None:
                continue
            if isinstance(raw, str):
                # assume base64-encoded if it's a string
                raw = base64.b64decode(raw)
            extracted.append({"filename": name, "content": raw})
        return extracted

    async def _upload_file(self, client: httpx.AsyncClient, filename: str, content: bytes) -> str:
        resp = await client.post(
            f"{self.valves.BACKEND_BASE_URL}/documents/upload",
            files={"file": (filename, content)},
        )
        resp.raise_for_status()
        return resp.json()["document"]["doc_id"]  # adjust key name if /documents/upload's
                                        # response shape differs — unverified,
                                        # documents.py wasn't shared

    async def pipe(self, body: dict) -> str:
        messages = body.get("messages", [])
        if not messages:
            return "Aucune question reçue."

        question = messages[-1].get("content", "")
        chat_id = body.get("chat_id") or body.get("id") or "default"

        async with httpx.AsyncClient(timeout=self.valves.REQUEST_TIMEOUT) as client:
            new_files = await self._extract_files(body)
            new_doc_ids = [
                await self._upload_file(client, f["filename"], f["content"])
                for f in new_files
            ]

            conversation_id = await self._get_conversation_id(chat_id)

            if conversation_id is None:
                if not new_doc_ids:
                    return (
                        "Merci de joindre au moins un document à votre premier "
                        "message pour démarrer une conversation."
                    )
                resp = await client.post(
                    f"{self.valves.BACKEND_BASE_URL}/qa/conversations",
                    json={"doc_ids": new_doc_ids},
                )
                resp.raise_for_status()
                conversation_id = resp.json()["conversation_id"]
                await self._set_conversation_id(chat_id, conversation_id)
            else:
                # Existing conversation — any newly attached files get added
                # to it rather than starting a new conversation.
                for doc_id in new_doc_ids:
                    add_resp = await client.post(
                        f"{self.valves.BACKEND_BASE_URL}/qa/conversations/{conversation_id}/documents",
                        json={"doc_id": doc_id},
                    )
                    add_resp.raise_for_status()

            resp = await client.post(
                f"{self.valves.BACKEND_BASE_URL}/qa/conversations/{conversation_id}/ask",
                json={"question": question},
            )

            if resp.status_code == 404:
                # Stale mapping (Redis TTL on the underlying conversation
                # elapsed server-side) — can't cleanly recover doc_ids here
                # since we didn't keep them client-side; surface clearly
                # rather than silently starting an empty new conversation.
                r = self._redis()
                await r.delete(f"openwebui_chat:{chat_id}:conversation_id")
                return (
                    "Cette conversation a expiré côté serveur. Merci de "
                    "rejoindre vos documents pour en démarrer une nouvelle."
                )

            resp.raise_for_status()
            result = resp.json()

        answer = result["answer"]
        sources = result.get("sources", [])
        if sources:
            source_lines = "\n".join(
                f"- {s['file_name']} (pertinence: {s['hybrid_score']:.0%})"
                for s in sources
            )
            answer += f"\n\n---\n**Sources consultées :**\n{source_lines}"

        return answer