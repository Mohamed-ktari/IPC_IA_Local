# ollama.py
# Ollama implementation of BaseLLM.
# Talks to the Ollama container running on the server.
# All other code is completely unaware this file exists —
# they only interact with the BaseLLM interface.

import httpx
from typing import Generator
from app.llm.base import BaseLLM, Message, LLMResponse
from app.config import settings


class OllamaLLM(BaseLLM):

    def __init__(self, model: str | None = None):
        # Allow overriding model per agent (e.g. vision agent uses different model)
        self.model = model or settings.OLLAMA_MODEL
        self.base_url = settings.OLLAMA_BASE_URL
        self.timeout = settings.OLLAMA_TIMEOUT

    def _format_messages(self, messages: list[Message]) -> list[dict]:
        # Ollama expects plain dicts, not Pydantic models
        return [{"role": m.role, "content": m.content} for m in messages]

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        try:
            payload = {
                "model": self.model,
                "messages": self._format_messages(messages),
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            if json_mode:
                payload["format"] = "json"   # forces valid JSON output

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                return LLMResponse(
                    content=data["message"]["content"],
                    model=self.model,
                    provider="ollama",
                    prompt_tokens=data.get("prompt_eval_count", 0),
                    completion_tokens=data.get("eval_count", 0),
                )
        except httpx.TimeoutException:
            raise RuntimeError(f"Ollama timed out after {self.timeout}s")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama HTTP error: {e.response.status_code} — {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {str(e)}")

    def stream(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": self._format_messages(messages),
                        "stream": True,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                    },
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            import json
                            chunk = json.loads(line)
                            if not chunk.get("done"):
                                yield chunk["message"]["content"]

        except Exception as e:
            raise RuntimeError(f"Ollama stream error: {str(e)}")