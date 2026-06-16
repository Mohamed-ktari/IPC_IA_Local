# base_agent.py
# Parent class that all agents inherit from.
# Handles everything that is common across all agents:
#   - loading the right system prompt from /llm/prompts/
#   - calling the LLM via the factory (never directly)
#   - streaming support
#   - error handling with consistent error messages
#   - audit logging (every agent action is logged for RGPD)
#
# To create a new agent:
#   1. Inherit from BaseAgent
#   2. Set agent_type and description
#   3. Implement the run() method
#   Nothing else is needed.

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator

from app.llm.factory import get_llm
from app.llm.base import BaseLLM, Message, LLMResponse
from app.config import settings

# Path to the prompt files
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"


class AgentResponse:
    # Consistent response object returned by every agent
    def __init__(
        self,
        content: str,
        agent_type: str,
        model: str,
        duration_seconds: float,
        metadata: dict | None = None,
    ):
        self.content = content
        self.agent_type = agent_type
        self.model = model
        self.duration_seconds = round(duration_seconds, 2)
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "agent_type": self.agent_type,
            "model": self.model,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):

    # Every subclass must define these two
    agent_type: str = "base"
    description: str = "Base agent"

    def __init__(self):
        self.llm: BaseLLM = get_llm(self.agent_type)
        self.system_prompt: str = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        # Loads the system prompt from /llm/prompts/<agent_type>.txt
        # Falls back to a generic prompt if the file doesn't exist yet
        prompt_file = PROMPTS_DIR / f"{self.agent_type}.txt"
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8").strip()

        # Fallback — lets you run agents before prompts are written
        return (
            "You are a helpful AI assistant for an architecture and engineering firm. "
            "Answer in the same language as the user. Be precise and professional."
        )

    def _build_messages(
        self,
        user_message: str,
        history: list[Message] | None = None,
        system_prompt_override: str | None = None,
    ) -> list[Message]:
        # Use override if provided — useful when an agent needs
        # a dynamic system prompt (e.g. injecting a template into the prompt)
        prompt = system_prompt_override or self.system_prompt
        return self.llm.build_messages(prompt, user_message, history)

    def chat(
    self,
    user_message: str,
    history: list[Message] | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    system_prompt_override: str | None = None,
    json_mode: bool = False,   
    ) -> AgentResponse:
        # Main entry point for all agents
        # Handles timing, error catching, and audit logging automatically
        start = time.time()
        try:
            messages = self._build_messages(
                user_message, history, system_prompt_override
            )
            llm_response: LLMResponse = self.llm.chat(
                messages, temperature=temperature, max_tokens=max_tokens,json_mode=json_mode,
            )
            duration = time.time() - start

            self._log(user_message, llm_response.content, duration)

            return AgentResponse(
                content=llm_response.content,
                agent_type=self.agent_type,
                model=llm_response.model,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start
            self._log(user_message, f"ERROR: {str(e)}", duration)
            raise RuntimeError(
                f"Agent '{self.agent_type}' failed after {duration:.1f}s: {str(e)}"
            )

    def stream(
        self,
        user_message: str,
        history: list[Message] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        system_prompt_override: str | None = None,
    ) -> Generator[str, None, None]:
        # Streaming version — yields tokens for real-time UI updates
        messages = self._build_messages(
            user_message, history, system_prompt_override
        )
        yield from self.llm.stream(
            messages, temperature=temperature, max_tokens=max_tokens
        )

    def _log(self, user_message: str, response: str, duration: float):
        # Minimal audit log — records the action without storing content
        # RGPD: we log WHO did WHAT and WHEN, not the actual document content
        try:
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "agent": self.agent_type,
                "duration_seconds": round(duration, 2),
                "message_length": len(user_message),
                "response_length": len(response),
                # NOTE: we intentionally do NOT log user_message or response content
                # Logging actual content would require explicit RGPD consent
            }
            log_file = settings.audit_log_path / "agent_calls.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                import json
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            # Logging must never crash the main application
            pass

    @abstractmethod
    def run(self, *args, **kwargs) -> AgentResponse:
        # Each agent implements this with its own specific logic
        # e.g. analysis_agent.run(document_text, template_name)
        # e.g. proofreading_agent.run(document_text)
        pass