# base.py
# Abstract interface that every LLM provider must implement.
# No business logic here — only the contract.
# Any file that needs an LLM imports BaseLLM and calls .chat() or .stream()
# It never knows or cares whether it's talking to Ollama, Mistral API, or anything else.

from abc import ABC, abstractmethod
from typing import Generator
from pydantic import BaseModel
from enum import Enum

class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    # Represents a single message in a conversation
    # role: "system" | "user" | "assistant"
    role: Role
    content: str


class LLMResponse(BaseModel):
    # Wraps every LLM response so callers always get a consistent object
    content: str          # the actual text response
    model: str            # which model produced it
    provider: str         # which provider was used
    prompt_tokens: int = 0
    completion_tokens: int = 0


class BaseLLM(ABC):

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        json_mode: bool = False,
        json_schema: dict | None = None,
    ) -> LLMResponse:
        # Sends a list of messages and returns a complete response.
        # Use this for: document analysis, proofreading, generation.
        # temperature 0.1-0.3 = focused/deterministic (good for technical docs)
        # temperature 0.7-0.9 = creative (good for generation tasks)
        pass

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        # Same as chat() but yields tokens one by one as they arrive.
        # Use this for: the chatbot UI so users see text appearing in real time.
        pass

    def build_messages(
        self,
        system_prompt: str,
        user_message: str,
        history: list[Message] | None = None,
    ) -> list[Message]:
        # Helper every agent will use to build a properly structured message list.
        # system prompt always goes first, then conversation history, then new message.
        messages = [Message(role=Role.system, content=system_prompt)]
        if history:
            messages.extend(history)
        messages.append(Message(role=Role.user, content=user_message))
        return messages