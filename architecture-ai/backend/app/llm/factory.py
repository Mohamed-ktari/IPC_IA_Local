# factory.py
# The only file in the entire project that knows which LLM provider exists.
# Every agent calls get_llm() and gets back a BaseLLM — never a concrete class.
# To add a new provider: add it here and nowhere else.

from app.llm.base import BaseLLM
from app.config import settings


def get_llm(agent_type: str = "default") -> BaseLLM:
    # agent_type lets specific agents request a different model
    # e.g. vision agent needs the multimodal model
    # e.g. coding agent could use a code-specialized model later

    if settings.LLM_PROVIDER == "ollama":
        from app.llm.ollama import OllamaLLM

        if agent_type == "vision":
            return OllamaLLM(model=settings.OLLAMA_VISION_MODEL)

        # All other agents use the default model for now
        # Later: return OllamaLLM(model=settings.OLLAMA_CODER_MODEL) for coding agent
        return OllamaLLM()

    elif settings.LLM_PROVIDER == "mistral":
        from app.llm.mistral import MistralLLM
        return MistralLLM()

    raise ValueError(
        f"Unknown LLM provider: '{settings.LLM_PROVIDER}'. "
        f"Valid options: 'ollama', 'mistral'"
    )