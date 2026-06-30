# coding_agent.py
#
# Feature 5 — Assistance au développement et à l'automatisation
#
# The simplest agent in the system:
#   - Pure LLM, no RAG, no structured output, no multi-pass
#   - Stateful: conversation history is the core mechanic
#   - The user describes a need, gets code back, iterates via follow-ups
#
# What makes this agent useful vs a raw LLM call:
#   - System prompt tailored to the firm's stack and conventions
#   - Code block extraction so the caller can offer a clean "copy" or "export" button
#   - Language detection so the model always responds in the user's language
#   - Export helper that writes the extracted script to a .py file

import re
from dataclasses import dataclass, field
from pathlib import Path
from app.agents.base_agent import BaseAgent, AgentResponse
from app.llm.base import Message


# ──────────────────────────────────────────────
# Domain types
# ──────────────────────────────────────────────

@dataclass
class CodeBlock:
    language: str       # "python", "bash", "sql", etc.
    code: str           # raw code content, no fences


@dataclass
class CodingResult:
    # What the API route gets back on every turn
    raw_response: str           # full markdown response from the model
    code_blocks: list[CodeBlock] # extracted code blocks — ready for export or display
    agent_type: str
    model: str
    duration_seconds: float

    def primary_code(self) -> CodeBlock | None:
        # Convenience: return the first (usually only) code block
        # If the model returned multiple blocks, the caller decides what to do
        return self.code_blocks[0] if self.code_blocks else None


# ──────────────────────────────────────────────
# Code extraction
# ──────────────────────────────────────────────

def _extract_code_blocks(text: str) -> list[CodeBlock]:
    # Pulls all fenced code blocks out of the markdown response
    # Handles ```python, ```bash, ```sql, ``` (no language tag), etc.
    pattern = r"```([a-zA-Z0-9_+-]*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    blocks = []
    for language, code in matches:
        blocks.append(CodeBlock(
            language=language.strip().lower() or "text",
            code=code.strip(),
        ))
    return blocks


# ──────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────

class CodingAgent(BaseAgent):

    agent_type  = "coding"
    description = "Génère des scripts, requêtes et automatisations à partir d'une description en langage naturel."

    def run(
        self,
        user_message: str,
        history: list[Message] | None = None,
    ) -> CodingResult:
        """
        Main entry point. Called on every turn of the conversation.

        The caller is responsible for maintaining history between turns:

            history = []
            result1 = agent.run("Write a script to parse Excel files", history)
            history += [
                Message(role="user",      content="Write a script to parse Excel files"),
                Message(role="assistant", content=result1.raw_response),
            ]
            result2 = agent.run("Now add support for merged cells", history)
            # The model still has the full script in context — it will update it correctly
        """
        response: AgentResponse = self.chat(
            user_message=user_message,
            history=history,
            temperature=0.2,    # low: deterministic code, not creative prose
            max_tokens=4096,    # scripts can be long
        )

        code_blocks = _extract_code_blocks(response.content)

        return CodingResult(
            raw_response     = response.content,
            code_blocks      = code_blocks,
            agent_type       = response.agent_type,
            model            = response.model,
            duration_seconds = response.duration_seconds,
        )

    def export(
        self,
        code_block: CodeBlock,
        output_path: Path,
    ) -> Path:
        """
        Writes a single code block to a file.
        The API route calls this when the user clicks "Export".

        Extension is inferred from the language tag:
            python → .py
            bash   → .sh
            sql    → .sql
            other  → .txt
        """
        extension_map = {
            "python": ".py",
            "bash":   ".sh",
            "shell":  ".sh",
            "sql":    ".sql",
            "json":   ".json",
            "yaml":   ".yaml",
            "yml":    ".yaml",
        }
        ext = extension_map.get(code_block.language, ".txt")

        # If caller passed a directory, auto-name the file
        if output_path.is_dir():
            output_path = output_path / f"script_{self.agent_type}{ext}"
        elif output_path.suffix == "":
            output_path = output_path.with_suffix(ext)

        output_path.write_text(code_block.code, encoding="utf-8")
        return output_path