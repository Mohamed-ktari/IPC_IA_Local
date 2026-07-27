# generation_agent.py
# Feature: fills an existing mémoire-technique structure doc with content,
# using per-section retrieval (mémoire lane / programme lane / none) and
# per-section user guidance (<<>> blocks), plus a project-wide steer.
#
# Does NOT build the structure — that's rc_agent + docx_writer's job.
# This agent only fills what's already there, via docx_writer.fill_memoire_docx.

import json
import time
from pathlib import Path

import redis

from app.agents.base_agent import BaseAgent, AgentResponse
from app.llm.base import Message, Role
from app.config import settings
from app.documents.parsers.docx_parser import parse_structure_docx
from app.documents.section_processor import process_section
from app.output.docx_writer import fill_memoire_docx

_redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)


def _set_job_status(job_id: str, status: dict):
    _redis_client.set(
        f"generation_job:{job_id}",
        json.dumps(status),
        ex=settings.CONVERSATION_TTL_SECONDS,  # reuse existing TTL setting —
                                                 # no need for a new config value
    )


def get_job_status(job_id: str) -> dict | None:
    raw = _redis_client.get(f"generation_job:{job_id}")
    return json.loads(raw) if raw else None


class GenerationAgent(BaseAgent):

    agent_type = "generation"
    description = (
        "Fills an existing mémoire technique structure document with "
        "generated content, using per-section retrieval from past "
        "mémoires and the current project's programme."
    )

    def run(
        self,
        structure_docx_path: str | Path,
        output_path: str | Path,
        project_id: str,
        global_prompt: str = "",
        job_id: str | None = None,
    ) -> AgentResponse:
        start = time.time()

        sections = parse_structure_docx(structure_docx_path)
        total = len(sections)

        if job_id:
            _set_job_status(job_id, {"status": "running", "current": 0, "total": total})

        filled_sections = []
        all_sources = []

        for i, section in enumerate(sections):
            result = process_section(
                section=section,
                project_id=project_id,
                global_prompt=global_prompt,
                llm=self.llm,
                system_prompt=self.system_prompt,
            )
            filled_sections.append(result)
            if result["sources"]:
                all_sources.append({
                    "section": result["title"],
                    "lane": result["lane"],
                    "sources": result["sources"],
                })

            if job_id:
                _set_job_status(job_id, {
                    "status": "running",
                    "current": i + 1,
                    "total": total,
                    "current_section": result["title"],
                })

        fill_memoire_docx(
            structure_docx_path=structure_docx_path,
            filled_sections=[
                {"level": s["level"], "title": s["title"], "content": s["content"]}
                for s in filled_sections
            ],
            output_path=output_path,
        )

        duration = time.time() - start
        self._log(
            f"generation_agent run — project {project_id}, {total} sections",
            f"OK — output at {output_path}",
            duration,
        )

        if job_id:
            _set_job_status(job_id, {
                "status": "done",
                "current": total,
                "total": total,
                "output_path": str(output_path),
            })

        return AgentResponse(
            content=str(output_path),
            agent_type=self.agent_type,
            model=self.llm.model,
            duration_seconds=duration,
            metadata={"sources": all_sources, "section_count": total},
        )
    # generation_agent.py — add near the bottom, alongside the class

def run_generation_job(
        structure_docx_path: str,
        output_path: str,
        project_id: str,
        global_prompt: str,
        job_id: str,
    ):
        # Module-level entry point for RQ — the worker process imports this
        # function by path ("app.agents.generation_agent.run_generation_job"),
        # so it must not depend on any in-memory state from the API process.
        agent = GenerationAgent()
        agent.run(
            structure_docx_path=structure_docx_path,
            output_path=output_path,
            project_id=project_id,
            global_prompt=global_prompt,
            job_id=job_id,
        )