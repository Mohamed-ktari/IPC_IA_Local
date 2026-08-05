# section_processor.py

from app.documents.section_classifier import get_classifier
from app.documents.retrieval import get_retriever
from app.documents.doc_type import DocType
from app.documents.parsers.docx_parser import parse_section_content
from app.config import settings
from app.llm.base import Message, Role


def process_section(
    section: dict,
    project_id: str,
    global_prompt: str,
    llm,
    system_prompt: str,          # ADDED — was missing, referenced below but never received
) -> dict:
    """
    section: {"level", "title", "existing_content"} from parse_structure_docx()
    Returns: {
        "level", "title", "content",       # content = final reassembled text
        "lane", "classification_method",   # traceability
        "sources": [ {doc_id, file_name, chunk_index, lane}, ... ],  # traceability
    }
    """
    title = section["title"]
    segments = parse_section_content(section["existing_content"])

    guidance_segments = [s for s in segments if s["type"] == "guidance"]

    if not guidance_segments:
        return {
            "level": section["level"],
            "title": title,
            "content": section["existing_content"],
            "lane": None,
            "classification_method": None,
            "sources": [],
        }

    classification = get_classifier().classify(title)
    lane = classification["lane"]
    retriever = get_retriever()

    all_sources = []

    def generate_for_guidance(guidance_text: str) -> str:
        query = f"{title}. {guidance_text}"

        context = ""
        if lane == "memoire":
            results = retriever.retrieve(query, where={"doc_type": DocType.memoire.value})
            context = retriever.format_context(results)
            all_sources.extend(_to_source_records(results, "memoire"))

        elif lane == "programme":
            results = retriever.retrieve(
                query,
                where={
                    "$and": [
                        {"doc_type": DocType.programme.value},
                        {"project_id": project_id},
                    ]
                },
            )
            context = retriever.format_context(results)
            all_sources.extend(_to_source_records(results, "programme"))

        return _generate_section_text(
            llm=llm,
            title=title,
            guidance=guidance_text,
            context=context,
            global_prompt=global_prompt,
            system_prompt=system_prompt,   # now correctly in scope via the outer function's parameter
        )

    final_parts = []
    for seg in segments:
        if seg["type"] == "guidance":
            final_parts.append(generate_for_guidance(seg["content"]))
        else:
            final_parts.append(seg["content"])

    return {
        "level": section["level"],
        "title": title,
        "content": "\n\n".join(final_parts),
        "lane": lane,
        "classification_method": classification["method"],
        "sources": all_sources,
    }


def _to_source_records(results, lane: str) -> list[dict]:
    return [
        {
            "doc_id": r.doc_id,
            "file_name": r.file_name,
            "chunk_index": r.chunk_index,
            "lane": lane,
            "hybrid_score": round(r.hybrid_score, 3),
        }
        for r in results
    ]


def _generate_section_text(
    llm,
    title: str,
    guidance: str,
    context: str,
    global_prompt: str,
    system_prompt: str,
    max_tokens: int | None = None,   # FIXED — now has a default
) -> str:
    max_tokens = max_tokens or settings.DEFAULT_MAX_TOKENS
    prompt_parts = [
        f"Section à rédiger : \"{title}\"",
        f"Consigne spécifique de l'utilisateur pour cette section : {guidance}",
    ]
    if global_prompt:
        prompt_parts.append(f"Consigne générale du projet (à respecter dans le ton/l'accent, sans qu'elle dicte le contenu factuel) : {global_prompt}")
    if context:
        prompt_parts.append(f"Informations disponibles pour cette section :\n{context}")
    else:
        prompt_parts.append("Aucune information de référence disponible — rédige à partir de la consigne uniquement.")

    messages = [
        Message(role=Role.system, content=system_prompt),
        Message(role=Role.user, content="\n\n".join(prompt_parts)),
    ]

    response = llm.chat(messages, temperature=0.3, max_tokens=max_tokens)
    return response.content.strip()