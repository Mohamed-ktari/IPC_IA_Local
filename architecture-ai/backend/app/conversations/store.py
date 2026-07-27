# conversation_store.py
# Redis-backed storage for Q&A conversations.
#
# One document can have MULTIPLE separate conversations (each gets its own
# conversation_id) — a user might ask about deadlines in one thread and
# about technical requirements in another, without them bleeding together.
#
# Key layout:
#   conversation:{conversation_id}   -> JSON: {conversation_id, doc_id,
#                                               created_at, messages: [...]}
#   doc_conversations:{doc_id}       -> Redis SET of conversation_ids
#                                       (lets the frontend list past
#                                       conversations for a document)
#
# IMPORTANT: "messages" stores only raw {"role", "content"} question/answer
# pairs — never the retrieval-augmented prompt QAAgent builds per turn.
# Storing the augmented version would make history balloon with redundant
# retrieved chunks every turn.
#
# TTL: conversations expire after settings.CONVERSATION_TTL_SECONDS of
# inactivity, refreshed on every new turn — an active conversation never
# expires mid-use. The doc_conversations SET itself has no TTL (it's just
# an index of IDs, harmless to keep even after individual conversations
# expire — get_conversation() on an expired ID simply returns None).

import json
import time
import uuid

import redis

from app.config import settings

CONVERSATION_KEY = "conversation:{conversation_id}"
DOC_CONVERSATIONS_KEY = "doc_conversations:{doc_id}"

_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _client


def create_conversation(doc_ids: list[str]) -> str:
    conversation_id = str(uuid.uuid4())
    client = get_redis_client()
    data = {
        "conversation_id": conversation_id,
        "doc_ids": doc_ids,
        "created_at": time.time(),
        "messages": [],
    }
    client.set(
        CONVERSATION_KEY.format(conversation_id=conversation_id),
        json.dumps(data),
        ex=settings.CONVERSATION_TTL_SECONDS,
    )
    for doc_id in doc_ids:
        client.sadd(
            DOC_CONVERSATIONS_KEY.format(doc_id=doc_id),
            conversation_id,
        )
    return conversation_id

def add_document(conversation_id: str, doc_id: str) -> None:
    # NEW — appends a doc_id to an existing conversation, for when a
    # second file is uploaded mid-chat rather than at conversation start.
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation '{conversation_id}' not found")
    if doc_id not in conversation["doc_ids"]:
        conversation["doc_ids"].append(doc_id)
        _save_conversation(conversation_id, conversation)  # however store persists updates
        client = get_redis_client()
        client.sadd(DOC_CONVERSATIONS_KEY.format(doc_id=doc_id), conversation_id)  # NEW

def get_conversation(conversation_id: str) -> dict | None:
    client = get_redis_client()
    raw = client.get(CONVERSATION_KEY.format(conversation_id=conversation_id))
    if raw is None:
        return None
    return json.loads(raw)


def append_turn(conversation_id: str, question: str, answer: str) -> None:
    client = get_redis_client()
    key = CONVERSATION_KEY.format(conversation_id=conversation_id)
    raw = client.get(key)
    if raw is None:
        raise ValueError(f"Conversation '{conversation_id}' not found")
    data = json.loads(raw)
    data["messages"].append({"role": "user", "content": question})
    data["messages"].append({"role": "assistant", "content": answer})
    client.set(key, json.dumps(data), ex=settings.CONVERSATION_TTL_SECONDS)


def list_conversations_for_doc(doc_id: str) -> list[str]:
    client = get_redis_client()
    return list(client.smembers(DOC_CONVERSATIONS_KEY.format(doc_id=doc_id)))


def delete_conversation(conversation_id: str) -> bool:
    client = get_redis_client()
    key = CONVERSATION_KEY.format(conversation_id=conversation_id)
    raw = client.get(key)
    if raw is None:
        return False
    data = json.loads(raw)
    client.delete(key)
    for doc_id in data["doc_ids"]:
        client.srem(DOC_CONVERSATIONS_KEY.format(doc_id=doc_id), conversation_id)
    return True

def _save_conversation(conversation_id: str, data: dict) -> None:
    client = get_redis_client()
    client.set(
        CONVERSATION_KEY.format(conversation_id=conversation_id),
        json.dumps(data),
        ex=settings.CONVERSATION_TTL_SECONDS,
    )