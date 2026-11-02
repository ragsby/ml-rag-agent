"""Semantic cache: if a new question means the same as one we already answered
(cosine similarity >= threshold), return the saved result and skip the LLM.
Stored in its own small Qdrant collection."""

import json

from langsmith import traceable

from config import settings
from src.retrieval.vector_store import create_collection, get_vector_store, reset_collection

RAG_CACHE, AGENT_CACHE = "rag_cache", "agent_cache"  # separate: they store different result shapes


def _store(collection: str):
    create_collection(collection, hybrid=False)
    return get_vector_store("dense", collection=collection)


@traceable(name="cache_lookup")
def lookup(question: str, collection: str = RAG_CACHE) -> dict | None:
    """The saved result (e.g. {"answer": ..., "sources": [...]}) or None."""
    if not settings.enable_semantic_cache:
        return None
    hits = _store(collection).similarity_search_with_score(
        question, k=1, score_threshold=settings.semantic_cache_similarity_threshold
    )
    return json.loads(hits[0][0].metadata["result"]) if hits else None


def save(question: str, result: dict, collection: str = RAG_CACHE):
    if settings.enable_semantic_cache:
        _store(collection).add_texts([question], metadatas=[{"result": json.dumps(result)}])


def clear():
    for collection in (RAG_CACHE, AGENT_CACHE):
        reset_collection(collection)
