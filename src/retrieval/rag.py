"""Plain RAG: retrieve -> put chunks in the prompt -> LLM answers with citations.
Same as `answer_question` in the combined chunking notebook. This is what the chat UI uses."""

from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from src.agent import cache
from src.agent.llm import get_llm
from src.retrieval.search import format_docs, search

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions using ONLY the context below, which comes from the user's machine learning documents. "
     "Cite every fact as [file, page]. If the context does not contain the answer, reply that the documents "
     "don't cover this question -- do not use outside knowledge and do not guess.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


@traceable(name="rag_answer")
def answer_question(question: str, k: int = 5, source: str | None = None, use_cache: bool = True) -> dict:
    """Returns {"answer", "sources": [{"text", "source", "page", "score"}], "context", "cached"}."""
    use_cache = use_cache and source is None  # a cached answer may come from a different PDF filter
    if use_cache and (cached := cache.lookup(question)):
        return {**cached, "cached": True}

    docs = search(question, k=k, source=source)
    context = format_docs(docs)
    answer = get_llm().invoke(RAG_PROMPT.format_messages(context=context, question=question)).text
    result = {
        "answer": answer,
        "sources": [{"text": d.page_content, **{key: d.metadata.get(key) for key in ("source", "page", "score")}}
                    for d in docs],
        "context": context,
    }
    if use_cache:
        cache.save(question, result)
    return {**result, "cached": False}
