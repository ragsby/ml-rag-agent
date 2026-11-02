"""LLM evaluation (LLM-as-a-judge). For each golden question we run the RAG
chain, then a judge LLM scores the answer:

  faithfulness (1-5) : is every claim supported by the retrieved context? (hallucination check)
  relevance    (1-5) : does the answer actually answer the question?
  refused      (y/n) : did it say the documents don't contain the answer?

Off-topic questions ("answerable": false) SHOULD be refused -- the bot must only
answer from the documents.
"""

import json

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agent.llm import get_llm
from src.evaluation.retrieval_eval import DATASET_PATH
from src.retrieval.rag import answer_question


class Grade(BaseModel):
    faithfulness: int = Field(ge=1, le=5, description="5 = every claim is supported by the context")
    relevance: int = Field(ge=1, le=5, description="5 = fully answers the question")
    refused: bool = Field(description="True if the answer says the context/documents don't contain the answer")
    reason: str = Field(description="One sentence explaining the scores")


JUDGE_PROMPT = ChatPromptTemplate.from_template("""\
You are grading an answer produced by a RAG system.

Question: {question}

Retrieved context:
{context}

Answer: {answer}

Score faithfulness (is the answer supported ONLY by the context?) and relevance (does it answer the question?), 1-5 each.
Set refused=true if the answer declines because the context doesn't contain the information.
""")


def judge(question: str, answer: str, context: str) -> Grade:
    # Retries: some models occasionally return a malformed grade, which the provider rejects
    chain = (JUDGE_PROMPT | get_llm().with_structured_output(Grade)).with_retry(stop_after_attempt=3)
    return chain.invoke({"question": question, "answer": answer, "context": context})


def run_llm_eval() -> list[dict]:
    rows = []
    for item in json.loads(DATASET_PATH.read_text()):
        result = answer_question(item["query"], use_cache=False)
        grade = judge(item["query"], result["answer"], result["context"])
        rows.append({"query": item["query"], "answerable": item.get("answerable", True),
                     "answer": result["answer"], **grade.model_dump()})
    return rows


def print_report(rows: list[dict]):
    on_topic = [r for r in rows if r["answerable"]]
    off_topic = [r for r in rows if not r["answerable"]]
    n = len(on_topic) or 1
    print(f"\nOn-topic ({len(on_topic)}):  faithfulness {sum(r['faithfulness'] for r in on_topic) / n:.2f}/5"
          f"  relevance {sum(r['relevance'] for r in on_topic) / n:.2f}/5"
          f"  wrongly refused {sum(r['refused'] for r in on_topic)}")
    print(f"Off-topic ({len(off_topic)}): correctly refused {sum(r['refused'] for r in off_topic)}/{len(off_topic)}\n")
    for r in rows:
        tag = "OFF" if not r["answerable"] else f"F{r['faithfulness']} R{r['relevance']}"
        print(f"  [{tag}{' refused' if r['refused'] else ''}] {r['query']}\n      {r['reason']}")
