"""LangSmith experiment: run the chat agent over the test questions, score every answer,
and save it all as a named run you can compare with other runs in the LangSmith website.

Run: uv run python run_server.py eval --langsmith --experiment groq-qwen
Change the model or prompt, run again with another name, then compare the two.
Needs a LangSmith Personal Access Token (a "tracing" key can't create datasets).
"""

import asyncio
import json

from langsmith import Client, aevaluate
from langsmith.utils import LangSmithError

from src.agent.agent import ask_agent
from src.evaluation.llm_eval import judge
from src.evaluation.retrieval_eval import DATASET_PATH

DATASET_NAME = "ml-rag-golden"

# Database questions with the value the answer must contain (checked against the sample data)
SQL_QUESTIONS = [
    ("What accuracy did the Gaussian Naive Bayes model get on the digits dataset?", "0.807"),
    ("Which model has the highest accuracy on the digits dataset, and what is that accuracy?", "0.991"),
    ("What R2 score did the Random Forest Regressor get on california_housing?", "0.805"),
]


def build_examples() -> list[dict]:
    """The golden questions + the database questions, in LangSmith's inputs/outputs format."""
    examples = [
        {"inputs": {"question": item["query"]},
         "outputs": {"answerable": item.get("answerable", True), "source": item.get("source"),
                     "expected_pages": item.get("expected_pages")}}
        for item in json.loads(DATASET_PATH.read_text())
    ]
    examples += [{"inputs": {"question": q}, "outputs": {"answerable": True, "expected_value": value}}
                 for q, value in SQL_QUESTIONS]
    return examples


def upload_dataset(client: Client):
    """Create the dataset once; later runs only add questions that are new."""
    if not client.has_dataset(dataset_name=DATASET_NAME):
        client.create_dataset(DATASET_NAME, description="ML RAG agent test questions")
    existing = {e.inputs["question"] for e in client.list_examples(dataset_name=DATASET_NAME)}
    new = [e for e in build_examples() if e["inputs"]["question"] not in existing]
    if new:
        client.create_examples(dataset_name=DATASET_NAME, examples=new)


async def target(inputs: dict) -> dict:
    """The thing being tested: the same agent the chat page uses, without the cache."""
    return await ask_agent(inputs["question"], allow_cache=False)


def evidence_text(outputs: dict) -> str:
    """Everything the agent's tools returned -- what the judge checks the answer against."""
    parts = [s["text"] for s in outputs.get("sources", [])]
    parts += [f"SQL: {q['sql']}\nRows: {json.dumps(q['rows'], default=str)}" for q in outputs.get("sql", [])]
    return "\n\n---\n\n".join(parts)


def score_answer(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """All the scores for one answer (each one is 0-1, higher is better)."""
    answer = outputs["answer"]
    grade = judge(inputs["question"], answer, evidence_text(outputs))
    if reference_outputs.get("answerable"):
        results = [{"key": "faithfulness", "score": grade.faithfulness / 5, "comment": grade.reason},
                   {"key": "relevance", "score": grade.relevance / 5, "comment": grade.reason}]
    else:
        results = [{"key": "correct_refusal", "score": int(grade.refused), "comment": grade.reason}]

    pages = reference_outputs.get("expected_pages")
    if pages:
        expected = {(reference_outputs["source"], page) for page in pages}
        found = {(s["source"], s["page"]) for s in outputs.get("sources", [])}
        results.append({"key": "page_hit", "score": int(bool(expected & found))})

    value = reference_outputs.get("expected_value")
    if value:
        results.append({"key": "correct_value", "score": int(value in answer)})
    return {"results": results}


def run_experiment(name: str, limit: int | None = None):
    client = Client()
    try:
        upload_dataset(client)
    except LangSmithError as e:
        if "tracing keys" not in str(e):
            raise
        print("Your LangSmith key is a tracing key: it can send traces but can't create datasets. "
              "Create a Personal Access Token in LangSmith (Settings -> API Keys) and put it in .env.")
        return
    data = client.list_examples(dataset_name=DATASET_NAME, limit=limit) if limit else DATASET_NAME
    # One question at a time: the local Qdrant and free-tier LLM limits don't suit parallel runs.
    results = asyncio.run(aevaluate(target, data=data, evaluators=[score_answer], experiment_prefix=name,
                                    max_concurrency=1, client=client))
    print(f"\nExperiment saved: {results.experiment_name}")
    print("Open LangSmith -> Datasets & Experiments -> ml-rag-golden to see and compare it.")
