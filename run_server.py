"""One entry point for everything.

  uv run python run_server.py ingest                      # every PDF in data/documents (strategy from config/chunking.json)
  uv run python run_server.py ingest --file "data/documents/x.pdf" --strategy semantic
  uv run python run_server.py ingest --sql "SELECT * FROM model_benchmarks" --name benchmarks
  uv run python run_server.py ingest --reset              # wipe the collection first
  uv run python run_server.py api                         # FastAPI on :8000
  uv run python run_server.py ui                          # Streamlit on :8501 (needs the api running)
  uv run python run_server.py eval                        # retrieval metrics (hit rate, MRR)
  uv run python run_server.py eval --llm                  # + LLM-as-judge (calls the LLM)
  uv run python run_server.py eval --chunking             # compare chunking strategies (slow)
  uv run python run_server.py eval --chunking --file "x.pdf" --pages 140-210 --only "fixed 600,agentic"
  uv run python run_server.py eval --langsmith --experiment groq-qwen   # saved, comparable run in LangSmith
  uv run python run_server.py mcp                         # our MCP server over stdio

Admin checks (see what the RAG pipeline does with a sample query):
  uv run python run_server.py inspect "What does StandardScaler do?"
  uv run python run_server.py stats                       # chunks per PDF, sizes, strategy
  uv run python run_server.py chunks --file "x.pdf" --page 147
"""

import argparse
import subprocess
import sys
from pathlib import Path

from config import settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["ingest", "api", "ui", "eval", "mcp", "inspect", "stats", "chunks"])
    parser.add_argument("question", nargs="?", help="inspect: the sample question")
    parser.add_argument("--file", help="ingest: only this PDF")
    parser.add_argument("--strategy", help="ingest: chunking strategy (default: config/chunking.json)")
    parser.add_argument("--sql", help="ingest: a SELECT whose rows become searchable chunks")
    parser.add_argument("--name", default="table", help="ingest: name for the --sql source")
    parser.add_argument("--reset", action="store_true", help="ingest: delete all indexed chunks first")
    parser.add_argument("--llm", action="store_true", help="eval: also run LLM-as-judge")
    parser.add_argument("--chunking", action="store_true", help="eval: compare chunking strategies")
    parser.add_argument("--pages", help="eval --chunking: page range like 140-210")
    parser.add_argument("--only", help='eval --chunking: comma-separated strategies, e.g. "fixed 600,semantic"')
    parser.add_argument("--page", type=int, help="chunks: page number")
    parser.add_argument("--langsmith", action="store_true", help="eval: run the agent as a LangSmith experiment")
    parser.add_argument("--experiment", default="experiment", help="eval --langsmith: name for this run")
    parser.add_argument("--limit", type=int, help="eval --langsmith: only use the first N questions")
    args = parser.parse_args()

    if args.command == "ingest":
        from src.database.schema import init_db
        from src.ingestion import pipeline
        from src.retrieval.vector_store import reset_collection

        init_db()
        if args.reset:
            reset_collection()
        if args.sql:
            pipeline.ingest_sql(args.sql, args.name)
        elif args.file:
            pipeline.ingest_pdf(args.file, args.strategy)
        else:
            pipeline.ingest_folder(strategy=args.strategy)

    elif args.command == "api":
        import uvicorn

        uvicorn.run("src.api.main:app", host=settings.api_host, port=settings.api_port)

    elif args.command == "ui":
        subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/app.py"])

    elif args.command == "eval":
        from src.evaluation import chunking_eval, llm_eval, retrieval_eval
        from src.evaluation.results import save_results

        if args.langsmith:
            from src.evaluation.langsmith_eval import run_experiment

            run_experiment(args.experiment, args.limit)
        elif args.chunking:
            kwargs = {}
            if args.file:
                kwargs["file_name"] = Path(args.file).name
            if args.pages:
                kwargs["pages"] = tuple(int(x) for x in args.pages.split("-"))
            if args.only:
                kwargs["candidates"] = [s.strip() for s in args.only.split(",")]
            rows = chunking_eval.compare_chunking(**kwargs)
            save_results("chunking", rows)
            chunking_eval.print_report(rows)
        else:
            rows = retrieval_eval.compare_strategies()
            save_results("retrieval", rows)
            retrieval_eval.print_report(rows)
        if args.llm:
            rows = llm_eval.run_llm_eval()
            save_results("llm", rows)
            llm_eval.print_report(rows)

    elif args.command == "inspect":
        from src.evaluation import inspector

        inspector.print_inspection(inspector.inspect_query(args.question or "What does StandardScaler do?"))

    elif args.command == "stats":
        from src.evaluation import inspector

        for row in inspector.document_stats():
            print(row)

    elif args.command == "chunks":
        from src.evaluation import inspector

        for i, text in enumerate(inspector.page_chunks(Path(args.file).name, args.page), 1):
            print(f"--- chunk {i} ({len(text)} chars)\n{text}\n")

    elif args.command == "mcp":
        from src.database.schema import init_db
        from src.mcp_server.server import mcp

        init_db()
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
