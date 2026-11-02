"""Agent orchestration: the LLM decides which tools to call (documents, SQL,
MCP tools), loops until it has an answer. `create_agent` runs that loop for us.
This is what the chat page uses."""

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langsmith import traceable

from src.agent import cache
from src.agent.llm import get_llm
from src.agent.prompts import RESEARCH_PROMPT, SYSTEM_PROMPT
from src.agent.tools import agent_tools, load_mcp_tools

# Two agents, built once on first use:
#   normal   -> your PDFs + SQL database only (the default chat)
#   research -> the same + external MCP tools (Hugging Face), when Research mode is on
_agents: dict[str, object] = {}


async def get_agent(research: bool = False):
    mode = "research" if research else "normal"
    if mode not in _agents:
        tools = agent_tools() + (await load_mcp_tools() if research else [])
        _agents[mode] = create_agent(
            get_llm(), tools, system_prompt=RESEARCH_PROMPT if research else SYSTEM_PROMPT,
            # Hard limits per question (some models over-search, which only adds seconds):
            # extra calls are blocked and the agent answers with what it already has.
            middleware=[
                ToolCallLimitMiddleware(tool_name="search_documents", run_limit=2),
                ToolCallLimitMiddleware(run_limit=8),
            ],
        )
    return _agents[mode]


def collect_sources(messages) -> tuple[list[dict], list[dict], list[dict]]:
    """Go through the tool results and pull out what each tool used:
    PDF passages (search_documents), SQL queries + rows (query_database), and results
    from external MCP tools such as Hugging Face (everything else)."""
    own_tools = {"search_documents", "query_database", "get_database_schema", "list_pdf_documents"}
    passages, sql, external = [], [], []
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        if m.name == "search_documents" and m.artifact:
            passages += m.artifact
        elif m.name == "query_database" and m.artifact:
            sql.append(m.artifact)
        elif m.name not in own_tools:
            external.append({"tool": m.name, "text": m.text[:3000]})
    # the same passage can come back from several searches -- keep it once
    unique = {(p["source"], p["page"], p["text"]): p for p in passages}
    return list(unique.values()), sql, external


@traceable(name="chat_agent")
async def ask_agent(question: str, history: list[dict] | None = None, research: bool = False,
                    allow_cache: bool = True) -> dict:
    """history: earlier turns as [{"role": "user"|"assistant", "content": "..."}].
    research: also use external MCP tools (Hugging Face).
    allow_cache: False makes evaluation run the real agent on every question.
    Returns {"answer", "sources" (PDF passages), "sql" (queries + rows),
             "external" (MCP tool results), "tools_used", "cached"}."""
    # Cache only standalone questions in normal mode -- follow-ups depend on history, and
    # Research mode answers come from outside data that keeps changing.
    use_cache = allow_cache and not history and not research
    if use_cache and (cached := cache.lookup(question, cache.AGENT_CACHE)):
        return {**cached, "cached": True}

    agent = await get_agent(research)
    messages = (history or []) + [{"role": "user", "content": question}]
    try:
        output = await agent.ainvoke({"messages": messages})
    except Exception as e:
        # Some models occasionally write a garbled tool call and the provider rejects it. Try once more.
        if "tool_use_failed" not in str(e):
            raise
        output = await agent.ainvoke({"messages": messages})

    messages = output["messages"]
    sources, sql, external = collect_sources(messages)
    result = {
        "answer": messages[-1].text,
        "sources": sources,
        "sql": sql,
        "external": external,
        "tools_used": [{"tool": call["name"], "args": call["args"]}
                       for m in messages if isinstance(m, AIMessage) for call in m.tool_calls],
    }
    if use_cache:
        cache.save(question, result, cache.AGENT_CACHE)
    return {**result, "cached": False}
