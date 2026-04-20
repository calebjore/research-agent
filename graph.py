from llm_client import chat
from dataclasses import dataclass, field
from typing import TypedDict, Annotated
import asyncio
import operator
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableConfig
from prompts import SYSTEM_PROMPT
from tools import dispatch_tool, TOOL_DEFINITIONS

tools = [*TOOL_DEFINITIONS]

# beef up the agent state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_results: Annotated[list, operator.add]
    final_answer: str | None

async def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    backend = config["configurable"].get("backend", "anthropic")
    model = config["configurable"].get("model", None)

    response = await chat(
        messages=state["messages"], system=SYSTEM_PROMPT, tools=tools, backend=backend, model=model
    )

    new_message = {
        "role": "assistant",
        "content": response["content"]
    }

    return {"messages": [new_message]}

async def tools_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_use_blocks = [
        b for b in last_message["content"]
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]

    async def run_tool(block) -> dict:
        result = await dispatch_tool(block["name"], block["input"])
        return {
            "type": "tool_result",
            "tool_use_id": block["id"],
            "content": result
        }

    results = await asyncio.gather(*[run_tool(b) for b in tool_use_blocks])

    log_entries = [
        {"tool": b["name"], "input": b["input"], "output": r["content"]}
        for b, r in zip(tool_use_blocks, results)
    ]

    return {
        "messages": [{"role": "user", "content": list(results)}],
        "tool_results": log_entries
    }

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    # Route to tools node, otherwise end
    if last_message.get("role") == "assistant":
        for block in last_message.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return "tools"
            
    return END

graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)
graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)

graph.add_edge("tools", "agent")

compiled_graph = graph.compile()