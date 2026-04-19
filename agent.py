from graph import compiled_graph
import asyncio
import argparse

async def main(query: str):
    initial_state = {
        "messages": [{"role": "user", "content": query}],
        "tool_results": [],
        "final_answer": None
    }

    result = await compiled_graph.ainvoke(initial_state, config={"recursion_limit": 20})

    if result["tool_results"]:
        print("Tools used:")
        for entry in result["tool_results"]:
            print(f"  • {entry['tool']}({list(entry['input'].values())[0][:60]})")
        print()

    final_message = next(
        m for m in reversed(result["messages"])
        if m.get("role") == "assistant"
    )
    for block in final_message["content"]:
        if block.get("type") == "text":
            print(block["text"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research agent")
    parser.add_argument("--query", type=str, required=True, help="Research question")
    args = parser.parse_args()
    asyncio.run(main(args.query))