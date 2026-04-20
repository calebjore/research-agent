import anthropic
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import random
import time
import json

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
MAX_RETRIES = 3

anthropic_client = anthropic.AsyncAnthropic()
ollama_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

async def chat(
    messages: list[dict],
    system: str | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
    tools: list[dict] | None = None,
    backend: str = "anthropic"
) -> object:
    if backend == "anthropic":
        return await _chat_anthropic(messages=messages, system=system, model=model, max_tokens=max_tokens, tools=tools)
    elif backend == "ollama":
        return await _chat_ollama(messages=messages, system=system, model=model, max_tokens=max_tokens, tools=tools)
    else:
        raise ValueError(f"Unknown backend: {backend}")

async def _chat_anthropic(messages, system="You are a helpful assistant", model="claude-haiku-4-5", tool_choice={"type": "auto"}, max_tokens=1024, tools=[], stream=False):
    for attempt in range(MAX_RETRIES):
        try:
            if stream:
                with anthropic_client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    tools=tools,
                    tool_choice = tool_choice,
                    messages=messages
                ) as stream:
                    for text in stream.text_stream:
                        print(text, end="", flush=True)
                print()
                return None
            else:
                response = await anthropic_client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    tools=tools,
                    tool_choice = tool_choice,
                    messages=messages
                )
                return { # normalize to dict to match Ollama backend
                    "content": [block.model_dump() for block in response.content],
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens
                    }
                }
        except anthropic.RateLimitError or anthropic.APIStatusError or anthropic.APIConnectionError:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"Error, retrying in {wait:.1f}s...")
            time.sleep(wait)

def _to_openai_messages(messages: list[dict], system: str | None) -> list[dict]:
    result = []
    if system:
        result.append({"role": "system", "content": system})
    
    for msg in messages:
        content = msg["content"]
        
        # Plain string content — pass through
        if isinstance(content, str):
            result.append({"role": msg["role"], "content": content})
            continue

        # List of blocks — inspect types
        tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]
        tool_result_blocks = [b for b in content if b.get("type") == "tool_result"]
        text_blocks = [b for b in content if b.get("type") == "text"]

        if tool_use_blocks:
            # Assistant is making tool calls
            result.append({
                "role": "assistant",
                "content": text_blocks[0]["text"] if text_blocks else None,
                "tool_calls": [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {
                            "name": b["name"],
                            "arguments": json.dumps(b["input"])
                        }
                    }
                    for b in tool_use_blocks
                ]
            })
        elif tool_result_blocks:
            # Tool results — one message per result in OpenAI format
            for b in tool_result_blocks:
                result.append({
                    "role": "tool",
                    "tool_call_id": b["tool_use_id"],
                    "content": b["content"]
                })
        elif text_blocks:
            # Plain assistant text response
            result.append({
                "role": msg["role"],
                "content": text_blocks[0]["text"]
            })

    return result

async def _chat_ollama(messages, system="You are a helpful assistant", model="gemma4:e4b", tool_choice={"type": "auto"}, max_tokens=1024, tools=[], stream=False):
    openai_messages = []
    if system:
        openai_messages.append({"role": "system", "content": system})
    openai_messages.extend(messages)

    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"]
            }
        }
        for t in tools
    ] if tools else []

    openai_tool_choice = "auto" if tools else "none"
    openai_messages = _to_openai_messages(messages, system)

    print(openai_messages)
    response = await ollama_client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=openai_messages,
        tools=openai_tools,
        tool_choice=openai_tool_choice
    )

    message = response.choices[0].message
    finish_reason = response.choices[0].finish_reason

    # Normalize content blocks
    content = []
    if message.content:
        content.append({"type": "text", "text": message.content})
    if message.tool_calls:
        for tc in message.tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": json.loads(tc.function.arguments)
            })

    # Normalize stop_reason
    stop_reason = "tool_use" if finish_reason == "tool_calls" else "end_turn"

    return {
        "content": content,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens
        }
    }