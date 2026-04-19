import anthropic
from dotenv import load_dotenv
import os
import random
import time

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.AsyncAnthropic()
MAX_RETRIES = 3

async def chat(messages, system="You are a helpful assistant", model="claude-haiku-4-5", tool_choice={"type": "auto"}, max_tokens=1024, tools=[], stream=False):
    for attempt in range(MAX_RETRIES):
        try:
            if stream:
                with client.messages.stream(
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
                response = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    tools=tools,
                    tool_choice = tool_choice,
                    messages=messages
                )
                return response
        except anthropic.RateLimitError or anthropic.APIStatusError or anthropic.APIConnectionError:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"Error, retrying in {wait:.1f}s...")
            time.sleep(wait)