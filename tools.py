import datetime
import wikipediaapi
import httpx
from html.parser import HTMLParser
from llm_client import chat
from prompts import SYSTEM_PROMPT

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="ResearchAgent/1.0"
)

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.text_parts.append(data.strip())

# JSON defs

TOOL_DEFINITIONS = [
    {
        "name": "wikipedia_search",
        "description": "Use this when you need to find information about something you're unsure about or when the user asks you to search the web",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic to search for"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_url",
        "description": "Fetch and extract the text content of a URL. Use for specific articles, papers, or web pages when you have an actual URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL including https://"}
            },
            "required": ["url"]
        }
    },
    {
        "name": "summarize",
        "description": "Summarize a long piece of text. Use when you have retrieved content that is too long to reason about directly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to summarize"},
                "focus": {"type": "string", "description": "Optional: what aspect to focus on"}
            },
            "required": ["text"]
        }
    }
]

async def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "wikipedia_search":
        return wikipedia_search(inputs["query"])
    elif name == "fetch_url":
        return await fetch_url(inputs["url"])
    elif name == "summarize":
        return await summarize(inputs["text"], inputs.get("focus", ""))
    return f"Unknown tool: {name}"

# tool defs
def wikipedia_search(query: str) -> str:
    page = wiki.page(query)
    if not page.exists():
        return f"No Wikipedia page found for {query}" 
    return f"Title: {page.title}\n\nSummary: {page.summary}\n\n{page.text[:1500]}"

async def fetch_url(url: str) -> str:
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
        except Exception as e:
            return f"Error fetching {url}: {e}"

    parser = TextExtractor()
    parser.feed(response["text"])
    text = " ".join(parser.text_parts)
    return text[:3000]

async def summarize(text: str, focus: str = "") -> str:
    focus_instruction = f" Focus on {focus}." if focus else ""
    prompt = f"Summarize the following text concisely.{focus_instruction}\n\n{text}"
    
    response = chat(
        system=SYSTEM_PROMPT,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )