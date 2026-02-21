"""Search agent — handles web searches and documentation lookups.

Uses the simple (Haiku) model with web_search and codebase search tools.
Always cites sources in its responses.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .base import BaseCocoAgent

SEARCH_SYSTEM_PROMPT = """You are a research assistant helping developers find information.

YOUR TOOLS:
- web_search: Search the web using DuckDuckGo for documentation, tutorials, and current info
- search_codebase: Search the indexed local codebase for relevant code
- get_index_stats: Check if the codebase is indexed

GUIDELINES:
1. For questions about the local codebase, use search_codebase first
2. For documentation, best practices, or external libraries, use web_search
3. Always include source URLs in your response when using web_search
4. Prefer official documentation over blog posts when available
5. Summarize findings concisely — focus on what's actionable for the developer
6. If DuckDuckGo rate-limits, apologize and suggest the user retry in a moment

FORMAT:
- Lead with the direct answer
- Follow with relevant details or examples
- End with sources (URLs) when from web search"""


class SearchAgent(BaseCocoAgent):
    name = "search_agent"
    system_prompt = SEARCH_SYSTEM_PROMPT

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool]):
        super().__init__(llm, tools)

    def get_tool_names(self) -> list[str]:
        return [t.name for t in self._tools]
