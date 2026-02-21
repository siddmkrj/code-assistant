"""Web search tools using DuckDuckGo (free, no API key required).

DuckDuckGo rate-limits aggressively, so errors are caught and returned
as informative strings so the agent can handle them gracefully.
"""
from __future__ import annotations

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return a summary of top results.

    Use this for finding documentation, tutorials, current events, or
    any information not available in the local codebase.

    Returns top search results as text with titles and snippets.
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchRun  # noqa: PLC0415
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        if not result:
            return f"No results found for: {query}"
        return result
    except Exception as e:
        error_msg = str(e).lower()
        if "rate" in error_msg or "429" in error_msg:
            return (
                "DuckDuckGo rate limit hit. Please wait a moment and try again. "
                f"Query was: {query}"
            )
        return (
            f"Web search failed: {e}\n"
            "Try rephrasing the query or check your internet connection."
        )
