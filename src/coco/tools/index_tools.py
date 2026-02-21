"""Codebase index search tools.

LangChain @tool functions must be top-level callables with no constructor
arguments. We use a module-level registry pattern to inject the indexer
without violating this constraint.

Call set_indexer(indexer) once during app startup (in main.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from ..indexer.codebase import CodebaseIndexer

# Module-level indexer registry — set once at startup via set_indexer()
_indexer: "CodebaseIndexer | None" = None


def set_indexer(indexer: "CodebaseIndexer") -> None:
    """Register the indexer instance. Called once during app initialization."""
    global _indexer
    _indexer = indexer


@tool
def search_codebase(query: str, n_results: int = 5) -> str:
    """Search the indexed codebase for code and documentation relevant to a query.

    Returns the most semantically similar code snippets with their file paths.
    Requires the codebase to be indexed first via the /index command.

    Args:
        query: Natural language or code query
        n_results: Number of results to return (default: 5)
    """
    if _indexer is None:
        return "Codebase not indexed. Ask the user to run /index first."
    return _indexer.search(query, n_results=n_results)


@tool
def get_index_stats() -> str:
    """Return statistics about the current codebase index.

    Shows the number of indexed chunks, collection name, and storage path.
    """
    if _indexer is None:
        return "No indexer configured. Codebase has not been indexed."
    return _indexer.get_stats()
