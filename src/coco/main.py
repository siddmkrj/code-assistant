"""coco entry point.

Called as `coco` via the console script registered in pyproject.toml:
    [project.scripts]
    coco = "coco.main:main"

Startup sequence:
1. Load .env
2. Validate ANTHROPIC_API_KEY
3. Set up SQLiteCache globally (before any LLM is created)
4. Load config from .coco (or defaults)
5. Build LLMs with retry
6. Build local embeddings
7. Build CodebaseIndexer
8. Register tools (including indexer registry)
9. Build LangGraph workflow
10. Build memory + logger
11. Launch REPL
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _setup_cache() -> None:
    """Enable SQLite-backed LLM response caching.

    Must be called before any LLM instance is created.
    Cache lives at ~/.coco/cache/llm_cache.db.
    """
    from langchain_core.globals import set_llm_cache  # noqa: PLC0415
    from langchain_community.cache import SQLiteCache  # noqa: PLC0415
    from .config.settings import APP_DIR  # noqa: PLC0415

    cache_dir = APP_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(cache_dir / "llm_cache.db")))


def _build_llms(config):
    """Build ChatAnthropic instances with built-in retry support.

    We use ChatAnthropic's native max_retries parameter rather than
    .with_retry(), because .with_retry() returns a RunnableRetry which
    loses the bind_tools() method required by create_react_agent.
    """
    from langchain_anthropic import ChatAnthropic  # noqa: PLC0415

    common_kwargs = dict(
        temperature=config.model.temperature,
        max_tokens=config.model.max_tokens,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        max_retries=3,  # Built-in retry with exponential backoff
    )

    simple_llm = ChatAnthropic(model=config.model.simple_model, **common_kwargs)
    complex_llm = ChatAnthropic(model=config.model.complex_model, **common_kwargs)

    return simple_llm, complex_llm


def _build_embeddings():
    """Build a local HuggingFace sentence-transformer embedding model.

    Falls back to FakeEmbeddings if sentence-transformers is not installed,
    which allows the app to start without a working indexer (degraded mode).
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: PLC0415
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError:
        from langchain_core.embeddings import FakeEmbeddings  # noqa: PLC0415
        return FakeEmbeddings(size=384)


def _build_tools(config, indexer) -> dict[str, list]:
    """Assemble per-agent tool lists and register the indexer."""
    from .tools.file_tools import (  # noqa: PLC0415
        list_directory, read_file, search_in_files, write_file,
    )
    from .tools.search_tools import web_search  # noqa: PLC0415
    from .tools.code_tools import get_git_diff, get_git_log, run_python_snippet  # noqa: PLC0415
    from .tools.index_tools import (  # noqa: PLC0415
        get_index_stats, search_codebase, set_indexer,
    )

    # Register the indexer so search_codebase and get_index_stats work
    set_indexer(indexer)

    return {
        "code": [
            read_file, write_file, list_directory, search_in_files,
            search_codebase, get_index_stats, get_git_diff, get_git_log,
            run_python_snippet,
        ],
        "plan": [
            read_file, list_directory, search_codebase, get_index_stats,
        ],
        "search": [
            web_search, search_codebase, get_index_stats,
        ],
        "ask": [
            read_file, list_directory, search_codebase, get_index_stats,
        ],
    }


def main() -> None:
    """Main entry point for the `coco` CLI command."""
    from dotenv import load_dotenv  # noqa: PLC0415
    load_dotenv()

    # Validate required env var
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Error: ANTHROPIC_API_KEY is not set.\n"
            "Add it to a .env file in this directory, or export it:\n"
            "  export ANTHROPIC_API_KEY=your_key_here",
            file=sys.stderr,
        )
        sys.exit(1)

    # Must run before any LLM is instantiated
    _setup_cache()

    # Load project config
    from .config.settings import load_config  # noqa: PLC0415
    config = load_config()

    # Build LLMs
    simple_llm, complex_llm = _build_llms(config)

    # Build embeddings + indexer
    embeddings = _build_embeddings()
    from .indexer.codebase import CodebaseIndexer  # noqa: PLC0415
    indexer = CodebaseIndexer(
        config=config.index,
        embeddings=embeddings,
        working_dir=Path(config.working_directory).resolve(),
    )

    # Wire up tools (also registers the indexer)
    tools = _build_tools(config, indexer)

    # Build LangGraph workflow
    from .graph.workflow import build_graph  # noqa: PLC0415
    graph = build_graph(simple_llm, complex_llm, tools)

    # Build memory + logger
    from .memory.compression import ContextCompressor  # noqa: PLC0415
    from .memory.history import HistoryLogger  # noqa: PLC0415
    compressor = ContextCompressor(llm=simple_llm, max_token_limit=config.memory.max_token_limit)
    logger = HistoryLogger(history_dir=config.memory.history_dir)

    # Launch the REPL
    from .cli.app import CocoApp  # noqa: PLC0415
    app = CocoApp(config=config, graph=graph, compressor=compressor, logger=logger)
    app.run()


if __name__ == "__main__":
    main()
