"""Tests for LangGraph state and workflow components."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage


# ----------------------------------------------------------------- state

def test_coco_state_shape():
    """CocoState TypedDict has the expected keys."""
    from coco.graph.state import CocoState
    # TypedDict keys can be inspected via __annotations__
    keys = CocoState.__annotations__.keys()
    required = {
        "messages", "task_type", "current_agent", "context",
        "working_directory", "codebase_indexed",
        "human_feedback_needed", "clarification_question",
        "pending_confirmation", "confirmation_granted",
    }
    assert required.issubset(set(keys))


# ----------------------------------------------------------------- workflow routing

def _make_mock_agent(response: str = "I'm a mock agent."):
    """Return a mock agent whose run() returns a simple AI message."""
    agent = MagicMock()
    agent.run.return_value = {
        "messages": [AIMessage(content=response)],
        "human_feedback_needed": False,
        "clarification_question": "",
    }
    return agent


def test_route_decision_code():
    from coco.graph.workflow import CocoGraph
    graph = CocoGraph.__new__(CocoGraph)  # skip __init__
    state = {"task_type": "code", "messages": [], "human_feedback_needed": False}
    assert graph._route_decision(state) == "code"


def test_route_decision_unknown_defaults_to_ask():
    from coco.graph.workflow import CocoGraph
    graph = CocoGraph.__new__(CocoGraph)
    state = {"task_type": "gibberish", "messages": [], "human_feedback_needed": False}
    assert graph._route_decision(state) == "ask"


def test_check_human_feedback_needed():
    from coco.graph.workflow import CocoGraph
    graph = CocoGraph.__new__(CocoGraph)
    state_needs = {"human_feedback_needed": True, "messages": []}
    state_done = {"human_feedback_needed": False, "messages": []}
    assert graph._check_human_feedback(state_needs) == "needs_feedback"
    assert graph._check_human_feedback(state_done) == "done"


# ----------------------------------------------------------------- indexer

def test_codebase_indexer_iter_files(tmp_path):
    """_iter_files yields only files with included extensions.

    Skipped on Python 3.14+ where chromadb's pydantic v1 dependency is
    incompatible. The app targets Python 3.11-3.13.
    """
    import sys
    if sys.version_info >= (3, 14):
        pytest.skip("chromadb requires pydantic v1 which is incompatible with Python 3.14+")

    from coco.config.settings import IndexConfig
    from coco.indexer.codebase import CodebaseIndexer

    (tmp_path / "main.py").write_text("pass")
    (tmp_path / "README.md").write_text("# hi")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main")

    config = IndexConfig()
    indexer = CodebaseIndexer(
        config=config,
        embeddings=MagicMock(),
        working_dir=tmp_path,
    )
    found = list(indexer._iter_files())
    names = [f.name for f in found]
    assert "main.py" in names
    assert "README.md" in names
    assert "image.png" not in names  # .png not in include_extensions
    assert "HEAD" not in names  # .git excluded


def test_indexer_get_stats_not_indexed(tmp_path):
    """get_stats returns an informative message when the index is empty."""
    import sys
    if sys.version_info >= (3, 14):
        pytest.skip("chromadb requires pydantic v1 which is incompatible with Python 3.14+")

    from coco.config.settings import IndexConfig
    from coco.indexer.codebase import CodebaseIndexer

    config = IndexConfig(persist_dir=str(tmp_path / "idx"))
    indexer = CodebaseIndexer(
        config=config,
        embeddings=MagicMock(),
        working_dir=tmp_path,
    )
    stats = indexer.get_stats()
    assert "not found" in stats.lower() or "index" in stats.lower()
