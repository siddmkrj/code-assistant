"""Tests for the RouterAgent classification logic."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage


def _make_router(return_value: str):
    """Build a RouterAgent with a mocked LLM that returns return_value."""
    from coco.agents.router import RouterAgent

    mock_llm = MagicMock()
    # create_react_agent is not used by RouterAgent (it uses a chain directly)
    # We need to mock the chain's invoke method
    router = RouterAgent(llm=mock_llm)

    # Patch the internal chain
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = return_value
    router._chain = mock_chain
    return router


def test_router_classifies_code():
    router = _make_router("code")
    result = router.classify([HumanMessage(content="Write a function to sort a list")])
    assert result == "code"


def test_router_classifies_plan():
    router = _make_router("plan")
    result = router.classify([HumanMessage(content="Design the architecture for my API")])
    assert result == "plan"


def test_router_classifies_search():
    router = _make_router("search")
    result = router.classify([HumanMessage(content="Find the latest FastAPI docs")])
    assert result == "search"


def test_router_classifies_ask():
    router = _make_router("ask")
    result = router.classify([HumanMessage(content="What does this function do?")])
    assert result == "ask"


def test_router_defaults_to_ask_on_unknown():
    router = _make_router("unknown_garbage_output")
    result = router.classify([HumanMessage(content="anything")])
    assert result == "ask"


def test_router_defaults_to_ask_on_empty():
    from coco.agents.router import RouterAgent
    mock_llm = MagicMock()
    router = RouterAgent(llm=mock_llm)
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("network error")
    router._chain = mock_chain
    result = router.classify([HumanMessage(content="test")])
    assert result == "ask"


def test_router_empty_messages():
    router = _make_router("code")
    result = router.classify([])
    assert result == "ask"
