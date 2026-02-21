"""LangGraph state definition for coco.

CocoState is the single source of truth for the entire agent workflow.
The `messages` field uses the `add_messages` reducer — LangGraph merges
new messages in rather than replacing the list (standard LCEL/LangGraph pattern).
All other fields use last-write-wins semantics.
"""
from __future__ import annotations

from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class CocoState(TypedDict):
    # Accumulated conversation messages (add_messages reducer merges new ones in)
    messages: Annotated[list[BaseMessage], add_messages]

    # Routing: set by the router node before dispatching to a specialized agent
    task_type: str  # "code" | "plan" | "search" | "ask" | "auto"

    # Which agent is currently handling the request
    current_agent: str

    # Additional context injected from the indexer or tools
    context: str

    # The resolved absolute path of the project being worked on
    working_directory: str

    # Whether the codebase has been indexed via /index
    codebase_indexed: bool

    # Human-in-the-loop: set by an agent to pause and ask the user
    human_feedback_needed: bool
    clarification_question: str

    # Safety gate: pending destructive action awaiting user confirmation
    pending_confirmation: Optional[dict]  # {"action": str, "details": str}
    confirmation_granted: bool
