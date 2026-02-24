"""Base agent class for all coco specialized agents.

All agents are thin wrappers around langgraph.prebuilt.create_react_agent.
They share a common pattern:
1. A system prompt that defines the agent's role and capabilities
2. A curated tool list appropriate for the task
3. A [CLARIFY]...[/CLARIFY] convention for requesting human input

The [CLARIFY] protocol:
    When an agent needs user clarification, it includes the tag in its response:
        [CLARIFY]What framework should I use for the API?[/CLARIFY]
    BaseCocoAgent.run() detects this and sets human_feedback_needed=True,
    which triggers the LangGraph human_feedback_node → interrupt() flow.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent


def _format_args(args: dict) -> str:
    """Return a compact one-line summary of tool call arguments."""
    if not args:
        return ""
    # Show the most informative single argument rather than dumping everything
    for key in ("path", "directory", "query", "pattern", "url", "command"):
        if key in args:
            return str(args[key])
    first_val = str(next(iter(args.values())))
    return first_val[:60] + ("..." if len(first_val) > 60 else "")


class BaseCocoAgent(ABC):
    """Base class for all coco agents.

    Subclasses must define:
        name: str — unique agent identifier
        system_prompt: str — the agent's system instructions
    """

    name: str = "base"
    system_prompt: str = ""

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool]):
        self._llm = llm
        self._tools = tools
        self._agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=self.system_prompt,
        )

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run the agent against the current conversation state.

        Returns a dict with:
            messages: list[BaseMessage] — new messages to add to state
            human_feedback_needed: bool — True if agent needs user input
            clarification_question: str — the question (if needed)
        """
        from ..cli.display import console  # noqa: PLC0415

        all_messages: list[BaseMessage] = []

        try:
            for chunk in self._agent.stream({"messages": state["messages"]}):
                if "agent" in chunk:
                    for msg in chunk["agent"].get("messages", []):
                        all_messages.append(msg)
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                detail = _format_args(tc.get("args", {}))
                                console.print(
                                    f"  [cyan]→[/cyan] [bold]{tc['name']}[/bold]"
                                    + (f"  [muted]{detail}[/muted]" if detail else "")
                                )
                elif "tools" in chunk:
                    for msg in chunk["tools"].get("messages", []):
                        all_messages.append(msg)
                        name = getattr(msg, "name", "tool")
                        preview = str(msg.content)[:80].replace("\n", " ")
                        console.print(f"  [muted]← {name}: {preview}[/muted]")
        except Exception as e:
            error_msg = AIMessage(
                content=f"I encountered an error: {e}\n\nPlease try rephrasing your request."
            )
            return {
                "messages": [error_msg],
                "human_feedback_needed": False,
                "clarification_question": "",
            }

        human_feedback_needed = False
        clarification_question = ""

        # Check if the last AI message contains a clarification request
        for msg in reversed(all_messages):
            if isinstance(msg, AIMessage):
                content = msg.content if isinstance(msg.content, str) else ""
                if "[CLARIFY]" in content:
                    human_feedback_needed = True
                    clarification_question = self._extract_clarification(content)
                break

        return {
            "messages": all_messages,
            "human_feedback_needed": human_feedback_needed,
            "clarification_question": clarification_question,
        }

    @staticmethod
    def _extract_clarification(content: str) -> str:
        """Extract question from [CLARIFY]...[/CLARIFY] tags."""
        match = re.search(r"\[CLARIFY\](.*?)\[/CLARIFY\]", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: return content after [CLARIFY] if closing tag is missing
        match = re.search(r"\[CLARIFY\](.*)", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content

    @abstractmethod
    def get_tool_names(self) -> list[str]:
        """Return the names of tools this agent has access to."""
        ...
