"""LangGraph StateGraph workflow for coco.

Graph topology:
    START
      └─> router_node (RouterAgent / Haiku — classifies task_type)
            ├─ "code"   → code_node   (CodeAgent / Sonnet)
            ├─ "plan"   → plan_node   (PlanAgent / Sonnet)
            ├─ "search" → search_node (SearchAgent / Haiku)
            └─ "ask"    → ask_node    (AskAgent / Haiku)
                  │
                  ├─ human_feedback_needed? → human_feedback_node
                  │                              └─> interrupt(question)
                  │                              ← CLI catches GraphInterrupt
                  │                              → user answers
                  │                              → Command(resume=answer) → END
                  └─ done → END

The graph uses MemorySaver as its checkpointer, enabling the interrupt/resume
flow within a single session. Each REPL turn is a separate thread_id to keep
state isolated between conversations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .state import CocoState

if TYPE_CHECKING:
    from ..agents.ask_agent import AskAgent
    from ..agents.code_agent import CodeAgent
    from ..agents.plan_agent import PlanAgent
    from ..agents.router import RouterAgent
    from ..agents.search_agent import SearchAgent


class CocoGraph:
    """Compiled LangGraph multi-agent graph.

    Owns the MemorySaver checkpointer and exposes invoke() / resume()
    for the CLI to drive.
    """

    def __init__(
        self,
        router_agent: "RouterAgent",
        code_agent: "CodeAgent",
        plan_agent: "PlanAgent",
        search_agent: "SearchAgent",
        ask_agent: "AskAgent",
    ):
        self._router = router_agent
        self._code = code_agent
        self._plan = plan_agent
        self._search = search_agent
        self._ask = ask_agent
        self._checkpointer = MemorySaver()
        self._graph = self._build()

    # ------------------------------------------------------------------ build

    def _build(self):
        builder = StateGraph(CocoState)

        builder.add_node("router", self._router_node)
        builder.add_node("code_agent", self._code_node)
        builder.add_node("plan_agent", self._plan_node)
        builder.add_node("search_agent", self._search_node)
        builder.add_node("ask_agent", self._ask_node)
        builder.add_node("human_feedback", self._human_feedback_node)

        builder.add_edge(START, "router")

        builder.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "code": "code_agent",
                "plan": "plan_agent",
                "search": "search_agent",
                "ask": "ask_agent",
            },
        )

        for agent_node in ("code_agent", "plan_agent", "search_agent", "ask_agent"):
            builder.add_conditional_edges(
                agent_node,
                self._check_human_feedback,
                {"needs_feedback": "human_feedback", "done": END},
            )

        builder.add_edge("human_feedback", END)

        return builder.compile(checkpointer=self._checkpointer)

    # --------------------------------------------------------------- nodes

    def _router_node(self, state: CocoState) -> dict:
        from ..cli.display import console  # noqa: PLC0415
        task_type = self._router.classify(state["messages"])
        # Respect explicit task_type passed in (e.g. /code, /plan commands)
        if state.get("task_type") not in ("auto", "", None):
            task_type = state["task_type"]
        console.print(f"[muted]routing → {task_type}[/muted]")
        return {"task_type": task_type, "current_agent": "router"}

    def _code_node(self, state: CocoState) -> dict:
        return self._run_agent(self._code, state, "code_agent")

    def _plan_node(self, state: CocoState) -> dict:
        return self._run_agent(self._plan, state, "plan_agent")

    def _search_node(self, state: CocoState) -> dict:
        return self._run_agent(self._search, state, "search_agent")

    def _ask_node(self, state: CocoState) -> dict:
        return self._run_agent(self._ask, state, "ask_agent")

    def _run_agent(self, agent, state: CocoState, agent_name: str) -> dict:
        """Run a specialized agent and normalize its output into state fields."""
        from ..cli.display import console  # noqa: PLC0415
        console.print(f"[bold blue]⟳[/bold blue] [bold]{agent_name}[/bold]")
        result = agent.run(state)
        return {
            "messages": result.get("messages", []),
            "current_agent": agent_name,
            "human_feedback_needed": result.get("human_feedback_needed", False),
            "clarification_question": result.get("clarification_question", ""),
        }

    def _human_feedback_node(self, state: CocoState) -> dict:
        """Pause execution and surface a clarification question to the CLI.

        Uses LangGraph's interrupt() which raises GraphInterrupt, pausing the
        graph. The CLI catches it, prompts the user, then calls graph.resume(answer)
        which feeds the answer back via Command(resume=answer).
        """
        question = state.get("clarification_question") or "Please provide clarification."
        # interrupt() raises GraphInterrupt — execution resumes here after resume()
        user_answer = interrupt({"question": question})
        return {
            "messages": [HumanMessage(content=str(user_answer))],
            "human_feedback_needed": False,
            "clarification_question": "",
        }

    # --------------------------------------------------------- routing fns

    def _route_decision(
        self, state: CocoState
    ) -> Literal["code", "plan", "search", "ask"]:
        task_type = state.get("task_type", "ask")
        if task_type in ("code", "plan", "search"):
            return task_type
        return "ask"

    def _check_human_feedback(
        self, state: CocoState
    ) -> Literal["needs_feedback", "done"]:
        if state.get("human_feedback_needed"):
            return "needs_feedback"
        return "done"

    # --------------------------------------------------------- public API

    def invoke(
        self,
        user_input: str,
        thread_id: str,
        state_updates: dict | None = None,
    ) -> dict:
        """Run the graph for one turn and return the final state.

        Args:
            user_input: The user's message
            thread_id: Unique identifier for this conversation thread
            state_updates: Optional overrides for state fields (e.g. task_type)
        """
        config = {"configurable": {"thread_id": thread_id}}
        initial = {
            "messages": [HumanMessage(content=user_input)],
            "task_type": "auto",
            "current_agent": "",
            "context": "",
            "human_feedback_needed": False,
            "clarification_question": "",
            "pending_confirmation": None,
            "confirmation_granted": False,
            **(state_updates or {}),
        }
        return self._graph.invoke(initial, config)

    def resume(self, answer: str, thread_id: str) -> dict:
        """Resume a graph paused at a human_feedback_node interrupt.

        Args:
            answer: The user's answer to the clarification question
            thread_id: The thread_id of the paused graph
        """
        config = {"configurable": {"thread_id": thread_id}}
        return self._graph.invoke(Command(resume=answer), config)

    def get_state(self, thread_id: str) -> dict:
        """Return the current state snapshot for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        return self._graph.get_state(config)


def build_graph(
    simple_llm: BaseChatModel,
    complex_llm: BaseChatModel,
    tools: dict[str, list[BaseTool]],
) -> CocoGraph:
    """Factory function: wire up all agents and return a compiled CocoGraph.

    Args:
        simple_llm: Fast/cheap LLM (Haiku) for routing, ask, search agents
        complex_llm: Powerful LLM (Sonnet) for code and plan agents
        tools: Dict mapping agent name to its tool list
               Keys: "code", "plan", "search", "ask"
    """
    from ..agents.ask_agent import AskAgent
    from ..agents.code_agent import CodeAgent
    from ..agents.plan_agent import PlanAgent
    from ..agents.router import RouterAgent
    from ..agents.search_agent import SearchAgent

    return CocoGraph(
        router_agent=RouterAgent(llm=simple_llm),
        code_agent=CodeAgent(llm=complex_llm, tools=tools.get("code", [])),
        plan_agent=PlanAgent(llm=complex_llm, tools=tools.get("plan", [])),
        search_agent=SearchAgent(llm=simple_llm, tools=tools.get("search", [])),
        ask_agent=AskAgent(llm=simple_llm, tools=tools.get("ask", [])),
    )
