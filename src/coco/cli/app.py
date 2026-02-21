"""Main REPL application for coco.

CocoApp owns:
- The prompt_toolkit PromptSession (input with history + autocomplete)
- The LangGraph graph (multi-agent workflow)
- The context compressor (conversation memory)
- The history logger

It drives the main loop: read input → dispatch command or route to graph →
display response. GraphInterrupt from human-in-the-loop nodes are caught
here and handled interactively.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from langchain_core.messages import AIMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from .commands import COMMANDS, HELP_TEXT, parse_command
from .display import (
    confirm_action,
    console,
    make_index_progress,
    print_error,
    print_info,
    print_muted,
    print_response,
    print_success,
    print_warning,
    print_welcome,
    spinner,
)
from ..config.settings import APP_DIR, CocoConfig, save_config
from ..graph.workflow import CocoGraph
from ..memory.compression import ContextCompressor
from ..memory.history import HistoryLogger

PROMPT_STYLE = Style.from_dict({
    "prompt": "bold ansicyan",
})


class CocoApp:
    """The main coco REPL application."""

    def __init__(
        self,
        config: CocoConfig,
        graph: CocoGraph,
        compressor: ContextCompressor,
        logger: HistoryLogger,
    ):
        self.config = config
        self.graph = graph
        self.compressor = compressor
        self.logger = logger

        self._thread_id = str(uuid.uuid4())
        self._codebase_indexed = False
        self._running = False
        self._last_user_input: str = ""  # passed to compressor in _display_result

        # prompt_toolkit session: persistent input history + autocomplete
        prompt_history_file = APP_DIR / "prompt_history"
        prompt_history_file.parent.mkdir(parents=True, exist_ok=True)

        self._session: PromptSession = PromptSession(
            history=FileHistory(str(prompt_history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(
                [f"/{cmd}" for cmd in COMMANDS.keys()],
                ignore_case=True,
                sentence=True,
            ),
            style=PROMPT_STYLE,
        )

    # ---------------------------------------------------------------- REPL

    def run(self) -> None:
        """Start the main REPL loop."""
        print_welcome()
        self.logger.log_system("session_start", {
            "working_directory": self.config.working_directory,
            "session_id": self.logger.session_id,
        })
        self._running = True

        while self._running:
            try:
                user_input = self._session.prompt("coco> ").strip()
            except KeyboardInterrupt:
                console.print("")
                print_muted("(Use /exit to quit)")
                continue
            except EOFError:
                self.cmd_exit("")
                break

            if not user_input:
                continue

            command, args = parse_command(user_input)
            if command is not None:
                self._dispatch_command(command, args)
            else:
                self._process_natural_input(user_input, task_type="auto")

    # -------------------------------------------------------- dispatch

    def _dispatch_command(self, command: str, args: str) -> None:
        handler_name = COMMANDS.get(command)
        if handler_name is None:
            print_error(f"Unknown command: /{command}. Type /help for help.")
            return
        handler = getattr(self, handler_name, None)
        if handler:
            handler(args)

    def _process_natural_input(
        self,
        user_input: str,
        task_type: str = "auto",
    ) -> None:
        """Send input to the LangGraph graph and display the response."""
        self.logger.log_user(user_input, task_type=task_type)
        self._last_user_input = user_input

        state_updates = {
            "task_type": task_type,
            "working_directory": str(Path(self.config.working_directory).resolve()),
            "codebase_indexed": self._codebase_indexed,
            "human_feedback_needed": False,
            "clarification_question": "",
            "pending_confirmation": None,
            "confirmation_granted": False,
            "context": self.compressor.get_summary(),
        }

        try:
            with spinner("Thinking..."):
                result = self.graph.invoke(user_input, self._thread_id, state_updates)
        except Exception as e:
            # Check for GraphInterrupt (human-in-the-loop pause)
            from langgraph.errors import GraphInterrupt  # noqa: PLC0415
            if isinstance(e, GraphInterrupt):
                self._handle_interrupt(e)
                return
            print_error(f"Agent error: {e}")
            self.logger.log_system("agent_error", {"error": str(e)})
            return

        self._display_result(result)

    def _handle_interrupt(self, exc: Exception) -> None:
        """Handle a GraphInterrupt by prompting the user for clarification."""
        # GraphInterrupt.args[0] is the value passed to interrupt()
        payload = exc.args[0] if exc.args else {}
        if isinstance(payload, dict):
            question = payload.get("question", "Please provide clarification:")
        else:
            question = str(payload)

        console.print()
        print_info("[bold]Agent needs clarification:[/bold]")
        console.print(f"  {question}\n")
        try:
            answer = self._session.prompt("Your answer> ").strip()
        except (KeyboardInterrupt, EOFError):
            print_muted("(Clarification skipped)")
            return

        if not answer:
            print_muted("(Empty answer, skipping clarification)")
            return

        try:
            with spinner("Continuing..."):
                result = self.graph.resume(answer, self._thread_id)
        except Exception as e:
            print_error(f"Error resuming: {e}")
            return

        self._display_result(result)

    def _display_result(self, result: dict) -> None:
        """Extract the last AI message from state and display it."""
        messages = result.get("messages", [])
        agent_name = result.get("current_agent", "coco")

        if not messages:
            print_muted("(no response)")
            return

        # Find the last AIMessage
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content

                # Handle multi-part content blocks (e.g. tool results interleaved)
                if isinstance(content, list):
                    text_parts = [
                        part["text"]
                        for part in content
                        if isinstance(part, dict) and "text" in part
                    ]
                    content = "\n\n".join(text_parts)

                if not content:
                    continue

                # Log and compress
                self.logger.log_assistant(content, agent=agent_name)
                self.compressor.add_interaction(
                    human_input=self._last_user_input,
                    ai_output=content,
                )

                print_response(content, agent_name)
                return

        print_muted("(agent returned no text response)")

    # ---------------------------------------------------- command handlers

    def cmd_help(self, args: str) -> None:
        from rich.markdown import Markdown  # noqa: PLC0415
        console.print(Markdown(HELP_TEXT))

    def cmd_exit(self, args: str) -> None:
        self.logger.log_system("session_end")
        print_info("Goodbye!")
        self._running = False

    def cmd_clear(self, args: str) -> None:
        self.compressor.clear()
        self._thread_id = str(uuid.uuid4())  # New thread = fresh graph checkpoint
        print_success("Conversation cleared. Starting fresh.")

    def cmd_status(self, args: str) -> None:
        from ..tools.index_tools import get_index_stats  # noqa: PLC0415
        index_info = get_index_stats.invoke({})
        lines = [
            f"  Working directory : {self.config.working_directory}",
            f"  Complex model     : {self.config.model.complex_model}",
            f"  Simple model      : {self.config.model.simple_model}",
            f"  Session ID        : {self.logger.session_id}",
            f"  Thread ID         : {self._thread_id[:8]}...",
            f"  History file      : {self.logger.log_path}",
            f"  Codebase index    : {index_info}",
            f"  Context summary   : {len(self.compressor.get_summary())} chars",
        ]
        console.print("\n[bold]coco status[/bold]")
        for line in lines:
            console.print(line)

    def cmd_index(self, args: str) -> None:
        """Index the codebase into Chroma."""
        from ..tools.index_tools import _indexer  # noqa: PLC0415
        if _indexer is None:
            print_error("Indexer not configured. This is a bug — please report it.")
            return

        print_info(f"Indexing {self.config.working_directory} ...")
        print_muted("Tip: Edit .coco to add/remove file extensions or exclude directories.")

        with make_index_progress() as progress:
            task = progress.add_task("Starting...", total=None)

            def callback(current: int, total: int, filename: str) -> None:
                short_name = Path(filename).name
                progress.update(
                    task,
                    completed=current,
                    total=total,
                    description=f"[cyan]{short_name}[/cyan]",
                )

            try:
                stats = _indexer.index(progress_callback=callback)
            except Exception as e:
                print_error(f"Indexing failed: {e}")
                return

        self._codebase_indexed = True
        print_success(
            f"Indexed {stats['files_processed']:,} files → "
            f"{stats['chunks_created']:,} chunks"
        )
        self.logger.log_system("index_complete", stats)

    def cmd_setup(self, args: str) -> None:
        """Interactive configuration wizard."""
        console.print("\n[bold]coco setup[/bold] [muted](press Enter to keep current value)[/muted]\n")

        complex_model = console.input(
            f"  Complex model [{self.config.model.complex_model}]: "
        ).strip() or self.config.model.complex_model

        simple_model = console.input(
            f"  Simple model  [{self.config.model.simple_model}]: "
        ).strip() or self.config.model.simple_model

        confirm_writes_str = console.input(
            f"  Confirm file writes [{self.config.safety.confirm_file_writes}]: "
        ).strip()
        if confirm_writes_str.lower() in ("true", "yes", "1", "y"):
            confirm_writes = True
        elif confirm_writes_str.lower() in ("false", "no", "0", "n"):
            confirm_writes = False
        else:
            confirm_writes = self.config.safety.confirm_file_writes

        self.config.model.complex_model = complex_model
        self.config.model.simple_model = simple_model
        self.config.safety.confirm_file_writes = confirm_writes

        try:
            path = save_config(self.config)
            print_success(f"Config saved to {path}")
        except Exception as e:
            print_error(f"Failed to save config: {e}")

    def cmd_ask(self, args: str) -> None:
        if not args:
            print_error("Usage: /ask <question>")
            return
        self._process_natural_input(args, task_type="ask")

    def cmd_code(self, args: str) -> None:
        if not args:
            print_error("Usage: /code <task description>")
            return
        # Safety check for file writes
        if self.config.safety.confirm_file_writes:
            if not confirm_action("This may write files. Proceed?"):
                print_muted("Cancelled.")
                return
        self._process_natural_input(args, task_type="code")

    def cmd_plan(self, args: str) -> None:
        if not args:
            print_error("Usage: /plan <task description>")
            return
        self._process_natural_input(args, task_type="plan")

    def cmd_search(self, args: str) -> None:
        if not args:
            print_error("Usage: /search <query>")
            return
        self._process_natural_input(args, task_type="search")

    def cmd_history(self, args: str) -> None:
        print_info(f"Session log: {self.logger.log_path}")
        print_muted("All sessions: ~/.coco/history/")
