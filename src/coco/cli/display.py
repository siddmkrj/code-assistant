"""Rich terminal display for coco.

All output goes through this module so the visual style is consistent.
Import `console` directly for custom output; use the helper functions
for standard message types.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.theme import Theme

COCO_THEME = Theme({
    "info":    "cyan",
    "success": "green",
    "warning": "yellow",
    "error":   "bold red",
    "agent":   "bold blue",
    "user":    "bold white",
    "command": "bold magenta",
    "muted":   "dim white",
})

console = Console(theme=COCO_THEME)


def print_welcome() -> None:
    console.print(Panel(
        "[bold cyan]coco[/bold cyan] [muted]— your CLI coding agent[/muted]\n\n"
        "Type [command]/help[/command] for commands.  "
        "Just type naturally — coco routes to the best agent automatically.",
        border_style="cyan",
        padding=(0, 1),
    ))


def print_response(content: str, agent_name: str = "") -> None:
    """Render an agent response as rich Markdown inside a labeled panel."""
    label = f"[agent]{agent_name}[/agent]" if agent_name else "[agent]coco[/agent]"
    console.print(Panel(
        Markdown(content),
        title=label,
        border_style="blue",
        padding=(0, 1),
    ))


def print_info(message: str) -> None:
    console.print(f"[info]>[/info] {message}")


def print_success(message: str) -> None:
    console.print(f"[success]✓[/success] {message}")


def print_warning(message: str) -> None:
    console.print(f"[warning]![/warning] {message}")


def print_error(message: str) -> None:
    console.print(f"[error]✗ Error:[/error] {message}")


def print_muted(message: str) -> None:
    console.print(f"[muted]{message}[/muted]")


def confirm_action(prompt: str) -> bool:
    """Prompt the user for yes/no confirmation. Returns True for yes."""
    response = console.input(f"[warning]?[/warning] {prompt} [muted][y/N][/muted] ")
    return response.strip().lower() in ("y", "yes")


@contextmanager
def spinner(message: str) -> Generator[None, None, None]:
    """Context manager showing a spinner during long-running operations."""
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[info]{message}[/info]"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("", total=None)
        yield


def make_index_progress() -> Progress:
    """Create a Progress instance for the /index command with file-level tracking."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
    )
