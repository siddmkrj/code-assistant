"""Rich terminal display for coco.

All output goes through this module so the visual style is consistent.
Import `console` directly for custom output; use the helper functions
for standard message types.
"""
from __future__ import annotations

import difflib
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
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

# Accumulates file changes during an agent run; flushed by print_git_diff.
_pending_changes: list[dict] = []


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


def print_file_diff(path: str, old: str | None, new: str) -> None:
    """Record a file write so it can be shown in the post-run summary."""
    if old is None:
        added = len(new.splitlines())
        removed = 0
        kind = "new"
    else:
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        delta = list(difflib.ndiff(old_lines, new_lines))
        added = sum(1 for d in delta if d.startswith("+ "))
        removed = sum(1 for d in delta if d.startswith("- "))
        kind = "updated" if (added or removed) else "unchanged"

    if kind != "unchanged":
        _pending_changes.append({"path": path, "kind": kind, "added": added, "removed": removed})


def print_git_diff(cwd: str | Path = ".") -> None:
    """Render the accumulated file-change summary and reset the pending list."""
    global _pending_changes

    if not _pending_changes:
        return

    console.print()
    console.print("[muted]Files changed:[/muted]")

    for change in _pending_changes:
        path = change["path"]
        kind = change["kind"]
        added = change["added"]
        removed = change["removed"]

        if kind == "new":
            line = Text("  ● ", style="green")
        elif kind == "deleted":
            line = Text("  ● ", style="red")
        else:
            line = Text("  ● ", style="yellow")

        line.append(path)
        line.append("  ")

        if kind == "new":
            line.append(f"+{added}", style="green")
        elif kind == "deleted":
            line.append(f"-{removed}", style="red")
        else:
            if added:
                line.append(f"+{added}", style="green")
            if added and removed:
                line.append(" / ", style="dim white")
            if removed:
                line.append(f"-{removed}", style="red")

        console.print(line)

    _pending_changes = []


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as '1m 29s' or '45s'."""
    if seconds < 60:
        return f"{int(seconds)}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if secs:
        return f"{mins}m {secs}s"
    return f"{mins}m"


def _format_token_count(count: int) -> str:
    """Format token count as '2.2k' or plain number."""
    if count >= 1000:
        return f"{count / 1000:.1f}k"
    return str(count)


def print_token_usage(
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    cache_creation: int,
    elapsed_seconds: float | None = None,
) -> None:
    """Print Claude Code-style token usage: ✳ (1m 29s · ↓ 2.2k · ↑ 1.5k)"""
    parts: list[str] = []
    if elapsed_seconds is not None and elapsed_seconds >= 0:
        parts.append(format_duration(elapsed_seconds))
    parts.append(f"↓ {_format_token_count(output_tokens)}")
    parts.append(f"↑ {_format_token_count(input_tokens)}")
    if cache_read:
        parts.append(f"cache-read {_format_token_count(cache_read)}")
    if cache_creation:
        parts.append(f"cache-write {_format_token_count(cache_creation)}")
    console.print(f"[muted]✳ ({' · '.join(parts)})[/muted]")


def confirm_action(prompt: str) -> bool:
    """Prompt the user for yes/no confirmation. Returns True for yes."""
    response = console.input(f"[warning]?[/warning] {prompt} [muted][y/N][/muted] ")
    return response.strip().lower() in ("y", "yes")


@contextmanager
def spinner(message: str = "Thinking…") -> Generator[None, None, None]:
    """Context manager showing Claude Code-style spinner with elapsed time."""
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[info]✳[/info] [info]{task.description}[/info]"),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task(message, total=None)
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
