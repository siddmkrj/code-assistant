"""Rich terminal display for coco.

All output goes through this module so the visual style is consistent.
Import `console` directly for custom output; use the helper functions
for standard message types.
"""
from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
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


def print_file_diff(path: str, old: str | None, new: str) -> None:
    """Print a coloured unified diff for a file write.

    old=None means the file is being created for the first time.
    """
    import difflib  # noqa: PLC0415

    if old is None:
        console.print(f"[success]+ new file[/success] [bold]{path}[/bold]")
        lines = new.splitlines()
        for line in lines[:50]:
            console.print(f"  [green]+{line}[/green]")
        if len(lines) > 50:
            console.print(f"  [muted]... ({len(lines) - 50} more lines)[/muted]")
        return

    diff = list(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    ))

    if not diff:
        console.print(f"[muted]~ {path} (no changes)[/muted]")
        return

    console.print(f"[bold]{path}[/bold]")
    for line in diff[2:]:  # skip the ---/+++ header lines
        if line.startswith("@@"):
            console.print(f"  [cyan]{line}[/cyan]")
        elif line.startswith("+"):
            console.print(f"  [green]{line}[/green]")
        elif line.startswith("-"):
            console.print(f"  [red]{line}[/red]")
        else:
            console.print(f"  [muted]{line}[/muted]")


def print_git_diff(cwd: str | Path = ".") -> None:
    """Print the current git diff in color (unstaged changes)."""
    env = {**os.environ, "GIT_PAGER": ""}
    try:
        result = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--color=always"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except FileNotFoundError:
        return
    except subprocess.TimeoutExpired:
        return

    if result.returncode != 0 or not result.stdout.strip():
        return

    console.print()
    console.print("[muted]Changes (git diff):[/muted]")
    # Git's --color=always produces ANSI; Rich renders it
    console.print(result.stdout, markup=False)


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
