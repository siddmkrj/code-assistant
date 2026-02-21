"""Code execution and git tools for coco agents.

Safety notes:
- run_python_snippet uses exec() which is intentional for a local developer tool.
  The user is assumed to own the codebase and be aware they're running an AI agent.
- Subprocess tools use list arguments (not shell=True) to prevent injection.
- All subprocess calls have timeouts to prevent hanging.
"""
from __future__ import annotations

import subprocess
from langchain_core.tools import tool


@tool
def run_python_snippet(code: str) -> str:
    """Execute a small Python snippet and return its output.

    Use for quick calculations, data transformations, or validation.
    This runs in the current Python environment.

    Returns combined stdout and stderr output.
    """
    import io
    import contextlib

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(code, {"__builtins__": __builtins__})  # noqa: S102
        stdout = stdout_buf.getvalue()
        stderr = stderr_buf.getvalue()
        parts = []
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        return "\n".join(parts) if parts else "(no output)"
    except Exception as e:
        stderr = stderr_buf.getvalue()
        result = f"Error: {type(e).__name__}: {e}"
        if stderr:
            result += f"\nstderr:\n{stderr}"
        return result


@tool
def get_git_diff(path: str = ".") -> str:
    """Get the current git diff showing unstaged changes.

    Shows which files have been modified and a summary of changes.
    Use before suggesting code changes to understand the current state.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return f"Git error: {result.stderr.strip()}"
        return result.stdout.strip() or "No unstaged changes."
    except FileNotFoundError:
        return "Error: git not found."
    except subprocess.TimeoutExpired:
        return "Error: git diff timed out."
    except Exception as e:
        return f"Error: {e}"


@tool
def get_git_log(path: str = ".", n: int = 10) -> str:
    """Get the recent git commit history.

    Args:
        path: Repository path (default: current directory)
        n: Number of recent commits to show (default: 10)

    Returns one-line commit summary per entry with hash, author, date, message.
    """
    try:
        result = subprocess.run(
            ["git", "log", f"-{n}", "--oneline", "--decorate"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return f"Git error: {result.stderr.strip()}"
        return result.stdout.strip() or "No commits found."
    except FileNotFoundError:
        return "Error: git not found."
    except subprocess.TimeoutExpired:
        return "Error: git log timed out."
    except Exception as e:
        return f"Error: {e}"
