"""File system tools for coco agents.

All tools are decorated with @tool for LangChain compatibility.
Error handling returns descriptive strings rather than raising exceptions,
so agents can read the error and decide what to do next.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool


@tool
def read_file(path: str) -> str:
    """Read and return the contents of a file at the given path."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except PermissionError:
        return f"Error: Permission denied reading: {path}"
    except IsADirectoryError:
        return f"Error: {path} is a directory, not a file. Use list_directory instead."
    except UnicodeDecodeError:
        return f"Error: {path} contains binary data and cannot be read as text."
    except Exception as e:
        return f"Error reading {path}: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories as needed.

    WARNING: This will overwrite the file if it already exists.
    The CLI will prompt for confirmation before this tool is called
    when safety.confirm_file_writes is true (the default).
    """
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {path}"
    except PermissionError:
        return f"Error: Permission denied writing to: {path}"
    except Exception as e:
        return f"Error writing to {path}: {e}"


@tool
def list_directory(path: str = ".") -> str:
    """List files and subdirectories at the given path.

    Directories are prefixed with [D], files with [F].
    """
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: Path does not exist: {path}"
        if not p.is_dir():
            return f"Error: {path} is a file, not a directory."
        entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
        lines = []
        for entry in entries:
            prefix = "[F]" if entry.is_file() else "[D]"
            size = f" ({entry.stat().st_size:,} bytes)" if entry.is_file() else ""
            lines.append(f"{prefix} {entry.name}{size}")
        return "\n".join(lines) if lines else "(empty directory)"
    except PermissionError:
        return f"Error: Permission denied listing: {path}"
    except Exception as e:
        return f"Error listing {path}: {e}"


@tool
def search_in_files(pattern: str, directory: str = ".", file_extension: str = "") -> str:
    """Search for a text pattern in files under a directory.

    Args:
        pattern: Text or regex pattern to search for
        directory: Directory to search in (default: current directory)
        file_extension: Filter by extension e.g. '.py' (default: all files)

    Returns matching lines with file path and line number.
    """
    import subprocess  # noqa: PLC0415
    try:
        cmd = ["grep", "-rn"]
        if file_extension:
            cmd += [f"--include=*{file_extension}"]
        cmd += ["--", pattern, directory]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if not output:
            return f"No matches found for '{pattern}' in {directory}"
        # Truncate very large results
        lines = output.strip().split("\n")
        if len(lines) > 100:
            return "\n".join(lines[:100]) + f"\n... (truncated, {len(lines)} total matches)"
        return output.strip()
    except subprocess.TimeoutExpired:
        return "Search timed out after 30 seconds."
    except FileNotFoundError:
        return "Error: grep not available on this system."
    except Exception as e:
        return f"Error searching: {e}"
