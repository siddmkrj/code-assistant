"""Tests for file tools and command parsing."""
from __future__ import annotations

from pathlib import Path

import pytest


# ----------------------------------------------------------------- file tools

def test_read_file_success(tmp_path: Path):
    from coco.tools.file_tools import read_file
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    result = read_file.invoke({"path": str(f)})
    assert result == "hello world"


def test_read_file_not_found(tmp_path: Path):
    from coco.tools.file_tools import read_file
    result = read_file.invoke({"path": str(tmp_path / "nonexistent.txt")})
    assert "Error" in result
    assert "not found" in result.lower()


def test_write_file_creates_file(tmp_path: Path):
    from coco.tools.file_tools import write_file
    target = tmp_path / "output.py"
    result = write_file.invoke({"path": str(target), "content": "print('hi')"})
    assert "Successfully" in result
    assert target.read_text() == "print('hi')"


def test_write_file_creates_parent_dirs(tmp_path: Path):
    from coco.tools.file_tools import write_file
    target = tmp_path / "nested" / "deep" / "file.py"
    write_file.invoke({"path": str(target), "content": "pass"})
    assert target.exists()


def test_list_directory(tmp_path: Path):
    from coco.tools.file_tools import list_directory
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "subdir").mkdir()
    result = list_directory.invoke({"path": str(tmp_path)})
    assert "a.py" in result
    assert "b.txt" in result
    assert "subdir" in result


def test_list_directory_nonexistent(tmp_path: Path):
    from coco.tools.file_tools import list_directory
    result = list_directory.invoke({"path": str(tmp_path / "nope")})
    assert "Error" in result


# ---------------------------------------------------------------- command parsing

def test_parse_command_slash():
    from coco.cli.commands import parse_command
    cmd, args = parse_command("/help")
    assert cmd == "help"
    assert args == ""


def test_parse_command_with_args():
    from coco.cli.commands import parse_command
    cmd, args = parse_command("/ask what is this?")
    assert cmd == "ask"
    assert args == "what is this?"


def test_parse_natural_input():
    from coco.cli.commands import parse_command
    cmd, args = parse_command("write me a function")
    assert cmd is None
    assert args == "write me a function"


def test_parse_bare_slash():
    from coco.cli.commands import parse_command
    cmd, args = parse_command("/")
    assert cmd is None


# ---------------------------------------------------------------- history logger

def test_history_logger_creates_file(tmp_path: Path):
    from coco.memory.history import HistoryLogger
    logger = HistoryLogger(history_dir=tmp_path)
    logger.log_user("hello", task_type="ask")
    logger.log_assistant("world", agent="ask_agent")
    assert logger.log_path.exists()
    lines = logger.log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    import json
    entry = json.loads(lines[0])
    assert entry["role"] == "user"
    assert entry["content"] == "hello"


# ---------------------------------------------------------------- base agent CLARIFY

def test_extract_clarification():
    from coco.agents.base import BaseCocoAgent
    content = "I need to know more. [CLARIFY]Which framework?[/CLARIFY] Thanks."
    result = BaseCocoAgent._extract_clarification(content)
    assert result == "Which framework?"


def test_extract_clarification_no_tag():
    from coco.agents.base import BaseCocoAgent
    content = "No clarification needed."
    result = BaseCocoAgent._extract_clarification(content)
    assert result == content
