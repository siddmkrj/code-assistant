"""Shared pytest fixtures for coco tests."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_project_dir(tmp_path: Path) -> Path:
    """A temporary directory simulating a project root."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "hello.py").write_text("def hello():\n    return 'world'\n")
    (tmp_path / "README.md").write_text("# Test Project\n")
    return tmp_path


@pytest.fixture
def default_config(tmp_project_dir: Path):
    """A default CocoConfig pointing at tmp_project_dir."""
    from coco.config.settings import CocoConfig, ModelConfig, IndexConfig, MemoryConfig, SafetyConfig
    config = CocoConfig(
        model=ModelConfig(),
        index=IndexConfig(),
        memory=MemoryConfig(history_dir=str(tmp_project_dir / "history")),
        safety=SafetyConfig(),
        working_directory=str(tmp_project_dir),
    )
    return config
