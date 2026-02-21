"""Configuration management for coco.

Config is stored as TOML in a .coco file in the project directory.
Global/default config lives at ~/.coco/config.toml.
Python 3.11+ tomllib (stdlib) is used for reading; tomli-w for writing.
"""
from __future__ import annotations

import tomllib
import tomli_w
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


CONFIG_FILENAME = ".coco"
APP_DIR = Path.home() / ".coco"


@dataclass
class ModelConfig:
    complex_model: str = "claude-sonnet-4-6"
    simple_model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class IndexConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    persist_dir: str = ".coco_index"
    collection_name: str = "codebase"
    include_extensions: list = field(default_factory=lambda: [
        ".py", ".ts", ".js", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb",
        ".cpp", ".c", ".h", ".cs", ".php", ".swift", ".kt",
        ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".env.example",
        ".sh", ".bash", ".zsh", ".sql",
    ])
    exclude_dirs: list = field(default_factory=lambda: [
        ".git", "node_modules", "__pycache__", ".venv", "venv", ".env",
        "dist", "build", ".coco_index", ".mypy_cache", ".ruff_cache",
        "coverage", ".coverage", "htmlcov", ".tox", "eggs", "*.egg-info",
    ])


@dataclass
class MemoryConfig:
    max_token_limit: int = 4000
    history_dir: str = str(APP_DIR / "history")


@dataclass
class SafetyConfig:
    confirm_file_writes: bool = True
    confirm_shell_commands: bool = True
    allowed_shell_commands: list = field(default_factory=lambda: [
        "ls", "cat", "head", "tail", "grep", "find", "git", "python", "pip",
    ])


@dataclass
class CocoConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    working_directory: str = "."


def load_config(project_dir: Optional[Path] = None) -> CocoConfig:
    """Load config from .coco TOML file, merging with defaults.

    Looks for .coco in project_dir (or CWD if not specified).
    Falls back to ~/.coco/config.toml for global defaults.
    """
    config = get_default_config()

    # Load global config first
    global_config_path = APP_DIR / "config.toml"
    if global_config_path.exists():
        try:
            with open(global_config_path, "rb") as f:
                data = tomllib.load(f)
            config = _merge_config(config, data)
        except Exception:
            pass  # Silently skip malformed global config

    # Load project config (overrides global)
    search_dir = project_dir or Path.cwd()
    config_path = search_dir / CONFIG_FILENAME
    if config_path.exists():
        try:
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
            config = _merge_config(config, data)
        except Exception:
            pass  # Silently skip malformed project config

    # Set working_directory to where the config was found (or CWD)
    config.working_directory = str(search_dir.resolve())
    return config


def save_config(config: CocoConfig, project_dir: Optional[Path] = None) -> Path:
    """Write config to .coco TOML file in project_dir (or CWD)."""
    path = (project_dir or Path.cwd()) / CONFIG_FILENAME
    with open(path, "wb") as f:
        tomli_w.dump(asdict(config), f)
    return path


def get_default_config() -> CocoConfig:
    """Return a fresh default config."""
    return CocoConfig()


def _merge_config(base: CocoConfig, data: dict) -> CocoConfig:
    """Merge a TOML dict into a CocoConfig, updating only keys present in data."""
    if "model" in data:
        for key, val in data["model"].items():
            if hasattr(base.model, key):
                setattr(base.model, key, val)
    if "index" in data:
        for key, val in data["index"].items():
            if hasattr(base.index, key):
                setattr(base.index, key, val)
    if "memory" in data:
        for key, val in data["memory"].items():
            if hasattr(base.memory, key):
                setattr(base.memory, key, val)
    if "safety" in data:
        for key, val in data["safety"].items():
            if hasattr(base.safety, key):
                setattr(base.safety, key, val)
    if "working_directory" in data:
        base.working_directory = data["working_directory"]
    return base
