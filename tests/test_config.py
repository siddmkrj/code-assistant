"""Tests for config loading and saving."""
from __future__ import annotations

from pathlib import Path

import pytest

from coco.config.settings import (
    CocoConfig,
    load_config,
    save_config,
    get_default_config,
    CONFIG_FILENAME,
)


def test_default_config_values():
    config = get_default_config()
    assert config.model.complex_model == "claude-sonnet-4-6"
    assert config.model.simple_model == "claude-haiku-4-5-20251001"
    assert config.model.temperature == 0.0
    assert config.index.chunk_size == 1000
    assert config.safety.confirm_file_writes is True


def test_load_config_no_file(tmp_path: Path):
    """load_config returns defaults when no .coco file exists."""
    config = load_config(tmp_path)
    assert isinstance(config, CocoConfig)
    assert config.model.complex_model == "claude-sonnet-4-6"


def test_save_and_reload_config(tmp_path: Path):
    """save_config writes TOML that load_config can read back."""
    original = get_default_config()
    original.model.complex_model = "claude-opus-4-6"
    original.model.temperature = 0.5

    save_config(original, tmp_path)
    assert (tmp_path / CONFIG_FILENAME).exists()

    loaded = load_config(tmp_path)
    assert loaded.model.complex_model == "claude-opus-4-6"
    assert loaded.model.temperature == 0.5
    # Other fields stay at defaults
    assert loaded.model.simple_model == "claude-haiku-4-5-20251001"


def test_load_config_partial_toml(tmp_path: Path):
    """A .coco with only some fields overrides only those fields."""
    import tomli_w
    partial = {"model": {"complex_model": "custom-model"}}
    with open(tmp_path / CONFIG_FILENAME, "wb") as f:
        tomli_w.dump(partial, f)

    config = load_config(tmp_path)
    assert config.model.complex_model == "custom-model"
    assert config.model.simple_model == "claude-haiku-4-5-20251001"  # default kept
