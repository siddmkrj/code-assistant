"""Session history logger.

Appends all interactions to a JSONL file at:
  ~/.coco/history/YYYY-MM-DD-<session_id>.jsonl

Each line is a JSON object with: ts, session, role, content, and optional metadata.
Files are append-only and never modified after writing.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class HistoryLogger:
    """Append-only JSONL logger for all coco interactions."""

    def __init__(self, history_dir: str | Path):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = str(uuid.uuid4())[:8]
        self._log_path = self._build_log_path()

    def _build_log_path(self) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.history_dir / f"{date_str}-{self._session_id}.jsonl"

    def log(
        self,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write a single log entry."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session": self._session_id,
            "role": role,
            "content": content,
            **(metadata or {}),
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_user(self, content: str, task_type: str = "auto") -> None:
        self.log("user", content, {"task_type": task_type})

    def log_assistant(self, content: str, agent: str = "coco") -> None:
        self.log("assistant", content, {"agent": agent})

    def log_system(self, event: str, details: Optional[dict] = None) -> None:
        self.log("system", event, details)

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def log_path(self) -> Path:
        return self._log_path
