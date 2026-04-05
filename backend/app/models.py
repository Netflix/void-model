from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass
class RunRecord:
    id: str
    workflow: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    created_at: str
    started_at: Optional[str]
    ended_at: Optional[str]
    exit_code: Optional[int]
    command: list[str]
    name: Optional[str]
    params: dict[str, Any]
    log_path: str
    output_dir: Optional[str]
    error: Optional[str] = None


@dataclass
class PresetRecord:
    id: str
    name: str
    workflow: str
    params: dict[str, Any]
    created_at: str
