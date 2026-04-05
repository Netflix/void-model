from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from .models import PresetRecord, RunRecord


class RunStore:
    def __init__(self, runs_file: Path, now_fn: Callable[[], str]) -> None:
        self._lock = threading.Lock()
        self._runs_file = runs_file
        self._now_fn = now_fn
        self._runs: dict[str, RunRecord] = {}
        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._load()

    def _load(self) -> None:
        if not self._runs_file.exists():
            return
        try:
            payload = json.loads(self._runs_file.read_text())
            for item in payload:
                record = RunRecord(**item)
                if record.status == "running":
                    record.status = "failed"
                    record.error = "Backend restarted while run was active"
                    record.ended_at = self._now_fn()
                self._runs[record.id] = record
        except Exception:
            self._runs = {}

    def _save(self) -> None:
        self._runs_file.write_text(json.dumps([asdict(v) for v in self._runs.values()], indent=2))

    def list(self) -> list[RunRecord]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    def get(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            return self._runs.get(run_id)

    def create(self, record: RunRecord) -> None:
        with self._lock:
            self._runs[record.id] = record
            self._save()

    def set_proc(self, run_id: str, proc: subprocess.Popen[str]) -> None:
        with self._lock:
            self._procs[run_id] = proc

    def get_proc(self, run_id: str) -> Optional[subprocess.Popen[str]]:
        with self._lock:
            return self._procs.get(run_id)

    def clear_proc(self, run_id: str) -> None:
        with self._lock:
            self._procs.pop(run_id, None)

    def patch(self, run_id: str, **updates: Any) -> None:
        with self._lock:
            run = self._runs[run_id]
            for key, value in updates.items():
                setattr(run, key, value)
            self._save()


class PresetStore:
    def __init__(self, presets_file: Path) -> None:
        self._lock = threading.Lock()
        self._presets_file = presets_file
        self._presets: dict[str, PresetRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self._presets_file.exists():
            return
        try:
            payload = json.loads(self._presets_file.read_text())
            for item in payload:
                record = PresetRecord(**item)
                self._presets[record.id] = record
        except Exception:
            self._presets = {}

    def _save(self) -> None:
        self._presets_file.write_text(json.dumps([asdict(v) for v in self._presets.values()], indent=2))

    def list(self) -> list[PresetRecord]:
        with self._lock:
            return sorted(self._presets.values(), key=lambda p: p.created_at, reverse=True)

    def create(self, preset: PresetRecord) -> None:
        with self._lock:
            self._presets[preset.id] = preset
            self._save()
