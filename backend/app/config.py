from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
STATE_DIR = BACKEND_ROOT / "state"
LOG_DIR = STATE_DIR / "logs"
RUNS_FILE = STATE_DIR / "runs.json"
PRESETS_FILE = STATE_DIR / "presets.json"


def ensure_state_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
