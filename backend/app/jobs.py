from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .config import PROJECT_ROOT, now_iso
from .stores import RunStore


def run_worker(run_id: str, cmd: list[str], log_path: Path, store: RunStore) -> None:
    env = os.environ.copy()

    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"$ {' '.join(cmd)}\\n\\n")
            log_file.flush()

            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1,
            )
            store.patch(run_id, status="running", started_at=now_iso())
            store.set_proc(run_id, proc)

            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    log_file.write(line)
                    log_file.flush()
                exit_code = proc.wait()
            finally:
                store.clear_proc(run_id)
    except Exception as exc:
        store.patch(
            run_id,
            status="failed",
            ended_at=now_iso(),
            exit_code=-1,
            error=f"Failed to launch process: {exc}",
        )
        return

    current = store.get(run_id)
    if current and current.status == "cancelled":
        return

    if exit_code == 0:
        store.patch(run_id, status="completed", ended_at=now_iso(), exit_code=0)
    else:
        store.patch(
            run_id,
            status="failed",
            ended_at=now_iso(),
            exit_code=exit_code,
            error=f"Process exited with code {exit_code}",
        )
