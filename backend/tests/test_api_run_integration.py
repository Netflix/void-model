from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app import api as api_module
from backend.app.config import now_iso


class ApiRunIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_root = Path(self.temp_dir.name)
        self.logs_dir = self.state_root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.runs_file = self.state_root / "runs.json"
        self.presets_file = self.state_root / "presets.json"

        self.worker_mode = "complete"
        self.worker_started = Event()
        self.release_worker = Event()

        def fake_worker(run_id: str, cmd: list[str], log_path: Path, store):
            store.patch(run_id, status="running", started_at=now_iso())
            if self.worker_mode == "complete":
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write("integration: run completed\n")
                store.patch(run_id, status="completed", ended_at=now_iso(), exit_code=0)
                return

            class DummyProc:
                pid = 999999

            store.set_proc(run_id, DummyProc())  # type: ignore[arg-type]
            self.worker_started.set()
            self.release_worker.wait(timeout=3)
            store.clear_proc(run_id)
            current = store.get(run_id)
            if current and current.status == "cancelled":
                return
            store.patch(run_id, status="completed", ended_at=now_iso(), exit_code=0)

        self.patches = [
            patch.object(api_module, "RUNS_FILE", self.runs_file),
            patch.object(api_module, "PRESETS_FILE", self.presets_file),
            patch.object(api_module, "LOG_DIR", self.logs_dir),
            patch.object(api_module, "ensure_state_dirs", lambda: None),
            patch.object(api_module, "subprocess_run_torch_cuda", lambda: {"ok": True, "available": False}),
            patch.object(api_module, "run_worker", fake_worker),
        ]
        for p in self.patches:
            p.start()

        self.client = TestClient(api_module.create_app())

    def tearDown(self) -> None:
        self.release_worker.set()
        for p in reversed(self.patches):
            p.stop()
        self.temp_dir.cleanup()

    def _wait_for_status(self, run_id: str, expected: str, timeout: float = 2.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            payload = self.client.get(f"/runs/{run_id}").json()
            if payload.get("status") == expected:
                return payload
            time.sleep(0.05)
        self.fail(f"Run {run_id} did not reach status {expected}")

    def test_run_lifecycle_and_logs_stream(self) -> None:
        create = self.client.post(
            "/runs",
            json={"workflow": "edit_quadmask_gui", "params": {}, "name": "integration-run"},
        )
        self.assertEqual(create.status_code, 200)
        run_id = create.json()["id"]

        run = self._wait_for_status(run_id, "completed")
        self.assertEqual(run["exit_code"], 0)

        logs = self.client.get(f"/runs/{run_id}/logs").json()
        self.assertIn("integration: run completed", logs["text"])

        stream = self.client.get(f"/runs/{run_id}/logs/stream")
        self.assertEqual(stream.status_code, 200)
        self.assertIn("integration: run completed", stream.text)
        self.assertIn("event: end", stream.text)
        self.assertIn("completed", stream.text)

    def test_cancel_running_run(self) -> None:
        self.worker_mode = "cancel"
        create = self.client.post(
            "/runs",
            json={"workflow": "point_selector_gui", "params": {}},
        )
        self.assertEqual(create.status_code, 200)
        run_id = create.json()["id"]

        self.assertTrue(self.worker_started.wait(timeout=1.0))

        with patch.object(api_module.os, "kill", lambda *_args, **_kwargs: None):
            cancelled = self.client.post(f"/runs/{run_id}/cancel")

        self.assertEqual(cancelled.status_code, 200)
        payload = cancelled.json()
        self.assertEqual(payload["status"], "cancelled")
        self.assertEqual(payload["exit_code"], -15)
