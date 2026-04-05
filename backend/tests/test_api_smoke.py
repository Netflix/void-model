from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app import api as api_module
from backend.app.config import PROJECT_ROOT


class ApiSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_root = Path(self.temp_dir.name)
        self.logs_dir = self.state_root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.runs_file = self.state_root / "runs.json"
        self.presets_file = self.state_root / "presets.json"

        self.patches = [
            patch.object(api_module, "RUNS_FILE", self.runs_file),
            patch.object(api_module, "PRESETS_FILE", self.presets_file),
            patch.object(api_module, "LOG_DIR", self.logs_dir),
            patch.object(api_module, "subprocess_run_torch_cuda", lambda: {"ok": True, "available": False}),
            patch.object(api_module, "ensure_state_dirs", lambda: None),
        ]
        for p in self.patches:
            p.start()

        self.app = api_module.create_app()
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        for p in reversed(self.patches):
            p.stop()
        self.temp_dir.cleanup()

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("time", payload)

    def test_validate_config_rejects_missing_pass2_paths(self) -> None:
        response = self.client.post(
            "/validate/config",
            json={
                "workflow": "pass2_refine",
                "params": {
                    "video_name": "lime",
                    "model_checkpoint": "./does-not-exist.safetensors",
                    "data_rootdir": "./does-not-exist-data",
                    "pass1_dir": "./does-not-exist-pass1",
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["valid"])
        joined = "\n".join(payload["errors"])
        self.assertIn("Pass2 model checkpoint not found", joined)
        self.assertIn("Data rootdir not found", joined)
        self.assertIn("Pass1 output dir not found", joined)

    def test_prompt_update_writes_prompt_json(self) -> None:
        rel_sequence = Path("backend/state/test-seq-api-smoke")
        abs_sequence = PROJECT_ROOT / rel_sequence
        abs_sequence.mkdir(parents=True, exist_ok=True)
        try:
            response = self.client.post(
                "/data/prompt",
                json={"sequence_path": str(rel_sequence), "bg": "A bright red studio background"},
            )
            self.assertEqual(response.status_code, 200)
            prompt_path = abs_sequence / "prompt.json"
            self.assertTrue(prompt_path.exists())
            data = json.loads(prompt_path.read_text(encoding="utf-8"))
            self.assertEqual(data, {"bg": "A bright red studio background"})
        finally:
            shutil.rmtree(abs_sequence, ignore_errors=True)

    def test_cache_clear_creates_directory_and_removes_files(self) -> None:
        rel_cache = Path("backend/state/test-cache-api-smoke")
        abs_cache = PROJECT_ROOT / rel_cache
        abs_cache.mkdir(parents=True, exist_ok=True)
        (abs_cache / "temp.bin").write_bytes(b"123")
        (abs_cache / "nested").mkdir(exist_ok=True)
        (abs_cache / "nested" / "inner.txt").write_text("abc", encoding="utf-8")
        try:
            response = self.client.post("/cache/clear", json={"path": str(rel_cache)})
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["ok"])
            self.assertTrue(abs_cache.exists())
            self.assertEqual(payload["files"], 0)
        finally:
            shutil.rmtree(abs_cache, ignore_errors=True)
