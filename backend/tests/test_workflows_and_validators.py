from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from fastapi import HTTPException

from backend.app.config import PROJECT_ROOT
from backend.app.validators import directory_stats, load_prompt_bg, path_under_project
from backend.app.workflows import build_command, override_args


class WorkflowAndValidatorTests(unittest.TestCase):
    def test_override_args_formats_booleans_and_values(self) -> None:
        args = override_args(
            {
                "config.foo.bool_true": True,
                "config.foo.bool_false": False,
                "config.foo.num": 7,
                "config.foo.text": "abc",
            }
        )
        self.assertIn("--config.foo.bool_true=true", args)
        self.assertIn("--config.foo.bool_false=false", args)
        self.assertIn("--config.foo.num=7", args)
        self.assertIn("--config.foo.text=abc", args)

    def test_override_args_rejects_non_config_keys(self) -> None:
        with self.assertRaises(HTTPException):
            override_args({"bad.key": "value"})

    def test_build_command_pass1(self) -> None:
        command, output_dir = build_command(
            "pass1_inference",
            {
                "config_path": "config/quadmask_cogvideox.py",
                "config_overrides": {
                    "config.experiment.save_path": "./outputs",
                    "config.data.data_rootdir": "./sample",
                },
            },
        )
        self.assertEqual(command[0:3], ["python", "inference/cogvideox_fun/predict_v2v.py", "--config"])
        self.assertIn("--config.experiment.save_path=./outputs", command)
        self.assertEqual(output_dir, "./outputs")

    def test_build_command_pass2_requires_video_fields(self) -> None:
        with self.assertRaises(HTTPException):
            build_command(
                "pass2_refine",
                {
                    "model_checkpoint": "./void_pass2.safetensors",
                    "data_rootdir": "./sample",
                    "pass1_dir": "./outputs",
                },
            )

    def test_path_under_project_rejects_escape(self) -> None:
        with self.assertRaises(HTTPException):
            path_under_project("../../etc")

    def test_directory_stats_and_load_prompt_bg(self) -> None:
        rel = Path("backend/state/test-validator-smoke")
        root = PROJECT_ROOT / rel
        root.mkdir(parents=True, exist_ok=True)
        try:
            (root / "a.txt").write_text("hello", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir(exist_ok=True)
            (nested / "b.txt").write_bytes(b"1234")
            prompt = root / "prompt.json"
            prompt.write_text(json.dumps({"bg": "test background"}), encoding="utf-8")

            stats = directory_stats(root)
            self.assertTrue(stats["exists"])
            self.assertEqual(stats["files"], 3)
            self.assertGreaterEqual(stats["bytes"], 5)
            self.assertEqual(load_prompt_bg(prompt), "test background")
        finally:
            shutil.rmtree(root, ignore_errors=True)
