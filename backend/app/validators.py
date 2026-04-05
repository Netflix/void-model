from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException

from .config import PROJECT_ROOT
from .schemas import Pass1Params, Pass2Params


def safe_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def path_under_project(path_value: str) -> Path:
    resolved = Path(safe_path(path_value)).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Path must be inside project root")
    return resolved


def validate_paths(
    workflow: str, params: dict[str, Any], command: list[str]
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if len(command) >= 2 and command[0] in {"python", "bash"}:
        script_path = PROJECT_ROOT / command[1]
        if not script_path.exists():
            errors.append(f"Script not found: {command[1]}")

    if workflow == "pass1_inference":
        p = Pass1Params(**params)
        cfg = PROJECT_ROOT / p.config_path
        if not cfg.exists():
            errors.append(f"Config file not found: {p.config_path}")

        data_root = p.config_overrides.get("config.data.data_rootdir")
        if data_root and not Path(safe_path(str(data_root))).exists():
            errors.append(f"Data rootdir not found: {data_root}")

        model_name = p.config_overrides.get("config.video_model.model_name")
        if model_name and not Path(safe_path(str(model_name))).exists():
            warnings.append(f"Base model path not found yet: {model_name}")

        transformer_path = p.config_overrides.get("config.video_model.transformer_path")
        if transformer_path and not Path(safe_path(str(transformer_path))).exists():
            warnings.append(f"Transformer checkpoint not found yet: {transformer_path}")

        run_seqs = str(p.config_overrides.get("config.experiment.run_seqs", "")).strip()
        if not run_seqs:
            warnings.append("No run sequences configured (config.experiment.run_seqs is empty).")

    elif workflow == "pass2_refine":
        p = Pass2Params(**params)
        if not Path(safe_path(p.model_checkpoint)).exists():
            errors.append(f"Pass2 model checkpoint not found: {p.model_checkpoint}")
        if not Path(safe_path(p.data_rootdir)).exists():
            errors.append(f"Data rootdir not found: {p.data_rootdir}")
        if not Path(safe_path(p.pass1_dir)).exists():
            errors.append(f"Pass1 output dir not found: {p.pass1_dir}")
        if not p.video_name and not p.video_names:
            errors.append("Provide video_name or video_names.")

    elif workflow in {
        "mask_pipeline",
        "mask_stage1_sam2",
        "mask_stage2_vlm",
        "mask_stage3_grey",
        "mask_stage4_combine",
        "point_selector_gui",
    }:
        cfg_path = params.get("config_points_json")
        if cfg_path and not Path(safe_path(str(cfg_path))).exists():
            errors.append(f"Config points JSON not found: {cfg_path}")

        if workflow in {"mask_pipeline", "mask_stage1_sam2"}:
            sam2 = params.get("sam2_checkpoint")
            if sam2 and not Path(safe_path(str(sam2))).exists():
                warnings.append(f"SAM2 checkpoint not found yet: {sam2}")

    return errors, warnings


def directory_stats(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_dir():
        return {"exists": False, "files": 0, "bytes": 0}

    files = 0
    total_bytes = 0
    for node in path.rglob("*"):
        if node.is_file():
            files += 1
            total_bytes += node.stat().st_size

    return {"exists": True, "files": files, "bytes": total_bytes}


def load_prompt_bg(prompt_json: Path) -> Optional[str]:
    if not prompt_json.exists():
        return None
    try:
        payload = json.loads(prompt_json.read_text(encoding="utf-8"))
        return payload.get("bg")
    except Exception:
        return None
