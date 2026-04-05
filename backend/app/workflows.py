from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException

from .schemas import (
    MaskPipelineParams,
    MaskStage2Params,
    MaskStage3Params,
    MaskStage4Params,
    Pass1Params,
    Pass2Params,
    PointSelectorParams,
)


def override_args(config_overrides: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in config_overrides.items():
        if not key.startswith("config."):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid override key '{key}'. Keys must start with 'config.'",
            )
        normalized = "true" if isinstance(value, bool) and value else "false" if isinstance(value, bool) else str(value)
        args.append(f"--{key}={normalized}")
    return args


def build_command(workflow: str, params: dict[str, Any]) -> tuple[list[str], Optional[str]]:
    if workflow == "pass1_inference":
        p = Pass1Params(**params)
        cmd = [
            "python",
            "inference/cogvideox_fun/predict_v2v.py",
            "--config",
            p.config_path,
            *override_args(p.config_overrides),
        ]
        output_dir = p.config_overrides.get("config.experiment.save_path")
        return cmd, str(output_dir) if output_dir else None

    if workflow == "pass2_refine":
        p = Pass2Params(**params)
        if not p.video_name and not p.video_names:
            raise HTTPException(
                status_code=400,
                detail="Provide either video_name or video_names for pass2_refine",
            )
        cmd = [
            "python",
            "inference/cogvideox_fun/inference_with_pass1_warped_noise.py",
            "--data_rootdir",
            p.data_rootdir,
            "--pass1_dir",
            p.pass1_dir,
            "--output_dir",
            p.output_dir,
            "--model_name",
            p.model_name,
            "--model_checkpoint",
            p.model_checkpoint,
            "--max_video_length",
            str(p.max_video_length),
            "--temporal_window_size",
            str(p.temporal_window_size),
            "--height",
            str(p.height),
            "--width",
            str(p.width),
            "--seed",
            str(p.seed),
            "--guidance_scale",
            str(p.guidance_scale),
            "--num_inference_steps",
            str(p.num_inference_steps),
            "--warped_noise_cache_dir",
            p.warped_noise_cache_dir,
        ]
        if p.video_name:
            cmd.extend(["--video_name", p.video_name])
        if p.video_names:
            cmd.extend(["--video_names", *p.video_names])
        if p.skip_noise_generation:
            cmd.append("--skip_noise_generation")
        if p.use_quadmask:
            cmd.append("--use_quadmask")
        return cmd, p.output_dir

    if workflow == "mask_pipeline":
        p = MaskPipelineParams(**params)
        return [
            "bash",
            "VLM-MASK-REASONER/run_pipeline.sh",
            p.config_points_json,
            "--sam2-checkpoint",
            p.sam2_checkpoint,
            "--device",
            p.device,
        ], None

    if workflow == "mask_stage1_sam2":
        p = MaskPipelineParams(**params)
        return [
            "python",
            "VLM-MASK-REASONER/stage1_sam2_segmentation.py",
            "--config",
            p.config_points_json,
            "--sam2-checkpoint",
            p.sam2_checkpoint,
            "--device",
            p.device,
        ], None

    if workflow == "mask_stage2_vlm":
        p = MaskStage2Params(**params)
        return [
            "python",
            "VLM-MASK-REASONER/stage2_vlm_analysis.py",
            "--config",
            p.config_points_json,
            "--model",
            p.model,
        ], None

    if workflow == "mask_stage3_grey":
        p = MaskStage3Params(**params)
        return [
            "python",
            "VLM-MASK-REASONER/stage3a_generate_grey_masks_v2.py",
            "--config",
            p.config_points_json,
            "--segmentation-model",
            p.segmentation_model,
        ], None

    if workflow == "mask_stage4_combine":
        p = MaskStage4Params(**params)
        return [
            "python",
            "VLM-MASK-REASONER/stage4_combine_masks.py",
            "--config",
            p.config_points_json,
        ], None

    if workflow == "point_selector_gui":
        p = PointSelectorParams(**params)
        cmd = ["python", "VLM-MASK-REASONER/point_selector_gui.py"]
        if p.config_points_json:
            cmd.extend(["--config", p.config_points_json])
        return cmd, None

    if workflow == "edit_quadmask_gui":
        return ["python", "VLM-MASK-REASONER/edit_quadmask.py"], None

    raise HTTPException(status_code=400, detail=f"Unsupported workflow: {workflow}")
