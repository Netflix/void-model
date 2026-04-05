from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


Workflow = Literal[
    "pass1_inference",
    "pass2_refine",
    "mask_pipeline",
    "mask_stage1_sam2",
    "mask_stage2_vlm",
    "mask_stage3_grey",
    "mask_stage4_combine",
    "point_selector_gui",
    "edit_quadmask_gui",
]


class Pass1Params(BaseModel):
    config_path: str = "config/quadmask_cogvideox.py"
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class Pass2Params(BaseModel):
    video_name: Optional[str] = None
    video_names: list[str] = Field(default_factory=list)
    data_rootdir: str = "./data"
    pass1_dir: str = "./pass1_outputs"
    output_dir: str = "./inference_with_warped_noise"
    model_name: str = "./CogVideoX-Fun-V1.5-5b-InP"
    model_checkpoint: str
    max_video_length: int = 197
    temporal_window_size: int = 85
    height: int = 384
    width: int = 672
    seed: int = 42
    guidance_scale: float = 6.0
    num_inference_steps: int = 50
    warped_noise_cache_dir: str = "./pass1_warped_noise_cache"
    skip_noise_generation: bool = False
    use_quadmask: bool = True


class MaskPipelineParams(BaseModel):
    config_points_json: str
    sam2_checkpoint: str = "../sam2_hiera_large.pt"
    device: Literal["cuda", "cpu"] = "cuda"


class MaskStage2Params(BaseModel):
    config_points_json: str
    model: str = "gemini-3-pro-preview"


class MaskStage3Params(BaseModel):
    config_points_json: str
    segmentation_model: Literal["langsam", "sam3"] = "sam3"


class MaskStage4Params(BaseModel):
    config_points_json: str


class PointSelectorParams(BaseModel):
    config_points_json: Optional[str] = None


class RunCreateRequest(BaseModel):
    workflow: Workflow
    params: dict[str, Any]
    name: Optional[str] = None


class PresetCreateRequest(BaseModel):
    name: str
    workflow: str
    params: dict[str, Any]


class ValidateConfigRequest(BaseModel):
    workflow: str
    params: dict[str, Any]


class PromptUpdateRequest(BaseModel):
    sequence_path: str
    bg: str


class CachePathRequest(BaseModel):
    path: str
