# Config Mapping

This document maps frontend fields to backend request payloads and runtime CLI arguments.

## Pass 1 (`pass1_inference`)

Frontend sends:

- `workflow: "pass1_inference"`
- `params.config_path`
- `params.config_overrides` (all `config.*` keys)

Backend command:

- `python inference/cogvideox_fun/predict_v2v.py --config <config_path> --config.xxx.yyy=value ...`

Field mapping:

- `configPath` -> `config_path`
- `dataRootdir` -> `config_overrides["config.data.data_rootdir"]`
- `runSeqs` -> `config_overrides["config.experiment.run_seqs"]`
- `savePath` -> `config_overrides["config.experiment.save_path"]`
- `modelName` -> `config_overrides["config.video_model.model_name"]`
- `transformerPath` -> `config_overrides["config.video_model.transformer_path"]`
- `vaePath` -> `config_overrides["config.video_model.vae_path"]`
- `loraPath` -> `config_overrides["config.video_model.lora_path"]`
- `sampleSize` -> `config_overrides["config.data.sample_size"]`
- `dilateWidth` -> `config_overrides["config.data.dilate_width"]`
- `maxVideoLength` -> `config_overrides["config.data.max_video_length"]`
- `fps` -> `config_overrides["config.data.fps"]`
- `temporalWindowSize` -> `config_overrides["config.video_model.temporal_window_size"]`
- `temproalMultidiffusionStride` -> `config_overrides["config.video_model.temproal_multidiffusion_stride"]`
- `samplerName` -> `config_overrides["config.video_model.sampler_name"]`
- `denoiseStrength` -> `config_overrides["config.video_model.denoise_strength"]`
- `guidanceScale` -> `config_overrides["config.video_model.guidance_scale"]`
- `numInferenceSteps` -> `config_overrides["config.video_model.num_inference_steps"]`
- `negativePrompt` -> `config_overrides["config.video_model.negative_prompt"]`
- `loraWeight` -> `config_overrides["config.video_model.lora_weight"]`
- `useQuadmask` -> `config_overrides["config.video_model.use_quadmask"]`
- `useTrimask` -> `config_overrides["config.video_model.use_trimask"]`
- `useVaeMask` -> `config_overrides["config.video_model.use_vae_mask"]`
- `stackMask` -> `config_overrides["config.video_model.stack_mask"]`
- `zeroOutMaskRegion` -> `config_overrides["config.video_model.zero_out_mask_region"]`
- `mattingMode` -> `config_overrides["config.experiment.matting_mode"]`
- `skipIfExists` -> `config_overrides["config.experiment.skip_if_exists"]`
- `validation` -> `config_overrides["config.experiment.validation"]`
- `skipUnet` -> `config_overrides["config.experiment.skip_unet"]`
- `maskToVae` -> `config_overrides["config.experiment.mask_to_vae"]`
- `seed` -> `config_overrides["config.system.seed"]`
- `device` -> `config_overrides["config.system.device"]`
- `gpuMemoryMode` -> `config_overrides["config.system.gpu_memory_mode"]`
- `ulyssesDegree` -> `config_overrides["config.system.ulysses_degree"]`
- `ringDegree` -> `config_overrides["config.system.ring_degree"]`
- `allowSkippingError` -> `config_overrides["config.system.allow_skipping_error"]`

## Pass 2 (`pass2_refine`)

Frontend sends:

- `workflow: "pass2_refine"`
- `params.video_names`
- `params.data_rootdir`
- `params.pass1_dir`
- `params.output_dir`
- `params.model_name`
- `params.model_checkpoint`
- `params.max_video_length`
- `params.temporal_window_size`
- `params.height`
- `params.width`
- `params.seed`
- `params.guidance_scale`
- `params.num_inference_steps`
- `params.warped_noise_cache_dir`
- `params.skip_noise_generation`
- `params.use_quadmask`

Backend command:

- `python inference/cogvideox_fun/inference_with_pass1_warped_noise.py ...`

CLI mapping:

- `video_names` -> `--video_names ...`
- `data_rootdir` -> `--data_rootdir`
- `pass1_dir` -> `--pass1_dir`
- `output_dir` -> `--output_dir`
- `model_name` -> `--model_name`
- `model_checkpoint` -> `--model_checkpoint`
- `max_video_length` -> `--max_video_length`
- `temporal_window_size` -> `--temporal_window_size`
- `height` -> `--height`
- `width` -> `--width`
- `seed` -> `--seed`
- `guidance_scale` -> `--guidance_scale`
- `num_inference_steps` -> `--num_inference_steps`
- `warped_noise_cache_dir` -> `--warped_noise_cache_dir`
- `skip_noise_generation=true` -> `--skip_noise_generation`
- `use_quadmask=true` -> `--use_quadmask`

## Mask Workflow (`mask_pipeline` and stage-specific)

Frontend derives target workflow from execution mode:

- `full` -> `mask_pipeline`
- `stage1` -> `mask_stage1_sam2`
- `stage2` -> `mask_stage2_vlm`
- `stage3` -> `mask_stage3_grey`
- `stage4` -> `mask_stage4_combine`

Shared fields:

- `maskConfigPath` -> `config_points_json`

Stage-specific mapping:

- Full/Stage1:
  - `maskSam2Checkpoint` -> `sam2_checkpoint`
  - `maskDevice` -> `device`
- Stage2:
  - `maskVlmModel` -> `model`
- Stage3:
  - `maskSegmentationModel` -> `segmentation_model`

## Validation and Safety

- Frontend calls `POST /validate/config` before creating runs.
- Backend rejects unknown workflows and invalid override keys.
- Backend validates path existence for key inputs and returns:
  - `errors` (blocking)
  - `warnings` (non-blocking)
