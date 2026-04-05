export type Workflow =
  | "pass1_inference"
  | "pass2_refine"
  | "mask_pipeline"
  | "mask_stage1_sam2"
  | "mask_stage2_vlm"
  | "mask_stage3_grey"
  | "mask_stage4_combine"
  | "point_selector_gui"
  | "edit_quadmask_gui";

export type RunStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface RunRecord {
  id: string;
  workflow: Workflow;
  status: RunStatus;
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
  exit_code: number | null;
  command: string[];
  name: string | null;
  params: Record<string, unknown>;
  log_path: string;
  output_dir: string | null;
  error: string | null;
}

export interface PresetRecord {
  id: string;
  name: string;
  workflow: string;
  params: Record<string, unknown>;
  created_at: string;
}

export interface ArtifactFile {
  path: string;
  relative: string;
  size_bytes: number;
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  command_preview: string[];
}

export interface EnvCheck {
  project_root: string;
  python: string | null;
  ffmpeg: string | null;
  nvidia_smi: string | null;
  cuda: {
    available: boolean | null;
    error: string | null;
  };
  gemini_api_key_set: boolean;
}
