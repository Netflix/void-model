export type Workflow = "pass1_inference" | "pass2_refine" | "mask_pipeline";

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
