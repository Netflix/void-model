"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

import {
  cancelRun,
  createPreset,
  createRun,
  getEnvCheck,
  getRunLogs,
  listPresets,
  listRuns,
} from "@/lib/api";
import type { EnvCheck, PresetRecord, RunRecord, Workflow } from "@/lib/types";

const sectionClass =
  "rounded-xl border border-slate-200 bg-white/90 p-4 shadow-sm backdrop-blur-sm";

type Pass1Form = {
  configPath: string;
  dataRootdir: string;
  runSeqs: string;
  savePath: string;
  modelName: string;
  transformerPath: string;
  vaePath: string;
  loraPath: string;
  sampleSize: string;
  dilateWidth: number;
  maxVideoLength: number;
  fps: number;
  temporalWindowSize: number;
  temproalMultidiffusionStride: number;
  samplerName: "DDIM_Origin" | "DDIM_Cog" | "Euler" | "Euler A" | "DPM++" | "PNDM";
  denoiseStrength: number;
  guidanceScale: number;
  numInferenceSteps: number;
  negativePrompt: string;
  loraWeight: number;
  useQuadmask: boolean;
  useTrimask: boolean;
  useVaeMask: boolean;
  stackMask: boolean;
  zeroOutMaskRegion: boolean;
  mattingMode: "solo" | "clean_bg";
  skipIfExists: boolean;
  validation: boolean;
  skipUnet: boolean;
  maskToVae: boolean;
  seed: number;
  device: "cuda" | "cpu";
  gpuMemoryMode:
    | "model_full_load"
    | "model_cpu_offload"
    | "model_cpu_offload_and_qfloat8"
    | "sequential_cpu_offload";
  ulyssesDegree: number;
  ringDegree: number;
  allowSkippingError: boolean;
};

function statusClass(status: RunRecord["status"]) {
  if (status === "completed") return "text-emerald-700";
  if (status === "failed") return "text-red-700";
  if (status === "running") return "text-blue-700";
  if (status === "cancelled") return "text-amber-700";
  return "text-slate-700";
}

function TextField({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
}) {
  return (
    <input
      className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
    />
  );
}

function NumberField({
  value,
  onChange,
  placeholder,
}: {
  value: number;
  onChange: (value: number) => void;
  placeholder: string;
}) {
  return (
    <input
      className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
      type="number"
      value={String(value)}
      onChange={(e) => onChange(Number(e.target.value))}
      placeholder={placeholder}
    />
  );
}

export function Dashboard() {
  const [workflow, setWorkflow] = useState<Workflow>("pass1_inference");
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [activeRunId, setActiveRunId] = useState<string>("");
  const [logs, setLogs] = useState<string>("");
  const [env, setEnv] = useState<EnvCheck | null>(null);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [showAdvancedPass1, setShowAdvancedPass1] = useState(false);
  const [presets, setPresets] = useState<PresetRecord[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState("");
  const [presetName, setPresetName] = useState("");

  const [pass1, setPass1] = useState<Pass1Form>({
    configPath: "config/quadmask_cogvideox.py",
    dataRootdir: "./sample",
    runSeqs: "lime",
    savePath: "./outputs",
    modelName: "./CogVideoX-Fun-V1.5-5b-InP",
    transformerPath: "./void_pass1.safetensors",
    vaePath: "",
    loraPath: "",
    sampleSize: "384x672",
    dilateWidth: 11,
    maxVideoLength: 197,
    fps: 12,
    temporalWindowSize: 85,
    temproalMultidiffusionStride: 16,
    samplerName: "DDIM_Origin",
    denoiseStrength: 1.0,
    guidanceScale: 1.0,
    numInferenceSteps: 50,
    negativePrompt:
      "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
    loraWeight: 0.55,
    useQuadmask: true,
    useTrimask: false,
    useVaeMask: true,
    stackMask: false,
    zeroOutMaskRegion: false,
    mattingMode: "solo",
    skipIfExists: true,
    validation: false,
    skipUnet: false,
    maskToVae: false,
    seed: 42,
    device: "cuda",
    gpuMemoryMode: "model_cpu_offload_and_qfloat8",
    ulyssesDegree: 1,
    ringDegree: 1,
    allowSkippingError: false,
  });

  const [pass2VideoNames, setPass2VideoNames] = useState("");
  const [pass2DataRoot, setPass2DataRoot] = useState("./sample");
  const [pass2Pass1Dir, setPass2Pass1Dir] = useState("./outputs");
  const [pass2OutputDir, setPass2OutputDir] = useState("./inference_with_warped_noise");
  const [pass2ModelCheckpoint, setPass2ModelCheckpoint] = useState("./void_pass2.safetensors");
  const [pass2ModelName, setPass2ModelName] = useState("./CogVideoX-Fun-V1.5-5b-InP");
  const [pass2Height, setPass2Height] = useState("384");
  const [pass2Width, setPass2Width] = useState("672");
  const [pass2GuidanceScale, setPass2GuidanceScale] = useState("6.0");
  const [pass2Steps, setPass2Steps] = useState("50");

  const [maskConfigPath, setMaskConfigPath] = useState("my_config_points.json");
  const [maskSam2Checkpoint, setMaskSam2Checkpoint] = useState("../sam2_hiera_large.pt");
  const [maskDevice, setMaskDevice] = useState<"cuda" | "cpu">("cuda");
  const [maskExecutionMode, setMaskExecutionMode] = useState<
    "full" | "stage1" | "stage2" | "stage3" | "stage4"
  >("full");
  const [maskVlmModel, setMaskVlmModel] = useState("gemini-3-pro-preview");
  const [maskSegmentationModel, setMaskSegmentationModel] = useState<"langsam" | "sam3">("sam3");

  const activeRun = useMemo(() => runs.find((r) => r.id === activeRunId) ?? null, [runs, activeRunId]);

  const refreshRuns = useCallback(async () => {
    const data = await listRuns();
    setRuns(data);
    if (!activeRunId && data.length > 0) {
      setActiveRunId(data[0].id);
    }
  }, [activeRunId]);

  const refreshEnv = useCallback(async () => {
    setEnv(await getEnvCheck());
  }, []);

  const refreshLogs = useCallback(async (runId: string) => {
    const data = await getRunLogs(runId);
    setLogs(data.text);
  }, []);

  const refreshPresets = useCallback(async () => {
    setPresets(await listPresets());
  }, []);

  useEffect(() => {
    void refreshEnv().catch((e: Error) => setError(e.message));
    void refreshRuns().catch((e: Error) => setError(e.message));
    void refreshPresets().catch((e: Error) => setError(e.message));
  }, [refreshEnv, refreshRuns, refreshPresets]);

  useEffect(() => {
    const id = window.setInterval(() => {
      void refreshRuns().catch(() => {});
      if (activeRunId) {
        void refreshLogs(activeRunId).catch(() => {});
      }
    }, 2500);
    return () => window.clearInterval(id);
  }, [activeRunId, refreshLogs, refreshRuns]);

  const currentWorkflowParams = useCallback((): Record<string, unknown> => {
    if (workflow === "pass1_inference") {
      return {
        config_path: pass1.configPath,
        config_overrides: {
          "config.data.data_rootdir": pass1.dataRootdir,
          "config.experiment.run_seqs": pass1.runSeqs,
          "config.experiment.save_path": pass1.savePath,
          "config.video_model.model_name": pass1.modelName,
          "config.video_model.transformer_path": pass1.transformerPath,
          "config.video_model.vae_path": pass1.vaePath,
          "config.video_model.lora_path": pass1.loraPath,
          "config.data.sample_size": pass1.sampleSize,
          "config.data.dilate_width": pass1.dilateWidth,
          "config.data.max_video_length": pass1.maxVideoLength,
          "config.data.fps": pass1.fps,
          "config.video_model.temporal_window_size": pass1.temporalWindowSize,
          "config.video_model.temproal_multidiffusion_stride": pass1.temproalMultidiffusionStride,
          "config.video_model.sampler_name": pass1.samplerName,
          "config.video_model.denoise_strength": pass1.denoiseStrength,
          "config.video_model.guidance_scale": pass1.guidanceScale,
          "config.video_model.num_inference_steps": pass1.numInferenceSteps,
          "config.video_model.negative_prompt": pass1.negativePrompt,
          "config.video_model.lora_weight": pass1.loraWeight,
          "config.video_model.use_quadmask": pass1.useQuadmask,
          "config.video_model.use_trimask": pass1.useTrimask,
          "config.video_model.use_vae_mask": pass1.useVaeMask,
          "config.video_model.stack_mask": pass1.stackMask,
          "config.video_model.zero_out_mask_region": pass1.zeroOutMaskRegion,
          "config.experiment.matting_mode": pass1.mattingMode,
          "config.experiment.skip_if_exists": pass1.skipIfExists,
          "config.experiment.validation": pass1.validation,
          "config.experiment.skip_unet": pass1.skipUnet,
          "config.experiment.mask_to_vae": pass1.maskToVae,
          "config.system.seed": pass1.seed,
          "config.system.device": pass1.device,
          "config.system.gpu_memory_mode": pass1.gpuMemoryMode,
          "config.system.ulysses_degree": pass1.ulyssesDegree,
          "config.system.ring_degree": pass1.ringDegree,
          "config.system.allow_skipping_error": pass1.allowSkippingError,
        },
      };
    }

    if (workflow === "pass2_refine") {
      return {
        video_names: pass2VideoNames
          .split(",")
          .map((v) => v.trim())
          .filter(Boolean),
        data_rootdir: pass2DataRoot,
        pass1_dir: pass2Pass1Dir,
        output_dir: pass2OutputDir,
        model_name: pass2ModelName,
        model_checkpoint: pass2ModelCheckpoint,
        height: Number(pass2Height),
        width: Number(pass2Width),
        guidance_scale: Number(pass2GuidanceScale),
        num_inference_steps: Number(pass2Steps),
      };
    }

    return {
      config_points_json: maskConfigPath,
      execution_mode: maskExecutionMode,
      sam2_checkpoint: maskSam2Checkpoint,
      device: maskDevice,
      model: maskVlmModel,
      segmentation_model: maskSegmentationModel,
    };
  }, [
    workflow,
    pass1,
    pass2VideoNames,
    pass2DataRoot,
    pass2Pass1Dir,
    pass2OutputDir,
    pass2ModelName,
    pass2ModelCheckpoint,
    pass2Height,
    pass2Width,
    pass2GuidanceScale,
    pass2Steps,
    maskConfigPath,
    maskExecutionMode,
    maskSam2Checkpoint,
    maskDevice,
    maskVlmModel,
    maskSegmentationModel,
  ]);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      if (workflow === "pass1_inference") {
        const run = await createRun({
          workflow,
          params: {
            config_path: pass1.configPath,
            config_overrides: {
              "config.data.data_rootdir": pass1.dataRootdir,
              "config.experiment.run_seqs": pass1.runSeqs,
              "config.experiment.save_path": pass1.savePath,
              "config.video_model.model_name": pass1.modelName,
              "config.video_model.transformer_path": pass1.transformerPath,
              "config.video_model.vae_path": pass1.vaePath,
              "config.video_model.lora_path": pass1.loraPath,
              "config.data.sample_size": pass1.sampleSize,
              "config.data.dilate_width": pass1.dilateWidth,
              "config.data.max_video_length": pass1.maxVideoLength,
              "config.data.fps": pass1.fps,
              "config.video_model.temporal_window_size": pass1.temporalWindowSize,
              "config.video_model.temproal_multidiffusion_stride":
                pass1.temproalMultidiffusionStride,
              "config.video_model.sampler_name": pass1.samplerName,
              "config.video_model.denoise_strength": pass1.denoiseStrength,
              "config.video_model.guidance_scale": pass1.guidanceScale,
              "config.video_model.num_inference_steps": pass1.numInferenceSteps,
              "config.video_model.negative_prompt": pass1.negativePrompt,
              "config.video_model.lora_weight": pass1.loraWeight,
              "config.video_model.use_quadmask": pass1.useQuadmask,
              "config.video_model.use_trimask": pass1.useTrimask,
              "config.video_model.use_vae_mask": pass1.useVaeMask,
              "config.video_model.stack_mask": pass1.stackMask,
              "config.video_model.zero_out_mask_region": pass1.zeroOutMaskRegion,
              "config.experiment.matting_mode": pass1.mattingMode,
              "config.experiment.skip_if_exists": pass1.skipIfExists,
              "config.experiment.validation": pass1.validation,
              "config.experiment.skip_unet": pass1.skipUnet,
              "config.experiment.mask_to_vae": pass1.maskToVae,
              "config.system.seed": pass1.seed,
              "config.system.device": pass1.device,
              "config.system.gpu_memory_mode": pass1.gpuMemoryMode,
              "config.system.ulysses_degree": pass1.ulyssesDegree,
              "config.system.ring_degree": pass1.ringDegree,
              "config.system.allow_skipping_error": pass1.allowSkippingError,
            },
          },
        });
        setActiveRunId(run.id);
      }

      if (workflow === "pass2_refine") {
        const run = await createRun({
          workflow,
          params: {
            video_names: pass2VideoNames
              .split(",")
              .map((v) => v.trim())
              .filter(Boolean),
            data_rootdir: pass2DataRoot,
            pass1_dir: pass2Pass1Dir,
            output_dir: pass2OutputDir,
            model_name: pass2ModelName,
            model_checkpoint: pass2ModelCheckpoint,
            height: Number(pass2Height),
            width: Number(pass2Width),
            guidance_scale: Number(pass2GuidanceScale),
            num_inference_steps: Number(pass2Steps),
          },
        });
        setActiveRunId(run.id);
      }

      if (workflow === "mask_pipeline") {
        const maskWorkflowMap = {
          full: "mask_pipeline",
          stage1: "mask_stage1_sam2",
          stage2: "mask_stage2_vlm",
          stage3: "mask_stage3_grey",
          stage4: "mask_stage4_combine",
        } as const;

        const selectedWorkflow = maskWorkflowMap[maskExecutionMode];
        const maskParamsBase = {
          config_points_json: maskConfigPath,
        };

        let params: Record<string, unknown> = maskParamsBase;
        if (selectedWorkflow === "mask_pipeline" || selectedWorkflow === "mask_stage1_sam2") {
          params = {
            ...maskParamsBase,
            sam2_checkpoint: maskSam2Checkpoint,
            device: maskDevice,
          };
        } else if (selectedWorkflow === "mask_stage2_vlm") {
          params = {
            ...maskParamsBase,
            model: maskVlmModel,
          };
        } else if (selectedWorkflow === "mask_stage3_grey") {
          params = {
            ...maskParamsBase,
            segmentation_model: maskSegmentationModel,
          };
        }

        const run = await createRun({
          workflow: selectedWorkflow,
          params,
        });
        setActiveRunId(run.id);
      }

      await refreshRuns();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Run creation failed");
    } finally {
      setLoading(false);
    }
  };

  const applySelectedPreset = () => {
    const preset = presets.find((p) => p.id === selectedPresetId);
    if (!preset) return;

    const presetWorkflow = preset.workflow as Workflow;
    setWorkflow(presetWorkflow);

    if (presetWorkflow === "pass1_inference") {
      const params = preset.params as {
        config_path?: string;
        config_overrides?: Record<string, unknown>;
      };
      const o = params.config_overrides ?? {};
      setPass1((p) => ({
        ...p,
        configPath: String(params.config_path ?? p.configPath),
        dataRootdir: String(o["config.data.data_rootdir"] ?? p.dataRootdir),
        runSeqs: String(o["config.experiment.run_seqs"] ?? p.runSeqs),
        savePath: String(o["config.experiment.save_path"] ?? p.savePath),
        modelName: String(o["config.video_model.model_name"] ?? p.modelName),
        transformerPath: String(o["config.video_model.transformer_path"] ?? p.transformerPath),
        vaePath: String(o["config.video_model.vae_path"] ?? p.vaePath),
        loraPath: String(o["config.video_model.lora_path"] ?? p.loraPath),
        sampleSize: String(o["config.data.sample_size"] ?? p.sampleSize),
        guidanceScale: Number(o["config.video_model.guidance_scale"] ?? p.guidanceScale),
        numInferenceSteps: Number(
          o["config.video_model.num_inference_steps"] ?? p.numInferenceSteps,
        ),
      }));
      return;
    }

    if (presetWorkflow === "pass2_refine") {
      const params = preset.params as Record<string, unknown>;
      setPass2VideoNames(((params.video_names as string[]) ?? []).join(","));
      setPass2DataRoot(String(params.data_rootdir ?? pass2DataRoot));
      setPass2Pass1Dir(String(params.pass1_dir ?? pass2Pass1Dir));
      setPass2OutputDir(String(params.output_dir ?? pass2OutputDir));
      setPass2ModelName(String(params.model_name ?? pass2ModelName));
      setPass2ModelCheckpoint(String(params.model_checkpoint ?? pass2ModelCheckpoint));
      setPass2Height(String(params.height ?? pass2Height));
      setPass2Width(String(params.width ?? pass2Width));
      setPass2GuidanceScale(String(params.guidance_scale ?? pass2GuidanceScale));
      setPass2Steps(String(params.num_inference_steps ?? pass2Steps));
      return;
    }

    const params = preset.params as Record<string, unknown>;
    setMaskConfigPath(String(params.config_points_json ?? maskConfigPath));
    setMaskExecutionMode(
      String(params.execution_mode ?? maskExecutionMode) as
        | "full"
        | "stage1"
        | "stage2"
        | "stage3"
        | "stage4",
    );
    setMaskSam2Checkpoint(String(params.sam2_checkpoint ?? maskSam2Checkpoint));
    setMaskDevice(String(params.device ?? maskDevice) as "cuda" | "cpu");
    setMaskVlmModel(String(params.model ?? maskVlmModel));
    setMaskSegmentationModel(String(params.segmentation_model ?? maskSegmentationModel) as "langsam" | "sam3");
  };

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6 px-4 py-8">
      <header className="rounded-2xl bg-gradient-to-r from-cyan-900 via-sky-800 to-indigo-900 p-6 text-white shadow-lg">
        <p className="text-xs uppercase tracking-[0.2em] text-cyan-100">VOID Operator</p>
        <h1 className="mt-2 text-3xl font-semibold">React + FastAPI Control Panel</h1>
        <p className="mt-1 text-sm text-cyan-100">
          Launch mask generation and inference workflows with reproducible config parameters.
        </p>
      </header>

      {error ? <p className="rounded-lg bg-red-50 p-3 text-sm text-red-800">{error}</p> : null}

      <section className={sectionClass}>
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-medium text-slate-900">Environment Check</h2>
          <button
            type="button"
            onClick={() => void refreshEnv().catch((e: Error) => setError(e.message))}
            className="rounded-md border border-slate-300 px-3 py-1 text-sm text-slate-700 hover:bg-slate-50"
          >
            Refresh
          </button>
        </div>
        {env ? (
          <div className="mt-3 grid gap-2 text-sm text-slate-700 md:grid-cols-2">
            <p>
              <span className="font-semibold">Python:</span> {env.python ?? "missing"}
            </p>
            <p>
              <span className="font-semibold">ffmpeg:</span> {env.ffmpeg ?? "missing"}
            </p>
            <p>
              <span className="font-semibold">nvidia-smi:</span> {env.nvidia_smi ?? "missing"}
            </p>
            <p>
              <span className="font-semibold">CUDA:</span>{" "}
              {env.cuda.available === null ? "unknown" : env.cuda.available ? "available" : "unavailable"}
            </p>
            <p>
              <span className="font-semibold">GEMINI_API_KEY:</span>{" "}
              {env.gemini_api_key_set ? "set" : "not set"}
            </p>
            <p className="md:col-span-2">
              <span className="font-semibold">Project root:</span> {env.project_root}
            </p>
          </div>
        ) : (
          <p className="mt-3 text-sm text-slate-600">Loading...</p>
        )}
      </section>

      <section className={sectionClass}>
        <h2 className="text-lg font-medium text-slate-900">Presets</h2>
        <div className="mt-3 grid gap-2 md:grid-cols-[1fr_auto]">
          <input
            className="rounded-md border border-slate-300 px-3 py-2 text-sm"
            placeholder="Preset name"
            value={presetName}
            onChange={(e) => setPresetName(e.target.value)}
          />
          <button
            type="button"
            className="rounded-md bg-slate-900 px-3 py-2 text-sm font-medium text-white hover:bg-slate-800"
            onClick={() =>
              void createPreset({
                name: presetName || `${workflow}-${new Date().toISOString()}`,
                workflow,
                params: currentWorkflowParams(),
              })
                .then(async () => {
                  setPresetName("");
                  await refreshPresets();
                })
                .catch((e: Error) => setError(e.message))
            }
          >
            Save Current As Preset
          </button>
        </div>

        <div className="mt-2 grid gap-2 md:grid-cols-[1fr_auto]">
          <select
            className="rounded-md border border-slate-300 px-3 py-2 text-sm"
            value={selectedPresetId}
            onChange={(e) => setSelectedPresetId(e.target.value)}
          >
            <option value="">Select preset to load</option>
            {presets.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} ({p.workflow})
              </option>
            ))}
          </select>
          <button
            type="button"
            className="rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-700 hover:bg-slate-50"
            onClick={applySelectedPreset}
            disabled={!selectedPresetId}
          >
            Load Preset
          </button>
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className={sectionClass}>
          <h2 className="text-lg font-medium text-slate-900">Launch Workflow</h2>
          <form onSubmit={onSubmit} className="mt-4 space-y-4">
            <label className="block text-sm text-slate-700">
              Workflow
              <select
                value={workflow}
                onChange={(e) => setWorkflow(e.target.value as Workflow)}
                className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2"
              >
                <option value="pass1_inference">Pass 1 Inference</option>
                <option value="pass2_refine">Pass 2 Refinement</option>
                <option value="mask_pipeline">Mask Pipeline</option>
              </select>
            </label>

            {workflow === "pass1_inference" ? (
              <div className="space-y-3">
                <div className="grid gap-2">
                  <TextField
                    value={pass1.configPath}
                    onChange={(value) => setPass1((p) => ({ ...p, configPath: value }))}
                    placeholder="config path"
                  />
                  <TextField
                    value={pass1.dataRootdir}
                    onChange={(value) => setPass1((p) => ({ ...p, dataRootdir: value }))}
                    placeholder="data root"
                  />
                  <TextField
                    value={pass1.runSeqs}
                    onChange={(value) => setPass1((p) => ({ ...p, runSeqs: value }))}
                    placeholder="run seqs (comma-separated)"
                  />
                  <TextField
                    value={pass1.savePath}
                    onChange={(value) => setPass1((p) => ({ ...p, savePath: value }))}
                    placeholder="save path"
                  />
                  <TextField
                    value={pass1.modelName}
                    onChange={(value) => setPass1((p) => ({ ...p, modelName: value }))}
                    placeholder="base model path"
                  />
                  <TextField
                    value={pass1.transformerPath}
                    onChange={(value) => setPass1((p) => ({ ...p, transformerPath: value }))}
                    placeholder="pass1 checkpoint path"
                  />
                  <div className="grid grid-cols-3 gap-2">
                    <TextField
                      value={pass1.sampleSize}
                      onChange={(value) => setPass1((p) => ({ ...p, sampleSize: value }))}
                      placeholder="sample_size"
                    />
                    <NumberField
                      value={pass1.guidanceScale}
                      onChange={(value) => setPass1((p) => ({ ...p, guidanceScale: value }))}
                      placeholder="guidance"
                    />
                    <NumberField
                      value={pass1.numInferenceSteps}
                      onChange={(value) => setPass1((p) => ({ ...p, numInferenceSteps: value }))}
                      placeholder="steps"
                    />
                  </div>
                </div>

                <button
                  type="button"
                  className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-700 hover:bg-slate-50"
                  onClick={() => setShowAdvancedPass1((v) => !v)}
                >
                  {showAdvancedPass1 ? "Hide Advanced Pass 1 Controls" : "Show Advanced Pass 1 Controls"}
                </button>

                {showAdvancedPass1 ? (
                  <div className="space-y-3 rounded-md border border-slate-200 bg-slate-50 p-3">
                    <div className="grid grid-cols-2 gap-2">
                      <TextField
                        value={pass1.vaePath}
                        onChange={(value) => setPass1((p) => ({ ...p, vaePath: value }))}
                        placeholder="vae path"
                      />
                      <TextField
                        value={pass1.loraPath}
                        onChange={(value) => setPass1((p) => ({ ...p, loraPath: value }))}
                        placeholder="lora path"
                      />
                    </div>

                    <div className="grid grid-cols-4 gap-2">
                      <NumberField
                        value={pass1.dilateWidth}
                        onChange={(value) => setPass1((p) => ({ ...p, dilateWidth: value }))}
                        placeholder="dilate width"
                      />
                      <NumberField
                        value={pass1.maxVideoLength}
                        onChange={(value) => setPass1((p) => ({ ...p, maxVideoLength: value }))}
                        placeholder="max frames"
                      />
                      <NumberField
                        value={pass1.fps}
                        onChange={(value) => setPass1((p) => ({ ...p, fps: value }))}
                        placeholder="fps"
                      />
                      <NumberField
                        value={pass1.temporalWindowSize}
                        onChange={(value) => setPass1((p) => ({ ...p, temporalWindowSize: value }))}
                        placeholder="temporal window"
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      <label className="text-xs text-slate-600">
                        sampler
                        <select
                          className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                          value={pass1.samplerName}
                          onChange={(e) =>
                            setPass1((p) => ({
                              ...p,
                              samplerName: e.target.value as Pass1Form["samplerName"],
                            }))
                          }
                        >
                          <option value="DDIM_Origin">DDIM_Origin</option>
                          <option value="DDIM_Cog">DDIM_Cog</option>
                          <option value="Euler">Euler</option>
                          <option value="Euler A">Euler A</option>
                          <option value="DPM++">DPM++</option>
                          <option value="PNDM">PNDM</option>
                        </select>
                      </label>
                      <label className="text-xs text-slate-600">
                        gpu memory mode
                        <select
                          className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                          value={pass1.gpuMemoryMode}
                          onChange={(e) =>
                            setPass1((p) => ({
                              ...p,
                              gpuMemoryMode: e.target.value as Pass1Form["gpuMemoryMode"],
                            }))
                          }
                        >
                          <option value="model_full_load">model_full_load</option>
                          <option value="model_cpu_offload">model_cpu_offload</option>
                          <option value="model_cpu_offload_and_qfloat8">
                            model_cpu_offload_and_qfloat8
                          </option>
                          <option value="sequential_cpu_offload">sequential_cpu_offload</option>
                        </select>
                      </label>
                    </div>

                    <div className="grid grid-cols-4 gap-2">
                      <NumberField
                        value={pass1.denoiseStrength}
                        onChange={(value) => setPass1((p) => ({ ...p, denoiseStrength: value }))}
                        placeholder="denoise strength"
                      />
                      <NumberField
                        value={pass1.loraWeight}
                        onChange={(value) => setPass1((p) => ({ ...p, loraWeight: value }))}
                        placeholder="lora weight"
                      />
                      <NumberField
                        value={pass1.temproalMultidiffusionStride}
                        onChange={(value) =>
                          setPass1((p) => ({ ...p, temproalMultidiffusionStride: value }))
                        }
                        placeholder="multidiff stride"
                      />
                      <NumberField
                        value={pass1.seed}
                        onChange={(value) => setPass1((p) => ({ ...p, seed: value }))}
                        placeholder="seed"
                      />
                    </div>

                    <TextField
                      value={pass1.negativePrompt}
                      onChange={(value) => setPass1((p) => ({ ...p, negativePrompt: value }))}
                      placeholder="negative prompt"
                    />

                    <div className="grid grid-cols-2 gap-2">
                      <label className="flex items-center gap-2 rounded-md border border-slate-300 bg-white p-2 text-xs text-slate-700">
                        <input
                          type="checkbox"
                          checked={pass1.useQuadmask}
                          onChange={(e) => setPass1((p) => ({ ...p, useQuadmask: e.target.checked }))}
                        />
                        use_quadmask
                      </label>
                      <label className="flex items-center gap-2 rounded-md border border-slate-300 bg-white p-2 text-xs text-slate-700">
                        <input
                          type="checkbox"
                          checked={pass1.useTrimask}
                          onChange={(e) => setPass1((p) => ({ ...p, useTrimask: e.target.checked }))}
                        />
                        use_trimask
                      </label>
                      <label className="flex items-center gap-2 rounded-md border border-slate-300 bg-white p-2 text-xs text-slate-700">
                        <input
                          type="checkbox"
                          checked={pass1.useVaeMask}
                          onChange={(e) => setPass1((p) => ({ ...p, useVaeMask: e.target.checked }))}
                        />
                        use_vae_mask
                      </label>
                      <label className="flex items-center gap-2 rounded-md border border-slate-300 bg-white p-2 text-xs text-slate-700">
                        <input
                          type="checkbox"
                          checked={pass1.stackMask}
                          onChange={(e) => setPass1((p) => ({ ...p, stackMask: e.target.checked }))}
                        />
                        stack_mask
                      </label>
                      <label className="flex items-center gap-2 rounded-md border border-slate-300 bg-white p-2 text-xs text-slate-700">
                        <input
                          type="checkbox"
                          checked={pass1.zeroOutMaskRegion}
                          onChange={(e) =>
                            setPass1((p) => ({ ...p, zeroOutMaskRegion: e.target.checked }))
                          }
                        />
                        zero_out_mask_region
                      </label>
                      <label className="flex items-center gap-2 rounded-md border border-slate-300 bg-white p-2 text-xs text-slate-700">
                        <input
                          type="checkbox"
                          checked={pass1.skipIfExists}
                          onChange={(e) => setPass1((p) => ({ ...p, skipIfExists: e.target.checked }))}
                        />
                        skip_if_exists
                      </label>
                      <label className="flex items-center gap-2 rounded-md border border-slate-300 bg-white p-2 text-xs text-slate-700">
                        <input
                          type="checkbox"
                          checked={pass1.validation}
                          onChange={(e) => setPass1((p) => ({ ...p, validation: e.target.checked }))}
                        />
                        validation
                      </label>
                      <label className="flex items-center gap-2 rounded-md border border-slate-300 bg-white p-2 text-xs text-slate-700">
                        <input
                          type="checkbox"
                          checked={pass1.skipUnet}
                          onChange={(e) => setPass1((p) => ({ ...p, skipUnet: e.target.checked }))}
                        />
                        skip_unet
                      </label>
                      <label className="flex items-center gap-2 rounded-md border border-slate-300 bg-white p-2 text-xs text-slate-700">
                        <input
                          type="checkbox"
                          checked={pass1.maskToVae}
                          onChange={(e) => setPass1((p) => ({ ...p, maskToVae: e.target.checked }))}
                        />
                        mask_to_vae
                      </label>
                      <label className="flex items-center gap-2 rounded-md border border-slate-300 bg-white p-2 text-xs text-slate-700">
                        <input
                          type="checkbox"
                          checked={pass1.allowSkippingError}
                          onChange={(e) =>
                            setPass1((p) => ({ ...p, allowSkippingError: e.target.checked }))
                          }
                        />
                        allow_skipping_error
                      </label>
                    </div>

                    <div className="grid grid-cols-4 gap-2">
                      <label className="text-xs text-slate-600">
                        device
                        <select
                          className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                          value={pass1.device}
                          onChange={(e) =>
                            setPass1((p) => ({
                              ...p,
                              device: e.target.value as Pass1Form["device"],
                            }))
                          }
                        >
                          <option value="cuda">cuda</option>
                          <option value="cpu">cpu</option>
                        </select>
                      </label>
                      <label className="text-xs text-slate-600">
                        matting mode
                        <select
                          className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                          value={pass1.mattingMode}
                          onChange={(e) =>
                            setPass1((p) => ({
                              ...p,
                              mattingMode: e.target.value as Pass1Form["mattingMode"],
                            }))
                          }
                        >
                          <option value="solo">solo</option>
                          <option value="clean_bg">clean_bg</option>
                        </select>
                      </label>
                      <NumberField
                        value={pass1.ulyssesDegree}
                        onChange={(value) => setPass1((p) => ({ ...p, ulyssesDegree: value }))}
                        placeholder="ulysses_degree"
                      />
                      <NumberField
                        value={pass1.ringDegree}
                        onChange={(value) => setPass1((p) => ({ ...p, ringDegree: value }))}
                        placeholder="ring_degree"
                      />
                    </div>
                  </div>
                ) : null}
              </div>
            ) : null}

            {workflow === "pass2_refine" ? (
              <div className="space-y-2">
                <input className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm" value={pass2VideoNames} onChange={(e) => setPass2VideoNames(e.target.value)} placeholder="video names (comma-separated)" />
                <input className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm" value={pass2DataRoot} onChange={(e) => setPass2DataRoot(e.target.value)} placeholder="data root" />
                <input className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm" value={pass2Pass1Dir} onChange={(e) => setPass2Pass1Dir(e.target.value)} placeholder="pass1 output dir" />
                <input className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm" value={pass2OutputDir} onChange={(e) => setPass2OutputDir(e.target.value)} placeholder="output dir" />
                <input className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm" value={pass2ModelName} onChange={(e) => setPass2ModelName(e.target.value)} placeholder="base model path" />
                <input className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm" value={pass2ModelCheckpoint} onChange={(e) => setPass2ModelCheckpoint(e.target.value)} placeholder="pass2 checkpoint" />
                <div className="grid grid-cols-4 gap-2">
                  <input className="rounded-md border border-slate-300 px-3 py-2 text-sm" value={pass2Height} onChange={(e) => setPass2Height(e.target.value)} placeholder="height" />
                  <input className="rounded-md border border-slate-300 px-3 py-2 text-sm" value={pass2Width} onChange={(e) => setPass2Width(e.target.value)} placeholder="width" />
                  <input className="rounded-md border border-slate-300 px-3 py-2 text-sm" value={pass2GuidanceScale} onChange={(e) => setPass2GuidanceScale(e.target.value)} placeholder="guidance" />
                  <input className="rounded-md border border-slate-300 px-3 py-2 text-sm" value={pass2Steps} onChange={(e) => setPass2Steps(e.target.value)} placeholder="steps" />
                </div>
              </div>
            ) : null}

            {workflow === "mask_pipeline" ? (
              <div className="space-y-2">
                <TextField
                  value={maskConfigPath}
                  onChange={setMaskConfigPath}
                  placeholder="config_points json"
                />
                <select
                  className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                  value={maskExecutionMode}
                  onChange={(e) =>
                    setMaskExecutionMode(
                      e.target.value as "full" | "stage1" | "stage2" | "stage3" | "stage4",
                    )
                  }
                >
                  <option value="full">Full pipeline (stage1-4)</option>
                  <option value="stage1">Stage 1 only (SAM2)</option>
                  <option value="stage2">Stage 2 only (VLM)</option>
                  <option value="stage3">Stage 3 only (Grey masks)</option>
                  <option value="stage4">Stage 4 only (Combine quadmask)</option>
                </select>

                {maskExecutionMode === "full" || maskExecutionMode === "stage1" ? (
                  <>
                    <TextField
                      value={maskSam2Checkpoint}
                      onChange={setMaskSam2Checkpoint}
                      placeholder="sam2 checkpoint"
                    />
                    <select
                      className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                      value={maskDevice}
                      onChange={(e) => setMaskDevice(e.target.value as "cuda" | "cpu")}
                    >
                      <option value="cuda">cuda</option>
                      <option value="cpu">cpu</option>
                    </select>
                  </>
                ) : null}

                {maskExecutionMode === "stage2" ? (
                  <TextField
                    value={maskVlmModel}
                    onChange={setMaskVlmModel}
                    placeholder="vlm model (example: gemini-3-pro-preview)"
                  />
                ) : null}

                {maskExecutionMode === "stage3" ? (
                  <select
                    className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
                    value={maskSegmentationModel}
                    onChange={(e) => setMaskSegmentationModel(e.target.value as "langsam" | "sam3")}
                  >
                    <option value="sam3">sam3</option>
                    <option value="langsam">langsam</option>
                  </select>
                ) : null}

                <div className="grid grid-cols-2 gap-2 pt-1">
                  <button
                    type="button"
                    className="rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-700 hover:bg-slate-50"
                    onClick={() =>
                      void createRun({
                        workflow: "point_selector_gui",
                        params: { config_points_json: maskConfigPath },
                      })
                        .then((run) => {
                          setActiveRunId(run.id);
                          return refreshRuns();
                        })
                        .catch((e: Error) => setError(e.message))
                    }
                  >
                    Launch Point Selector GUI
                  </button>
                  <button
                    type="button"
                    className="rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-700 hover:bg-slate-50"
                    onClick={() =>
                      void createRun({
                        workflow: "edit_quadmask_gui",
                        params: {},
                      })
                        .then((run) => {
                          setActiveRunId(run.id);
                          return refreshRuns();
                        })
                        .catch((e: Error) => setError(e.message))
                    }
                  >
                    Launch Mask Editor GUI
                  </button>
                </div>
              </div>
            ) : null}

            <button
              disabled={loading}
              className="w-full rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-60"
            >
              {loading ? "Launching..." : "Launch Run"}
            </button>
          </form>
        </section>

        <section className={sectionClass}>
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-medium text-slate-900">Runs</h2>
            <button
              type="button"
              onClick={() => void refreshRuns().catch((e: Error) => setError(e.message))}
              className="rounded-md border border-slate-300 px-3 py-1 text-sm text-slate-700 hover:bg-slate-50"
            >
              Refresh
            </button>
          </div>
          <div className="mt-3 max-h-[420px] space-y-2 overflow-y-auto pr-1">
            {runs.map((run) => (
              <button
                key={run.id}
                type="button"
                onClick={() => {
                  setActiveRunId(run.id);
                  void refreshLogs(run.id).catch(() => {});
                }}
                className={`w-full rounded-md border p-3 text-left ${
                  activeRunId === run.id ? "border-sky-400 bg-sky-50" : "border-slate-200 bg-white"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm font-medium text-slate-900">{run.workflow}</span>
                  <span className={`text-xs font-semibold uppercase ${statusClass(run.status)}`}>{run.status}</span>
                </div>
                <p className="mt-1 text-xs text-slate-600">id: {run.id}</p>
                <p className="text-xs text-slate-500">created: {new Date(run.created_at).toLocaleString()}</p>
              </button>
            ))}
            {runs.length === 0 ? <p className="text-sm text-slate-600">No runs yet.</p> : null}
          </div>
        </section>
      </div>

      <section className={sectionClass}>
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-lg font-medium text-slate-900">Run Details</h2>
          {activeRun?.status === "running" ? (
            <button
              type="button"
              onClick={() =>
                void cancelRun(activeRun.id)
                  .then(() => refreshRuns())
                  .catch((e: Error) => setError(e.message))
              }
              className="rounded-md border border-red-300 px-3 py-1 text-sm text-red-700 hover:bg-red-50"
            >
              Cancel Run
            </button>
          ) : null}
        </div>

        {activeRun ? (
          <div className="mt-3 space-y-3 text-sm text-slate-700">
            <p>
              <span className="font-semibold">Run:</span> {activeRun.id}
            </p>
            <p>
              <span className="font-semibold">Command:</span> {activeRun.command.join(" ")}
            </p>
            <p>
              <span className="font-semibold">Status:</span>{" "}
              <span className={statusClass(activeRun.status)}>{activeRun.status}</span>
            </p>
            {activeRun.error ? (
              <p className="rounded-md bg-red-50 p-2 text-red-700">{activeRun.error}</p>
            ) : null}
            <pre className="max-h-[420px] overflow-auto rounded-md bg-slate-950 p-4 text-xs text-slate-100">
              {logs || "No logs yet."}
            </pre>
          </div>
        ) : (
          <p className="mt-3 text-sm text-slate-600">Select a run to inspect logs.</p>
        )}
      </section>
    </div>
  );
}
