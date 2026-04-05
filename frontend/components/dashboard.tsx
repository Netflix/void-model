"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { Pass1FormPanel } from "@/components/dashboard/pass1-form";
import { Pass2FormPanel } from "@/components/dashboard/pass2-form";
import { TextField } from "@/components/dashboard/fields";
import type { MaskExecutionMode, Pass1Form } from "@/components/dashboard/types";

import {
  cancelRun,
  clearCache,
  createPreset,
  createRun,
  getCacheInfo,
  getEnvCheck,
  getRunLogs,
  listDataSequences,
  listArtifacts,
  listPresets,
  listRuns,
  updatePromptBg,
  validateConfig,
} from "@/lib/api";
import type {
  ArtifactFile,
  CacheInfo,
  DataSequence,
  EnvCheck,
  PresetRecord,
  RunRecord,
  WorkflowConfig,
  Workflow,
} from "@/lib/types";

const sectionClass =
  "rounded-xl border border-zinc-800 bg-zinc-950/90 p-4 shadow-sm backdrop-blur-sm";

function statusClass(status: RunRecord["status"]) {
  if (status === "completed") return "text-emerald-400";
  if (status === "failed") return "text-red-400";
  if (status === "running") return "text-blue-300";
  if (status === "cancelled") return "text-amber-300";
  return "text-zinc-300";
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
  const [artifacts, setArtifacts] = useState<ArtifactFile[]>([]);
  const [validationWarnings, setValidationWarnings] = useState<string[]>([]);
  const [sequenceRoot, setSequenceRoot] = useState("./sample");
  const [sequences, setSequences] = useState<DataSequence[]>([]);
  const [promptDrafts, setPromptDrafts] = useState<Record<string, string>>({});
  const [cachePath, setCachePath] = useState("./pass1_warped_noise_cache");
  const [cacheInfo, setCacheInfo] = useState<CacheInfo | null>(null);

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
  const [pass2MaxVideoLength, setPass2MaxVideoLength] = useState("197");
  const [pass2TemporalWindowSize, setPass2TemporalWindowSize] = useState("85");
  const [pass2Seed, setPass2Seed] = useState("42");
  const [pass2GuidanceScale, setPass2GuidanceScale] = useState("6.0");
  const [pass2Steps, setPass2Steps] = useState("50");
  const [pass2WarpedNoiseCacheDir, setPass2WarpedNoiseCacheDir] = useState("./pass1_warped_noise_cache");
  const [pass2SkipNoiseGeneration, setPass2SkipNoiseGeneration] = useState(false);
  const [pass2UseQuadmask, setPass2UseQuadmask] = useState(true);

  const [maskConfigPath, setMaskConfigPath] = useState("my_config_points.json");
  const [maskSam2Checkpoint, setMaskSam2Checkpoint] = useState("../sam2_hiera_large.pt");
  const [maskDevice, setMaskDevice] = useState<"cuda" | "cpu">("cuda");
  const [maskExecutionMode, setMaskExecutionMode] = useState<MaskExecutionMode>("full");
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

  const refreshArtifacts = useCallback(async (runId: string) => {
    setArtifacts(await listArtifacts(runId));
  }, []);

  const refreshCacheInfo = useCallback(async () => {
    setCacheInfo(await getCacheInfo(cachePath));
  }, [cachePath]);

  const refreshSequences = useCallback(async () => {
    const data = await listDataSequences(sequenceRoot);
    setSequences(data);
    setPromptDrafts((prev) => {
      const next = { ...prev };
      for (const seq of data) {
        if (next[seq.path] === undefined) {
          next[seq.path] = seq.prompt_bg ?? "";
        }
      }
      return next;
    });
  }, [sequenceRoot]);

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
        void refreshArtifacts(activeRunId).catch(() => {});
      }
    }, 2500);
    return () => window.clearInterval(id);
  }, [activeRunId, refreshArtifacts, refreshLogs, refreshRuns]);

  useEffect(() => {
    if (!activeRunId) {
      setArtifacts([]);
      return;
    }
    void refreshArtifacts(activeRunId).catch(() => {});
  }, [activeRunId, refreshArtifacts]);

  useEffect(() => {
    setSequenceRoot(pass1.dataRootdir);
  }, [pass1.dataRootdir]);

  useEffect(() => {
    setCachePath(pass2WarpedNoiseCacheDir);
  }, [pass2WarpedNoiseCacheDir]);

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
        max_video_length: Number(pass2MaxVideoLength),
        temporal_window_size: Number(pass2TemporalWindowSize),
        height: Number(pass2Height),
        width: Number(pass2Width),
        seed: Number(pass2Seed),
        guidance_scale: Number(pass2GuidanceScale),
        num_inference_steps: Number(pass2Steps),
        warped_noise_cache_dir: pass2WarpedNoiseCacheDir,
        skip_noise_generation: pass2SkipNoiseGeneration,
        use_quadmask: pass2UseQuadmask,
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
    pass2MaxVideoLength,
    pass2TemporalWindowSize,
    pass2Height,
    pass2Width,
    pass2Seed,
    pass2GuidanceScale,
    pass2Steps,
    pass2WarpedNoiseCacheDir,
    pass2SkipNoiseGeneration,
    pass2UseQuadmask,
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
      let submitWorkflow: Workflow = workflow;
      let submitParams: Record<string, unknown> = {};

      if (workflow === "pass1_inference") {
        submitWorkflow = workflow;
        submitParams = {
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
        };
      }

      if (workflow === "pass2_refine") {
        submitWorkflow = workflow;
        submitParams = {
          video_names: pass2VideoNames
            .split(",")
            .map((v) => v.trim())
            .filter(Boolean),
          data_rootdir: pass2DataRoot,
          pass1_dir: pass2Pass1Dir,
          output_dir: pass2OutputDir,
          model_name: pass2ModelName,
          model_checkpoint: pass2ModelCheckpoint,
          max_video_length: Number(pass2MaxVideoLength),
          temporal_window_size: Number(pass2TemporalWindowSize),
          height: Number(pass2Height),
          width: Number(pass2Width),
          seed: Number(pass2Seed),
          guidance_scale: Number(pass2GuidanceScale),
          num_inference_steps: Number(pass2Steps),
          warped_noise_cache_dir: pass2WarpedNoiseCacheDir,
          skip_noise_generation: pass2SkipNoiseGeneration,
          use_quadmask: pass2UseQuadmask,
        };
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

        submitWorkflow = selectedWorkflow;
        submitParams = params;
      }

      const validation = await validateConfig({
        workflow: submitWorkflow,
        params: submitParams,
      });
      setValidationWarnings(validation.warnings);
      if (!validation.valid) {
        setError(`Validation failed: ${validation.errors.join(" | ")}`);
        return;
      }

      const run = await createRun({
        workflow: submitWorkflow,
        params: submitParams,
      });
      setActiveRunId(run.id);

      await refreshRuns();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Run creation failed");
    } finally {
      setLoading(false);
    }
  };

  const applyWorkflowConfig = (targetWorkflow: Workflow, params: Record<string, unknown>) => {
    setWorkflow(targetWorkflow);

    if (targetWorkflow === "pass1_inference") {
      const p = params as { config_path?: string; config_overrides?: Record<string, unknown> };
      const o = p.config_overrides ?? {};
      setPass1((prev) => ({
        ...prev,
        configPath: String(p.config_path ?? prev.configPath),
        dataRootdir: String(o["config.data.data_rootdir"] ?? prev.dataRootdir),
        runSeqs: String(o["config.experiment.run_seqs"] ?? prev.runSeqs),
        savePath: String(o["config.experiment.save_path"] ?? prev.savePath),
        modelName: String(o["config.video_model.model_name"] ?? prev.modelName),
        transformerPath: String(o["config.video_model.transformer_path"] ?? prev.transformerPath),
        vaePath: String(o["config.video_model.vae_path"] ?? prev.vaePath),
        loraPath: String(o["config.video_model.lora_path"] ?? prev.loraPath),
        sampleSize: String(o["config.data.sample_size"] ?? prev.sampleSize),
        guidanceScale: Number(o["config.video_model.guidance_scale"] ?? prev.guidanceScale),
        numInferenceSteps: Number(o["config.video_model.num_inference_steps"] ?? prev.numInferenceSteps),
      }));
      return;
    }

    if (targetWorkflow === "pass2_refine") {
      setPass2VideoNames(((params.video_names as string[]) ?? []).join(","));
      setPass2DataRoot(String(params.data_rootdir ?? pass2DataRoot));
      setPass2Pass1Dir(String(params.pass1_dir ?? pass2Pass1Dir));
      setPass2OutputDir(String(params.output_dir ?? pass2OutputDir));
      setPass2ModelName(String(params.model_name ?? pass2ModelName));
      setPass2ModelCheckpoint(String(params.model_checkpoint ?? pass2ModelCheckpoint));
      setPass2MaxVideoLength(String(params.max_video_length ?? pass2MaxVideoLength));
      setPass2TemporalWindowSize(String(params.temporal_window_size ?? pass2TemporalWindowSize));
      setPass2Height(String(params.height ?? pass2Height));
      setPass2Width(String(params.width ?? pass2Width));
      setPass2Seed(String(params.seed ?? pass2Seed));
      setPass2GuidanceScale(String(params.guidance_scale ?? pass2GuidanceScale));
      setPass2Steps(String(params.num_inference_steps ?? pass2Steps));
      setPass2WarpedNoiseCacheDir(String(params.warped_noise_cache_dir ?? pass2WarpedNoiseCacheDir));
      setPass2SkipNoiseGeneration(Boolean(params.skip_noise_generation ?? pass2SkipNoiseGeneration));
      setPass2UseQuadmask(Boolean(params.use_quadmask ?? pass2UseQuadmask));
      return;
    }

    setMaskConfigPath(String(params.config_points_json ?? maskConfigPath));
    setMaskExecutionMode(String(params.execution_mode ?? maskExecutionMode) as MaskExecutionMode);
    setMaskSam2Checkpoint(String(params.sam2_checkpoint ?? maskSam2Checkpoint));
    setMaskDevice(String(params.device ?? maskDevice) as "cuda" | "cpu");
    setMaskVlmModel(String(params.model ?? maskVlmModel));
    setMaskSegmentationModel(
      String(params.segmentation_model ?? maskSegmentationModel) as "langsam" | "sam3",
    );
  };

  const applySelectedPreset = () => {
    const preset = presets.find((p) => p.id === selectedPresetId);
    if (!preset) return;
    applyWorkflowConfig(preset.workflow as Workflow, preset.params);
  };

  const exportConfigFile = (cfg: WorkflowConfig, name: string) => {
    const blob = new Blob([JSON.stringify(cfg, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${name}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const onImportConfigFile = async (file: File) => {
    try {
      const text = await file.text();
      const parsed = JSON.parse(text) as WorkflowConfig;
      if (!parsed.workflow || !parsed.params) {
        throw new Error("Invalid config file format");
      }
      applyWorkflowConfig(parsed.workflow, parsed.params);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to import config");
    }
  };

  const selectedSeqs = useMemo(
    () =>
      new Set(
        pass1.runSeqs
          .split(",")
          .map((v) => v.trim())
          .filter(Boolean),
      ),
    [pass1.runSeqs],
  );

  const toggleRunSequence = (name: string) => {
    const current = new Set(selectedSeqs);
    if (current.has(name)) {
      current.delete(name);
    } else {
      current.add(name);
    }
    setPass1((p) => ({ ...p, runSeqs: Array.from(current).join(",") }));
  };

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6 px-4 py-8">
      <header className="rounded-2xl border border-red-700/40 bg-gradient-to-r from-black via-zinc-950 to-red-950 p-6 text-white shadow-lg shadow-red-950/40">
        <p className="text-xs uppercase tracking-[0.2em] text-red-300">VOID Workflow Ops</p>
        <h1 className="mt-2 text-3xl font-semibold">VOID Production Bay</h1>
        <p className="mt-1 text-sm text-red-200">
          Launch, monitor, and rerun workflows with consistent configuration.
        </p>
      </header>

      {error ? <p className="rounded-lg border border-red-700 bg-red-950/60 p-3 text-sm text-red-200">{error}</p> : null}
      {validationWarnings.length > 0 ? (
        <p className="rounded-lg border border-amber-700 bg-amber-950/40 p-3 text-sm text-amber-200">
          Validation warnings: {validationWarnings.join(" | ")}
        </p>
      ) : null}

      <section className={sectionClass}>
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-medium text-zinc-100">Environment Check</h2>
          <button
            type="button"
            onClick={() => void refreshEnv().catch((e: Error) => setError(e.message))}
            className="rounded-md border border-zinc-700 px-3 py-1 text-sm text-zinc-300 hover:bg-zinc-900/60"
          >
            Refresh
          </button>
        </div>
        {env ? (
          <div className="mt-3 grid gap-2 text-sm text-zinc-300 md:grid-cols-2">
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
          <p className="mt-3 text-sm text-zinc-400">Loading...</p>
        )}
      </section>

      <section className={sectionClass}>
        <h2 className="text-lg font-medium text-zinc-100">Presets</h2>
        <div className="mt-3 grid gap-2 md:grid-cols-[1fr_auto]">
          <label className="block text-xs text-zinc-400">
            Preset Name
            <input
              className="mt-1 w-full rounded-md border border-zinc-700 px-3 py-2 text-sm"
              placeholder="Preset name"
              value={presetName}
              onChange={(e) => setPresetName(e.target.value)}
            />
          </label>
          <button
            type="button"
            className="rounded-md bg-red-700 px-3 py-2 text-sm font-medium text-white hover:bg-red-600"
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
          <label className="block text-xs text-zinc-400">
            Saved Presets
            <select
              className="mt-1 w-full rounded-md border border-zinc-700 px-3 py-2 text-sm"
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
          </label>
          <button
            type="button"
            className="rounded-md border border-zinc-700 px-3 py-2 text-sm text-zinc-300 hover:bg-zinc-900/60"
            onClick={applySelectedPreset}
            disabled={!selectedPresetId}
          >
            Load Preset
          </button>
        </div>

        <div className="mt-2 grid gap-2 md:grid-cols-2">
          <button
            type="button"
            className="w-full rounded-md border border-zinc-700 px-3 py-2 text-sm text-zinc-300 hover:bg-zinc-900/60"
            onClick={() =>
              exportConfigFile(
                { workflow, params: currentWorkflowParams() },
                `workflow-config-${workflow}`,
              )
            }
          >
            Export Current Config
          </button>
          <button
            type="button"
            className="w-full rounded-md border border-zinc-700 px-3 py-2 text-sm text-zinc-300 hover:bg-zinc-900/60"
            onClick={() => document.getElementById("config-import-input")?.click()}
          >
            Import Config File
          </button>
          <input
            id="config-import-input"
            type="file"
            accept="application/json"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (!file) return;
              void onImportConfigFile(file);
              e.currentTarget.value = "";
            }}
          />
        </div>
      </section>

      <section className={sectionClass}>
        <h2 className="text-lg font-medium text-zinc-100">Data & Inputs</h2>
        <div className="mt-3 grid gap-2 md:grid-cols-[1fr_auto]">
          <label className="block text-xs text-zinc-400">
            Data Root Directory
            <input
              className="mt-1 w-full rounded-md border border-zinc-700 px-3 py-2 text-sm"
              value={sequenceRoot}
              onChange={(e) => setSequenceRoot(e.target.value)}
              placeholder="data root directory (for example ./sample)"
            />
          </label>
          <button
            type="button"
            className="rounded-md border border-zinc-700 px-3 py-2 text-sm text-zinc-300 hover:bg-zinc-900/60"
            onClick={() => void refreshSequences().catch((e: Error) => setError(e.message))}
          >
            Load Sequences
          </button>
        </div>
        <p className="mt-2 text-xs text-zinc-500">
          Toggle sequences to update Pass 1 <code>run_seqs</code>, and edit each <code>prompt.json</code> bg
          text in place.
        </p>

        <div className="mt-3 max-h-72 space-y-2 overflow-auto rounded-md border border-zinc-800 bg-zinc-900 p-2">
          {sequences.length === 0 ? (
            <p className="text-xs text-zinc-500">No sequences loaded.</p>
          ) : (
            sequences.map((seq) => (
              <div key={seq.path} className="rounded border border-zinc-800 bg-black/30 p-2">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <p className="text-sm font-medium text-zinc-100">{seq.name}</p>
                    <p className="text-[11px] text-zinc-500">
                      {seq.has_input_video ? "input_video" : "no input_video"} •{" "}
                      {seq.has_quadmask ? "quadmask" : "no quadmask"} •{" "}
                      {seq.has_prompt ? "prompt" : "no prompt"}
                    </p>
                  </div>
                  <label className="flex items-center gap-2 text-xs text-zinc-300">
                    <input
                      type="checkbox"
                      checked={selectedSeqs.has(seq.name)}
                      onChange={() => toggleRunSequence(seq.name)}
                    />
                    run
                  </label>
                </div>
                <div className="mt-2 grid gap-2 md:grid-cols-[1fr_auto]">
                  <label className="block text-[11px] text-zinc-400">
                    Prompt Background
                    <input
                      className="mt-1 w-full rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-200"
                      value={promptDrafts[seq.path] ?? ""}
                      placeholder='prompt bg text, e.g. "A table with a cup on it."'
                      onChange={(e) =>
                        setPromptDrafts((prev) => ({ ...prev, [seq.path]: e.target.value }))
                      }
                    />
                  </label>
                  <button
                    type="button"
                    className="rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-900/60"
                    onClick={() =>
                      void updatePromptBg(seq.path, promptDrafts[seq.path] ?? "")
                        .then(() => refreshSequences())
                        .catch((err: Error) => setError(err.message))
                    }
                  >
                    Save Prompt
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className={sectionClass}>
          <h2 className="text-lg font-medium text-zinc-100">Launch Workflow</h2>
          <form onSubmit={onSubmit} className="mt-4 space-y-4">
            <label className="block text-sm text-zinc-300">
              Workflow
              <select
                value={workflow}
                onChange={(e) => setWorkflow(e.target.value as Workflow)}
                className="mt-1 w-full rounded-md border border-zinc-700 px-3 py-2"
              >
                <option value="pass1_inference">Pass 1 Inference</option>
                <option value="pass2_refine">Pass 2 Refinement</option>
                <option value="mask_pipeline">Mask Pipeline</option>
              </select>
            </label>

            {workflow === "pass1_inference" ? (
              <Pass1FormPanel
                pass1={pass1}
                setPass1={setPass1}
                showAdvancedPass1={showAdvancedPass1}
                setShowAdvancedPass1={setShowAdvancedPass1}
              />
            ) : null}

            {workflow === "pass2_refine" ? (
              <Pass2FormPanel
                pass2VideoNames={pass2VideoNames}
                setPass2VideoNames={setPass2VideoNames}
                pass2DataRoot={pass2DataRoot}
                setPass2DataRoot={setPass2DataRoot}
                pass2Pass1Dir={pass2Pass1Dir}
                setPass2Pass1Dir={setPass2Pass1Dir}
                pass2OutputDir={pass2OutputDir}
                setPass2OutputDir={setPass2OutputDir}
                pass2ModelName={pass2ModelName}
                setPass2ModelName={setPass2ModelName}
                pass2ModelCheckpoint={pass2ModelCheckpoint}
                setPass2ModelCheckpoint={setPass2ModelCheckpoint}
                pass2Height={pass2Height}
                setPass2Height={setPass2Height}
                pass2Width={pass2Width}
                setPass2Width={setPass2Width}
                pass2MaxVideoLength={pass2MaxVideoLength}
                setPass2MaxVideoLength={setPass2MaxVideoLength}
                pass2TemporalWindowSize={pass2TemporalWindowSize}
                setPass2TemporalWindowSize={setPass2TemporalWindowSize}
                pass2Seed={pass2Seed}
                setPass2Seed={setPass2Seed}
                pass2GuidanceScale={pass2GuidanceScale}
                setPass2GuidanceScale={setPass2GuidanceScale}
                pass2Steps={pass2Steps}
                setPass2Steps={setPass2Steps}
                pass2WarpedNoiseCacheDir={pass2WarpedNoiseCacheDir}
                setPass2WarpedNoiseCacheDir={setPass2WarpedNoiseCacheDir}
                pass2SkipNoiseGeneration={pass2SkipNoiseGeneration}
                setPass2SkipNoiseGeneration={setPass2SkipNoiseGeneration}
                pass2UseQuadmask={pass2UseQuadmask}
                setPass2UseQuadmask={setPass2UseQuadmask}
                cachePath={cachePath}
                setCachePath={setCachePath}
                cacheInfo={cacheInfo}
                onCheckCache={() => void refreshCacheInfo().catch((e: Error) => setError(e.message))}
                onClearCache={() =>
                  void clearCache(cachePath)
                    .then((info) => setCacheInfo(info))
                    .catch((e: Error) => setError(e.message))
                }
              />
            ) : null}

            {workflow === "mask_pipeline" ? (
              <div className="space-y-2">
                <TextField
                  value={maskConfigPath}
                  onChange={setMaskConfigPath}
                  placeholder="config_points json"
                />
                <select
                  className="w-full rounded-md border border-zinc-700 px-3 py-2 text-sm"
                  value={maskExecutionMode}
                  onChange={(e) =>
                    setMaskExecutionMode(e.target.value as MaskExecutionMode)
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
                      className="w-full rounded-md border border-zinc-700 px-3 py-2 text-sm"
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
                    className="w-full rounded-md border border-zinc-700 px-3 py-2 text-sm"
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
                    className="rounded-md border border-zinc-700 px-3 py-2 text-sm text-zinc-300 hover:bg-zinc-900/60"
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
                    className="rounded-md border border-zinc-700 px-3 py-2 text-sm text-zinc-300 hover:bg-zinc-900/60"
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
              className="w-full rounded-md bg-red-700 px-4 py-2 text-sm font-medium text-white hover:bg-red-600 disabled:opacity-60"
            >
              {loading ? "Launching..." : "Launch Run"}
            </button>
          </form>
        </section>

        <section className={sectionClass}>
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-medium text-zinc-100">Runs</h2>
            <button
              type="button"
              onClick={() => void refreshRuns().catch((e: Error) => setError(e.message))}
              className="rounded-md border border-zinc-700 px-3 py-1 text-sm text-zinc-300 hover:bg-zinc-900/60"
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
                  void refreshArtifacts(run.id).catch(() => {});
                }}
                className={`w-full rounded-md border p-3 text-left ${
                  activeRunId === run.id ? "border-red-500 bg-red-950/40" : "border-zinc-800 bg-zinc-900"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm font-medium text-zinc-100">{run.workflow}</span>
                  <span className={`text-xs font-semibold uppercase ${statusClass(run.status)}`}>{run.status}</span>
                </div>
                <p className="mt-1 text-xs text-zinc-400">id: {run.id}</p>
                <p className="text-xs text-zinc-500">created: {new Date(run.created_at).toLocaleString()}</p>
              </button>
            ))}
            {runs.length === 0 ? <p className="text-sm text-zinc-400">No runs yet.</p> : null}
          </div>
        </section>
      </div>

      <section className={sectionClass}>
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-lg font-medium text-zinc-100">Run Details</h2>
          <div className="flex gap-2">
            {activeRun ? (
              <>
                <button
                  type="button"
                  className="rounded-md border border-zinc-700 px-3 py-1 text-xs text-zinc-300 hover:bg-zinc-900/60"
                  onClick={() => applyWorkflowConfig(activeRun.workflow, activeRun.params)}
                >
                  Clone To Form
                </button>
                <button
                  type="button"
                  className="rounded-md border border-zinc-700 px-3 py-1 text-xs text-zinc-300 hover:bg-zinc-900/60"
                  onClick={() =>
                    void validateConfig({
                      workflow: activeRun.workflow,
                      params: activeRun.params,
                    })
                      .then((v) => {
                        if (!v.valid) {
                          setError(`Validation failed: ${v.errors.join(" | ")}`);
                          return null;
                        }
                        setValidationWarnings(v.warnings);
                        return createRun({
                          workflow: activeRun.workflow,
                          params: activeRun.params,
                        });
                      })
                      .then((run) => {
                        if (!run) return;
                        setActiveRunId(run.id);
                        return refreshRuns();
                      })
                      .catch((e: Error) => setError(e.message))
                  }
                >
                  Clone & Run
                </button>
                <button
                  type="button"
                  className="rounded-md border border-zinc-700 px-3 py-1 text-xs text-zinc-300 hover:bg-zinc-900/60"
                  onClick={() =>
                    exportConfigFile(
                      { workflow: activeRun.workflow, params: activeRun.params },
                      `run-config-${activeRun.id}`,
                    )
                  }
                >
                  Export Run Config
                </button>
              </>
            ) : null}
            {activeRun?.status === "running" ? (
              <button
                type="button"
                onClick={() =>
                  void cancelRun(activeRun.id)
                    .then(() => refreshRuns())
                    .catch((e: Error) => setError(e.message))
                }
                className="rounded-md border border-red-700 px-3 py-1 text-sm text-red-300 hover:bg-red-950/30"
              >
                Cancel Run
              </button>
            ) : null}
          </div>
        </div>

        {activeRun ? (
          <div className="mt-3 space-y-3 text-sm text-zinc-300">
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

            <div className="rounded-md border border-zinc-800 bg-zinc-900/60 p-3">
              <div className="mb-2 flex items-center justify-between">
                <p className="font-semibold text-zinc-100">Artifacts</p>
                <button
                  type="button"
                  className="rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-900"
                  onClick={() => void refreshArtifacts(activeRun.id).catch((e: Error) => setError(e.message))}
                >
                  Refresh Artifacts
                </button>
              </div>
              {activeRun.output_dir ? (
                <p className="mb-2 text-xs text-zinc-400">Output dir: {activeRun.output_dir}</p>
              ) : (
                <p className="mb-2 text-xs text-zinc-400">This workflow does not declare an output directory.</p>
              )}
              <div className="max-h-48 space-y-1 overflow-auto rounded-md bg-zinc-900 p-2">
                {artifacts.length === 0 ? (
                  <p className="text-xs text-zinc-500">No artifact files found yet.</p>
                ) : (
                  artifacts.map((file) => (
                    <div key={file.path} className="rounded border border-zinc-800 p-2">
                      <p className="text-xs font-medium text-zinc-100">{file.relative}</p>
                      <p className="text-[11px] text-zinc-400">{Math.round(file.size_bytes / 1024)} KB</p>
                      <p className="truncate text-[11px] text-zinc-500">{file.path}</p>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        ) : (
          <p className="mt-3 text-sm text-zinc-400">Select a run to inspect logs.</p>
        )}
      </section>
    </div>
  );
}
