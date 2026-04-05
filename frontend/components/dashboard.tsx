"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

import { cancelRun, createRun, getEnvCheck, getRunLogs, listRuns } from "@/lib/api";
import type { EnvCheck, RunRecord, Workflow } from "@/lib/types";

const sectionClass =
  "rounded-xl border border-slate-200 bg-white/90 p-4 shadow-sm backdrop-blur-sm";

function statusClass(status: RunRecord["status"]) {
  if (status === "completed") return "text-emerald-700";
  if (status === "failed") return "text-red-700";
  if (status === "running") return "text-blue-700";
  if (status === "cancelled") return "text-amber-700";
  return "text-slate-700";
}

export function Dashboard() {
  const [workflow, setWorkflow] = useState<Workflow>("pass1_inference");
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [activeRunId, setActiveRunId] = useState<string>("");
  const [logs, setLogs] = useState<string>("");
  const [env, setEnv] = useState<EnvCheck | null>(null);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);

  const [pass1ConfigPath, setPass1ConfigPath] = useState("config/quadmask_cogvideox.py");
  const [pass1DataRoot, setPass1DataRoot] = useState("./sample");
  const [pass1RunSeqs, setPass1RunSeqs] = useState("lime");
  const [pass1SavePath, setPass1SavePath] = useState("./outputs");
  const [pass1ModelName, setPass1ModelName] = useState("./CogVideoX-Fun-V1.5-5b-InP");
  const [pass1TransformerPath, setPass1TransformerPath] = useState("./void_pass1.safetensors");
  const [pass1SampleSize, setPass1SampleSize] = useState("384x672");
  const [pass1GuidanceScale, setPass1GuidanceScale] = useState("1.0");
  const [pass1Steps, setPass1Steps] = useState("50");

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

  useEffect(() => {
    void refreshEnv().catch((e: Error) => setError(e.message));
    void refreshRuns().catch((e: Error) => setError(e.message));
  }, [refreshEnv, refreshRuns]);

  useEffect(() => {
    const id = window.setInterval(() => {
      void refreshRuns().catch(() => {});
      if (activeRunId) {
        void refreshLogs(activeRunId).catch(() => {});
      }
    }, 2500);
    return () => window.clearInterval(id);
  }, [activeRunId, refreshLogs, refreshRuns]);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      if (workflow === "pass1_inference") {
        const run = await createRun({
          workflow,
          params: {
            config_path: pass1ConfigPath,
            config_overrides: {
              "config.data.data_rootdir": pass1DataRoot,
              "config.experiment.run_seqs": pass1RunSeqs,
              "config.experiment.save_path": pass1SavePath,
              "config.video_model.model_name": pass1ModelName,
              "config.video_model.transformer_path": pass1TransformerPath,
              "config.data.sample_size": pass1SampleSize,
              "config.video_model.guidance_scale": Number(pass1GuidanceScale),
              "config.video_model.num_inference_steps": Number(pass1Steps),
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
        const run = await createRun({
          workflow,
          params: {
            config_points_json: maskConfigPath,
            sam2_checkpoint: maskSam2Checkpoint,
            device: maskDevice,
          },
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
              <div className="space-y-2">
                <input className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm" value={pass1ConfigPath} onChange={(e) => setPass1ConfigPath(e.target.value)} placeholder="config path" />
                <input className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm" value={pass1DataRoot} onChange={(e) => setPass1DataRoot(e.target.value)} placeholder="data root" />
                <input className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm" value={pass1RunSeqs} onChange={(e) => setPass1RunSeqs(e.target.value)} placeholder="run seqs (comma-separated)" />
                <input className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm" value={pass1SavePath} onChange={(e) => setPass1SavePath(e.target.value)} placeholder="save path" />
                <input className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm" value={pass1ModelName} onChange={(e) => setPass1ModelName(e.target.value)} placeholder="base model path" />
                <input className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm" value={pass1TransformerPath} onChange={(e) => setPass1TransformerPath(e.target.value)} placeholder="pass1 checkpoint path" />
                <div className="grid grid-cols-3 gap-2">
                  <input className="rounded-md border border-slate-300 px-3 py-2 text-sm" value={pass1SampleSize} onChange={(e) => setPass1SampleSize(e.target.value)} placeholder="sample_size" />
                  <input className="rounded-md border border-slate-300 px-3 py-2 text-sm" value={pass1GuidanceScale} onChange={(e) => setPass1GuidanceScale(e.target.value)} placeholder="guidance" />
                  <input className="rounded-md border border-slate-300 px-3 py-2 text-sm" value={pass1Steps} onChange={(e) => setPass1Steps(e.target.value)} placeholder="steps" />
                </div>
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
                <input className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm" value={maskConfigPath} onChange={(e) => setMaskConfigPath(e.target.value)} placeholder="config_points json" />
                <input className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm" value={maskSam2Checkpoint} onChange={(e) => setMaskSam2Checkpoint(e.target.value)} placeholder="sam2 checkpoint" />
                <select className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm" value={maskDevice} onChange={(e) => setMaskDevice(e.target.value as "cuda" | "cpu")}>
                  <option value="cuda">cuda</option>
                  <option value="cpu">cpu</option>
                </select>
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
