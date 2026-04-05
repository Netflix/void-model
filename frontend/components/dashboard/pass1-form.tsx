import { Dispatch, SetStateAction } from "react";

import { NumberField, TextField } from "@/components/dashboard/fields";
import type { Pass1Form } from "@/components/dashboard/types";

export function Pass1FormPanel({
  pass1,
  setPass1,
  showAdvancedPass1,
  setShowAdvancedPass1,
}: {
  pass1: Pass1Form;
  setPass1: Dispatch<SetStateAction<Pass1Form>>;
  showAdvancedPass1: boolean;
  setShowAdvancedPass1: Dispatch<SetStateAction<boolean>>;
}) {
  return (
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
        className="w-full rounded-md border border-zinc-700 px-3 py-2 text-sm text-zinc-300 hover:bg-zinc-900/60"
        onClick={() => setShowAdvancedPass1((v) => !v)}
      >
        {showAdvancedPass1 ? "Hide Advanced Pass 1 Controls" : "Show Advanced Pass 1 Controls"}
      </button>

      {showAdvancedPass1 ? (
        <div className="space-y-3 rounded-md border border-zinc-800 bg-zinc-900/60 p-3">
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
            <label className="text-xs text-zinc-400">
              sampler
              <select
                className="mt-1 w-full rounded-md border border-zinc-700 px-3 py-2 text-sm"
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
            <label className="text-xs text-zinc-400">
              gpu memory mode
              <select
                className="mt-1 w-full rounded-md border border-zinc-700 px-3 py-2 text-sm"
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
                <option value="model_cpu_offload_and_qfloat8">model_cpu_offload_and_qfloat8</option>
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
              onChange={(value) => setPass1((p) => ({ ...p, temproalMultidiffusionStride: value }))}
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
            {[
              ["useQuadmask", "use_quadmask"],
              ["useTrimask", "use_trimask"],
              ["useVaeMask", "use_vae_mask"],
              ["stackMask", "stack_mask"],
              ["zeroOutMaskRegion", "zero_out_mask_region"],
              ["skipIfExists", "skip_if_exists"],
              ["validation", "validation"],
              ["skipUnet", "skip_unet"],
              ["maskToVae", "mask_to_vae"],
              ["allowSkippingError", "allow_skipping_error"],
            ].map(([key, label]) => (
              <label
                key={key}
                className="flex items-center gap-2 rounded-md border border-zinc-700 bg-zinc-900 p-2 text-xs text-zinc-300"
              >
                <input
                  type="checkbox"
                  checked={Boolean(pass1[key as keyof Pass1Form])}
                  onChange={(e) =>
                    setPass1((p) => ({
                      ...p,
                      [key]: e.target.checked,
                    }))
                  }
                />
                {label}
              </label>
            ))}
          </div>

          <div className="grid grid-cols-4 gap-2">
            <label className="text-xs text-zinc-400">
              device
              <select
                className="mt-1 w-full rounded-md border border-zinc-700 px-3 py-2 text-sm"
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
            <label className="text-xs text-zinc-400">
              matting mode
              <select
                className="mt-1 w-full rounded-md border border-zinc-700 px-3 py-2 text-sm"
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
  );
}
