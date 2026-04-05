import type { CacheInfo } from "@/lib/types";

function LabeledInput({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
}) {
  return (
    <label className="block text-xs text-zinc-400">
      {label}
      <input
        className="mt-1 w-full rounded-md border border-zinc-700 px-3 py-2 text-sm"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
    </label>
  );
}

export function Pass2FormPanel({
  pass2VideoNames,
  setPass2VideoNames,
  pass2DataRoot,
  setPass2DataRoot,
  pass2Pass1Dir,
  setPass2Pass1Dir,
  pass2OutputDir,
  setPass2OutputDir,
  pass2ModelName,
  setPass2ModelName,
  pass2ModelCheckpoint,
  setPass2ModelCheckpoint,
  pass2Height,
  setPass2Height,
  pass2Width,
  setPass2Width,
  pass2MaxVideoLength,
  setPass2MaxVideoLength,
  pass2TemporalWindowSize,
  setPass2TemporalWindowSize,
  pass2Seed,
  setPass2Seed,
  pass2GuidanceScale,
  setPass2GuidanceScale,
  pass2Steps,
  setPass2Steps,
  pass2WarpedNoiseCacheDir,
  setPass2WarpedNoiseCacheDir,
  pass2SkipNoiseGeneration,
  setPass2SkipNoiseGeneration,
  pass2UseQuadmask,
  setPass2UseQuadmask,
  cachePath,
  setCachePath,
  cacheInfo,
  onCheckCache,
  onClearCache,
}: {
  pass2VideoNames: string;
  setPass2VideoNames: (v: string) => void;
  pass2DataRoot: string;
  setPass2DataRoot: (v: string) => void;
  pass2Pass1Dir: string;
  setPass2Pass1Dir: (v: string) => void;
  pass2OutputDir: string;
  setPass2OutputDir: (v: string) => void;
  pass2ModelName: string;
  setPass2ModelName: (v: string) => void;
  pass2ModelCheckpoint: string;
  setPass2ModelCheckpoint: (v: string) => void;
  pass2Height: string;
  setPass2Height: (v: string) => void;
  pass2Width: string;
  setPass2Width: (v: string) => void;
  pass2MaxVideoLength: string;
  setPass2MaxVideoLength: (v: string) => void;
  pass2TemporalWindowSize: string;
  setPass2TemporalWindowSize: (v: string) => void;
  pass2Seed: string;
  setPass2Seed: (v: string) => void;
  pass2GuidanceScale: string;
  setPass2GuidanceScale: (v: string) => void;
  pass2Steps: string;
  setPass2Steps: (v: string) => void;
  pass2WarpedNoiseCacheDir: string;
  setPass2WarpedNoiseCacheDir: (v: string) => void;
  pass2SkipNoiseGeneration: boolean;
  setPass2SkipNoiseGeneration: (v: boolean) => void;
  pass2UseQuadmask: boolean;
  setPass2UseQuadmask: (v: boolean) => void;
  cachePath: string;
  setCachePath: (v: string) => void;
  cacheInfo: CacheInfo | null;
  onCheckCache: () => void;
  onClearCache: () => void;
}) {
  return (
    <div className="space-y-2">
      <LabeledInput label="Video Names (Comma-Separated)" value={pass2VideoNames} onChange={setPass2VideoNames} placeholder="video names (comma-separated)" />
      <LabeledInput label="Data Root Directory" value={pass2DataRoot} onChange={setPass2DataRoot} placeholder="data root" />
      <LabeledInput label="Pass 1 Output Directory" value={pass2Pass1Dir} onChange={setPass2Pass1Dir} placeholder="pass1 output dir" />
      <LabeledInput label="Output Directory" value={pass2OutputDir} onChange={setPass2OutputDir} placeholder="output dir" />
      <LabeledInput label="Base Model Path" value={pass2ModelName} onChange={setPass2ModelName} placeholder="base model path" />
      <LabeledInput label="Pass 2 Checkpoint Path" value={pass2ModelCheckpoint} onChange={setPass2ModelCheckpoint} placeholder="pass2 checkpoint" />
      <div className="grid grid-cols-4 gap-2">
        <LabeledInput label="Height" value={pass2Height} onChange={setPass2Height} placeholder="height" />
        <LabeledInput label="Width" value={pass2Width} onChange={setPass2Width} placeholder="width" />
        <LabeledInput label="Guidance Scale" value={pass2GuidanceScale} onChange={setPass2GuidanceScale} placeholder="guidance" />
        <LabeledInput label="Inference Steps" value={pass2Steps} onChange={setPass2Steps} placeholder="steps" />
      </div>
      <div className="grid grid-cols-3 gap-2">
        <LabeledInput label="Max Video Length" value={pass2MaxVideoLength} onChange={setPass2MaxVideoLength} placeholder="max video length" />
        <LabeledInput label="Temporal Window Size" value={pass2TemporalWindowSize} onChange={setPass2TemporalWindowSize} placeholder="temporal window" />
        <LabeledInput label="Seed" value={pass2Seed} onChange={setPass2Seed} placeholder="seed" />
      </div>
      <LabeledInput
        label="Warped Noise Cache Directory"
        value={pass2WarpedNoiseCacheDir}
        onChange={setPass2WarpedNoiseCacheDir}
        placeholder="warped noise cache dir"
      />
      <div className="grid grid-cols-2 gap-2">
        <label className="flex items-center gap-2 rounded-md border border-zinc-700 bg-zinc-900 p-2 text-xs text-zinc-300">
          <input
            type="checkbox"
            checked={pass2SkipNoiseGeneration}
            onChange={(e) => setPass2SkipNoiseGeneration(e.target.checked)}
          />
          skip_noise_generation
        </label>
        <label className="flex items-center gap-2 rounded-md border border-zinc-700 bg-zinc-900 p-2 text-xs text-zinc-300">
          <input
            type="checkbox"
            checked={pass2UseQuadmask}
            onChange={(e) => setPass2UseQuadmask(e.target.checked)}
          />
          use_quadmask
        </label>
      </div>
      <div className="rounded-md border border-zinc-800 bg-zinc-900/60 p-2">
        <p className="mb-2 text-xs font-semibold text-zinc-300">Cache Manager</p>
        <div className="grid grid-cols-2 gap-2">
          <label className="block text-xs text-zinc-400">
            Cache Path
            <input
              className="mt-1 w-full rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-200"
              value={cachePath}
              onChange={(e) => setCachePath(e.target.value)}
              placeholder="cache path"
            />
          </label>
          <div className="flex gap-2">
            <button
              type="button"
              className="w-full rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-900/60"
              onClick={onCheckCache}
            >
              Check
            </button>
            <button
              type="button"
              className="w-full rounded-md border border-red-700 px-2 py-1 text-xs text-red-300 hover:bg-red-950/30"
              onClick={onClearCache}
            >
              Clear
            </button>
          </div>
        </div>
        {cacheInfo ? (
          <p className="mt-2 text-xs text-zinc-400">
            {cacheInfo.exists ? "exists" : "missing"} • {cacheInfo.files} files •{" "}
            {Math.round(cacheInfo.bytes / 1024)} KB
          </p>
        ) : null}
      </div>
    </div>
  );
}
