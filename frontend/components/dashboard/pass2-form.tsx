import type { CacheInfo } from "@/lib/types";

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
      <input className="w-full rounded-md border border-zinc-700 px-3 py-2 text-sm" value={pass2VideoNames} onChange={(e) => setPass2VideoNames(e.target.value)} placeholder="video names (comma-separated)" />
      <input className="w-full rounded-md border border-zinc-700 px-3 py-2 text-sm" value={pass2DataRoot} onChange={(e) => setPass2DataRoot(e.target.value)} placeholder="data root" />
      <input className="w-full rounded-md border border-zinc-700 px-3 py-2 text-sm" value={pass2Pass1Dir} onChange={(e) => setPass2Pass1Dir(e.target.value)} placeholder="pass1 output dir" />
      <input className="w-full rounded-md border border-zinc-700 px-3 py-2 text-sm" value={pass2OutputDir} onChange={(e) => setPass2OutputDir(e.target.value)} placeholder="output dir" />
      <input className="w-full rounded-md border border-zinc-700 px-3 py-2 text-sm" value={pass2ModelName} onChange={(e) => setPass2ModelName(e.target.value)} placeholder="base model path" />
      <input className="w-full rounded-md border border-zinc-700 px-3 py-2 text-sm" value={pass2ModelCheckpoint} onChange={(e) => setPass2ModelCheckpoint(e.target.value)} placeholder="pass2 checkpoint" />
      <div className="grid grid-cols-4 gap-2">
        <input className="rounded-md border border-zinc-700 px-3 py-2 text-sm" value={pass2Height} onChange={(e) => setPass2Height(e.target.value)} placeholder="height" />
        <input className="rounded-md border border-zinc-700 px-3 py-2 text-sm" value={pass2Width} onChange={(e) => setPass2Width(e.target.value)} placeholder="width" />
        <input className="rounded-md border border-zinc-700 px-3 py-2 text-sm" value={pass2GuidanceScale} onChange={(e) => setPass2GuidanceScale(e.target.value)} placeholder="guidance" />
        <input className="rounded-md border border-zinc-700 px-3 py-2 text-sm" value={pass2Steps} onChange={(e) => setPass2Steps(e.target.value)} placeholder="steps" />
      </div>
      <input
        className="w-full rounded-md border border-zinc-700 px-3 py-2 text-sm"
        value={pass2WarpedNoiseCacheDir}
        onChange={(e) => setPass2WarpedNoiseCacheDir(e.target.value)}
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
          <input
            className="rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-200"
            value={cachePath}
            onChange={(e) => setCachePath(e.target.value)}
            placeholder="cache path"
          />
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
