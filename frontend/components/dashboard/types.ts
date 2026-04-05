export type Pass1Form = {
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

export type MaskExecutionMode = "full" | "stage1" | "stage2" | "stage3" | "stage4";
