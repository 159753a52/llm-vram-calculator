import { GIB, kvCacheBytes, weightBytes, type ModelSpec, type Precision } from './vram.ts';

/** `gib` is what the model may use; `bandwidth` is the memory bandwidth in GB/s (10^9 bytes). */
export type SpeedGpu = { name: string; gib: number; bandwidth: number; note?: string };

// Bandwidths from the makers' spec sheets. Unified-memory machines let the GPU use about 75% of
// their memory by default, so `gib` is 75% of what they are sold with.
export const SPEED_GPUS: SpeedGpu[] = [
  { name: 'RTX 3060 12GB', gib: 12, bandwidth: 360 },
  { name: 'RTX 4060 Ti 16GB', gib: 16, bandwidth: 288 },
  { name: 'RTX 3090', gib: 24, bandwidth: 936 },
  { name: 'RTX 4090', gib: 24, bandwidth: 1008 },
  { name: 'RX 7900 XTX', gib: 24, bandwidth: 960 },
  { name: 'RTX 5090', gib: 32, bandwidth: 1792 },
  { name: 'M4 Pro Mac (64 GB)', gib: 48, bandwidth: 273, note: 'about 75% of unified memory is usable by the GPU' },
  { name: 'M4 Max Mac (128 GB)', gib: 96, bandwidth: 546, note: 'about 75% of unified memory is usable by the GPU' },
  { name: 'M3 Ultra Mac Studio (512 GB)', gib: 384, bandwidth: 819, note: 'about 75% of unified memory is usable by the GPU' },
  { name: 'Ryzen AI Max+ 395 (128 GB)', gib: 96, bandwidth: 256, note: 'up to 96 GB can be assigned to the GPU' },
  { name: 'DGX Spark (128 GB)', gib: 120, bandwidth: 273 },
  { name: 'L40S', gib: 48, bandwidth: 864 },
  { name: 'RTX PRO 6000 Blackwell', gib: 96, bandwidth: 1792 },
  { name: 'A100 80GB', gib: 80, bandwidth: 2039 },
  { name: 'H100 SXM', gib: 80, bandwidth: 3350 },
  { name: 'H200', gib: 141, bandwidth: 4800 },
  { name: 'B200', gib: 180, bandwidth: 8000 },
];

/**
 * Share of the bandwidth limit that engines such as llama.cpp and vLLM reach for one request,
 * plus a fixed cost per token for launching kernels and sampling, which is what caps small models
 * on fast GPUs. Checked against published benchmarks: Llama 3.1 8B at Q4_K_M runs at about
 * 130 tokens/s on an RTX 4090 (bandwidth limit 205), Qwen3-30B-A3B at about 170 (limit 500).
 */
export const EFFICIENCY = { dense: [0.55, 0.75], moe: [0.3, 0.5] } as const;
export const FIXED_MS = [1.5, 0.5] as const;

/** Weight bytes read for every generated token: all of a dense model, the active part of an MoE. */
export function activeWeightBytes(spec: ModelSpec, precision: Precision): number {
  const share = spec.activeParams && spec.activeParams < spec.params ? spec.activeParams / spec.params : 1;
  return weightBytes(spec, precision) * share;
}

/** Bytes read per generated token: the active weights plus the whole KV cache of the request. */
export function bytesPerToken(spec: ModelSpec, precision: Precision, context: number, kvBits: number): number {
  return activeWeightBytes(spec, precision) + kvCacheBytes(spec, context, 1, kvBits);
}

export type Speed = { limit: number; low: number; high: number; bytes: number };

/** Generation speed of one request, in tokens per second, on `gpus` cards in tensor parallel. */
export function tokensPerSecond(
  spec: ModelSpec,
  precision: Precision,
  context: number,
  kvBits: number,
  gpu: SpeedGpu,
  gpus = 1,
): Speed {
  const bytes = bytesPerToken(spec, precision, context, kvBits);
  const limit = (gpu.bandwidth * 1e9 * gpus) / bytes;
  const [low, high] = spec.activeParams && spec.activeParams < spec.params ? EFFICIENCY.moe : EFFICIENCY.dense;
  const speed = (efficiency: number, fixedMs: number) => 1 / (1 / (limit * efficiency) + fixedMs / 1000);
  return { limit, low: speed(low, FIXED_MS[0]), high: speed(high, FIXED_MS[1]), bytes };
}

/** "120–165" or, below 10, one decimal: "3.1–4.2". */
export function formatRange(speed: Speed): string {
  const f = (value: number) => (value >= 10 ? Math.round(value).toLocaleString('en-US') : value.toFixed(1));
  return `${f(speed.low)}–${f(speed.high)}`;
}

export const formatBytes = (bytes: number) =>
  bytes >= GIB ? `${(bytes / GIB).toFixed(bytes >= 10 * GIB ? 1 : 2)} GB` : `${Math.round(bytes / 1024 ** 2)} MB`;
