/**
 * GPU memory for LLM inference = weights + KV cache + runtime overhead.
 * Pure functions with no DOM access, so they can be tested with `node --test`.
 */

export const GIB = 1024 ** 3;

export type ModelSpec = {
  /** Hugging Face id, or "custom" for hand-entered numbers. */
  id: string;
  name: string;
  /** Total parameters. For MoE models every expert counts: all of them sit in memory. */
  params: number;
  /** Parameter counts by stored dtype (safetensors metadata), used for "as published". */
  dtypes?: Record<string, number>;
  /** Routed experts of a mixture-of-experts model; all of them sit in memory. */
  experts?: number;
  /**
   * Parameters a mixture-of-experts model uses per token, from its model card; `activeEstimated`
   * marks the few worked out from config.json because the card gives no number.
   */
  activeParams?: number;
  activeEstimated?: boolean;
  /** quantization_config.quant_method, e.g. "mxfp4" or "fp8". */
  quantMethod?: string;
  /**
   * Bytes of the published weight files. When known, "as published" uses it, which stays exact
   * for storage formats the dtype table cannot describe (DeepSeek V4 keeps FP4 experts in I8).
   */
  publishedBytes?: number;
  layers: number;
  kvHeads: number;
  headDim: number;
  /**
   * Value head size when it differs from the key's (MiMo V2 caches 192-wide keys, 128-wide
   * values). 0 means the values are the keys, cached once (Gemma 4's global layers).
   */
  vHeadDim?: number;
  /** Multi-head latent attention (DeepSeek V2/V3): values cached per token per layer. */
  mlaDim?: number;
  /**
   * DeepSeek sparse attention (DeepSeek V3.2, GLM-5): the indexer also caches `indexDim` FP8
   * values per token, in `indexLayers` layers (every full-attention layer unless shared).
   */
  indexDim?: number;
  indexLayers?: number;
  /** Layers that keep only the last `slidingWindow` tokens. */
  slidingLayers?: number;
  slidingWindow?: number;
  /**
   * Cache shape of the sliding-window layers when it differs from the full-attention layers,
   * which `kvHeads` and `headDim` describe (Gemma 4: 16 × 256 sliding, 4 × 512 global).
   */
  slidingKvHeads?: number;
  slidingHeadDim?: number;
  slidingVHeadDim?: number;
  /** Layers with a fixed-size state instead of a KV cache (linear attention, Mamba). */
  stateLayers?: number;
  /** What those layers are, for display; defaults to linear attention. */
  stateKind?: string;
  maxContext?: number;
  /** Shown next to the result when the cache is known to be smaller than modelled here. */
  kvNote?: string;
};

export type Precision = { id: string; label: string; bits: number };

// GGUF sizes are llama.cpp's averages over a whole model (bits per weight, scales included).
export const WEIGHT_PRECISIONS: Precision[] = [
  { id: 'native', label: 'As published', bits: 0 },
  { id: 'fp32', label: 'FP32', bits: 32 },
  { id: 'bf16', label: 'FP16 / BF16', bits: 16 },
  { id: 'fp8', label: 'FP8 / INT8', bits: 8 },
  { id: 'int4', label: 'INT4 (AWQ / GPTQ)', bits: 4.25 },
  { id: 'q8_0', label: 'GGUF Q8_0', bits: 8.5 },
  { id: 'q6_k', label: 'GGUF Q6_K', bits: 6.56 },
  { id: 'q5_k_m', label: 'GGUF Q5_K_M', bits: 5.67 },
  { id: 'q4_k_m', label: 'GGUF Q4_K_M', bits: 4.84 },
  { id: 'q3_k_m', label: 'GGUF Q3_K_M', bits: 3.91 },
  { id: 'q2_k', label: 'GGUF Q2_K', bits: 3.35 },
];

export const KV_PRECISIONS: Precision[] = [
  { id: 'fp16', label: 'FP16 / BF16', bits: 16 },
  { id: 'fp8', label: 'FP8', bits: 8 },
  { id: 'q8_0', label: 'Q8_0 (llama.cpp)', bits: 8.5 },
  { id: 'q4_0', label: 'Q4_0 (llama.cpp)', bits: 4.5 },
];

const DTYPE_BITS: Record<string, number> = {
  F64: 64,
  I64: 64,
  F32: 32,
  I32: 32,
  BF16: 16,
  F16: 16,
  I16: 16,
  F8_E4M3: 8,
  F8_E5M2: 8,
  I8: 8,
  U8: 8,
  BOOL: 8,
};

/** Bytes to hold the weights. MXFP4 checkpoints store 4.25-bit weights in tensors typed U8. */
export function weightBytes(spec: ModelSpec, precision: Precision): number {
  if (precision.id !== 'native') return (spec.params * precision.bits) / 8;
  if (spec.publishedBytes) return spec.publishedBytes;
  if (!spec.dtypes) return spec.params * 2;
  let bits = 0;
  for (const [dtype, count] of Object.entries(spec.dtypes)) {
    const perWeight = dtype === 'U8' && spec.quantMethod === 'mxfp4' ? 4.25 : (DTYPE_BITS[dtype] ?? 16);
    bits += count * perWeight;
  }
  return bits / 8;
}

/** Values cached per token in one attention layer: K and V, or MLA's compressed latent. */
export function kvValuesPerTokenPerLayer(spec: ModelSpec): number {
  return spec.mlaDim ?? spec.kvHeads * (spec.headDim + (spec.vHeadDim ?? spec.headDim));
}

/** Values one sliding-window layer caches per token. */
export function slidingValuesPerTokenPerLayer(spec: ModelSpec): number {
  if (spec.slidingKvHeads === undefined && spec.slidingHeadDim === undefined) return kvValuesPerTokenPerLayer(spec);
  const headDim = spec.slidingHeadDim ?? spec.headDim;
  return (spec.slidingKvHeads ?? spec.kvHeads) * (headDim + (spec.slidingVHeadDim ?? headDim));
}

/** KV cache for `requests` sequences of `context` tokens each. */
export function kvCacheBytes(spec: ModelSpec, context: number, requests: number, kvBits: number): number {
  const attentionLayers = Math.max(0, spec.layers - (spec.stateLayers ?? 0));
  const sliding = Math.min(spec.slidingLayers ?? 0, attentionLayers);
  const full = attentionLayers - sliding;
  const slidingTokens = Math.min(context, spec.slidingWindow ?? context);
  const values =
    full * context * kvValuesPerTokenPerLayer(spec) + sliding * slidingTokens * slidingValuesPerTokenPerLayer(spec);
  return requests * ((values * kvBits) / 8 + context * indexerBytesPerToken(spec));
}

/** The sparse-attention indexer's own cache, kept in FP8 whatever the KV cache precision. */
function indexerBytesPerToken(spec: ModelSpec): number {
  if (!spec.indexDim) return 0;
  const attentionLayers = Math.max(0, spec.layers - (spec.stateLayers ?? 0) - (spec.slidingLayers ?? 0));
  return Math.min(spec.indexLayers ?? attentionLayers, attentionLayers) * spec.indexDim;
}

/** Cache added by each extra token of one request once every sliding window is full. */
export function kvGrowthPerToken(spec: ModelSpec, kvBits: number): number {
  const attentionLayers = Math.max(0, spec.layers - (spec.stateLayers ?? 0));
  const full = attentionLayers - Math.min(spec.slidingLayers ?? 0, attentionLayers);
  return (full * kvValuesPerTokenPerLayer(spec) * kvBits) / 8 + indexerBytesPerToken(spec);
}

/** Memory every GPU needs no matter how small the model: the CUDA context and allocator pools. */
export const RUNTIME_BASE_BYTES = 0.5 * GIB;

export type Estimate = {
  weights: number;
  kvCache: number;
  overhead: number;
  total: number;
  /** How much the cache of one request grows per token once the sliding windows are full. */
  kvPerToken: number;
};

/**
 * `overheadPct` covers activations, temporary buffers and fragmentation on top of weights and
 * cache; engines differ, so it is an input rather than a constant.
 */
export function estimate(
  spec: ModelSpec,
  precision: Precision,
  context: number,
  requests: number,
  kvBits: number,
  overheadPct: number,
): Estimate {
  const weights = weightBytes(spec, precision);
  const kvCache = kvCacheBytes(spec, context, requests, kvBits);
  const overhead = RUNTIME_BASE_BYTES + ((weights + kvCache) * overheadPct) / 100;
  return {
    weights,
    kvCache,
    overhead,
    total: weights + kvCache + overhead,
    kvPerToken: kvGrowthPerToken(spec, kvBits),
  };
}

export type Gpu = { name: string; gib: number; note?: string };

export const GPUS: Gpu[] = [
  { name: 'RTX 3060', gib: 12 },
  { name: 'RTX 4060 Ti 16GB', gib: 16 },
  { name: 'RTX 3090 / 4090', gib: 24 },
  { name: 'RTX 5090', gib: 32 },
  { name: 'A100 40GB', gib: 40 },
  { name: 'Mac, 64 GB unified memory', gib: 48, note: 'about 75% of it is usable by the GPU by default' },
  { name: 'L40S / RTX 6000 Ada', gib: 48 },
  { name: 'A100 / H100 80GB', gib: 80 },
  { name: 'Mac, 128 GB unified memory', gib: 96, note: 'about 75% of it is usable by the GPU by default' },
  { name: 'H200', gib: 141 },
  { name: 'B200', gib: 180 },
];

/**
 * Smallest tensor-parallel group (1, 2, 4 or 8 GPUs) that fits: weights and cache split evenly,
 * while every GPU pays the runtime base. Undefined when even 8 are not enough.
 */
export function gpusNeeded(est: Estimate, overheadPct: number, gpuGib: number): number | undefined {
  for (const count of [1, 2, 4, 8]) {
    const share = (est.weights + est.kvCache) / count;
    const perGpu = share * (1 + overheadPct / 100) + RUNTIME_BASE_BYTES;
    if (perGpu <= gpuGib * GIB) return count;
  }
  return undefined;
}

/** "12.3 GB" with GB meaning GiB, the unit GPU memory is sold in. */
export function formatGib(bytes: number): string {
  const gib = bytes / GIB;
  if (gib >= 100) return `${Math.round(gib).toLocaleString('en-US')} GB`;
  if (gib >= 10) return `${gib.toFixed(1)} GB`;
  if (gib >= 1) return `${gib.toFixed(2)} GB`;
  return `${Math.round(bytes / 1024 ** 2).toLocaleString('en-US')} MB`;
}
