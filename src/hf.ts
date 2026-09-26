import type { ModelSpec } from './vram.ts';

type HubConfig = Record<string, unknown>;
export type HubInfo = {
  safetensors?: { total?: number; parameters?: Record<string, number> };
};
/** An entry of the Hub's file listing (api/models/<id>/tree/main). */
export type HubFile = { type?: string; path: string; size?: number; lfs?: { size?: number } };

/**
 * Total size of the .safetensors files in the repository root, or undefined if there are none.
 * Shard names vary (MiMo uses model_pp0_ep3_shard0.safetensors), so every root file counts
 * except Mistral's consolidated.safetensors, a second copy of the same weights. Subfolders hold
 * extras such as draft models and audio tokenizers.
 */
export function publishedWeightBytes(files: HubFile[]): number | undefined {
  const total = files
    .filter((file) => file.type !== 'directory' && !file.path.includes('/'))
    .filter((file) => file.path.endsWith('.safetensors') && !file.path.startsWith('consolidated'))
    .reduce((sum, file) => sum + (file.lfs?.size ?? file.size ?? 0), 0);
  return total > 0 ? total : undefined;
}

const HUB = 'https://huggingface.co';

const positive = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : undefined;

/**
 * Reads the numbers that decide inference memory from a Hub config.json plus the parameter
 * counts in the safetensors metadata. Layers not listed as sliding-window or linear are
 * treated as full attention, which errs on the side of more memory.
 */
export function specFromHub(id: string, rawConfig: HubConfig, info: HubInfo, files: HubFile[] = []): ModelSpec {
  // Multimodal checkpoints keep the language model's numbers in text_config.
  const config = positive(rawConfig.num_hidden_layers)
    ? rawConfig
    : { ...rawConfig, ...((rawConfig.text_config as HubConfig | undefined) ?? {}) };

  const layers = positive(config.num_hidden_layers) ?? positive(config.n_layer);
  const heads = positive(config.num_attention_heads) ?? positive(config.n_head);
  const hidden = positive(config.hidden_size) ?? positive(config.n_embd);
  if (!layers || !heads) throw new Error('config.json does not list layers and attention heads.');
  const mlaKeyDim = (positive(config.qk_nope_head_dim) ?? 0) + (positive(config.qk_rope_head_dim) ?? 0);
  const headDim = positive(config.head_dim) ?? (mlaKeyDim || undefined) ?? (hidden ? hidden / heads : undefined);
  if (!headDim) throw new Error('config.json does not list head_dim or hidden_size.');
  const params = positive(info.safetensors?.total);
  if (!params) throw new Error('The model files do not report a parameter count.');

  const spec: ModelSpec = {
    id,
    name: id.split('/').pop() ?? id,
    params,
    dtypes: info.safetensors?.parameters,
    layers,
    kvHeads: positive(config.num_key_value_heads) ?? positive(config.multi_query_group_num) ?? heads,
    headDim,
  };

  const experts = positive(config.n_routed_experts) ?? positive(config.num_experts) ?? positive(config.num_local_experts);
  if (experts) spec.experts = experts;
  const publishedBytes = publishedWeightBytes(files);
  if (publishedBytes) spec.publishedBytes = publishedBytes;
  const quantMethod = (config.quantization_config as HubConfig | undefined)?.quant_method;
  if (typeof quantMethod === 'string') spec.quantMethod = quantMethod;

  // DeepSeek-style MLA caches one compressed latent plus the rotary part of the key.
  const kvLoraRank = positive(config.kv_lora_rank);
  if (kvLoraRank) spec.mlaDim = kvLoraRank + (positive(config.qk_rope_head_dim) ?? 0);
  // DeepSeek sparse attention keeps an indexer key per token; GLM-5 lets some layers share one.
  const indexDim = positive(config.index_head_dim);
  if (kvLoraRank && indexDim) {
    spec.indexDim = indexDim;
    const indexerTypes = Array.isArray(config.indexer_types) ? config.indexer_types : [];
    if (indexerTypes.length) spec.indexLayers = indexerTypes.filter((type) => type === 'full').length;
  }
  const vHeadDim = positive(config.v_head_dim);
  if (!kvLoraRank && vHeadDim && vHeadDim !== headDim) spec.vHeadDim = vHeadDim;

  // Gemma 4 gives its global layers their own cache shape, with keys doubling as values.
  const globalHeads = positive(config.num_global_key_value_heads);
  const globalHeadDim = positive(config.global_head_dim);
  if (!kvLoraRank && (globalHeads || globalHeadDim)) {
    spec.slidingKvHeads = spec.kvHeads;
    spec.slidingHeadDim = spec.headDim;
    spec.kvHeads = globalHeads ?? spec.kvHeads;
    spec.headDim = globalHeadDim ?? spec.headDim;
    if (config.attention_k_eq_v === true) spec.vHeadDim = 0;
  }
  // MiMo V2 describes its sliding-window layers with swa_* fields.
  const swaHeads = positive(config.swa_num_key_value_heads);
  if (!kvLoraRank && swaHeads) {
    spec.slidingKvHeads = swaHeads;
    spec.slidingHeadDim = positive(config.swa_head_dim) ?? spec.headDim;
    const swaVHeadDim = positive(config.swa_v_head_dim);
    if (swaVHeadDim && swaVHeadDim !== spec.slidingHeadDim) spec.slidingVHeadDim = swaVHeadDim;
  }

  // Hybrid models name their layer kinds in several ways: layer_types (most); a 0/1
  // hybrid_layer_pattern with 1 for sliding-window layers (MiMo V2); linear_attn_config's list of
  // linear layers (Kimi); one full-attention layer every full_attention_interval (Qwen3-Next);
  // or a hybrid_override_pattern string where only "*" layers are attention (Nemotron-H: M is
  // Mamba, E and - are feed-forward blocks).
  const layerTypes = Array.isArray(config.layer_types) ? config.layer_types : [];
  const pattern = Array.isArray(config.hybrid_layer_pattern) ? config.hybrid_layer_pattern : [];
  const linearConfig = (config.linear_attn_config as HubConfig | undefined) ?? {};
  const kdaLayers = Array.isArray(linearConfig.kda_layers) ? linearConfig.kda_layers.length : 0;
  const overridePattern = typeof config.hybrid_override_pattern === 'string' ? config.hybrid_override_pattern : '';
  const interval = positive(config.full_attention_interval);
  const window = positive(config.sliding_window);
  const slidingLayers = layerTypes.length
    ? layerTypes.filter((type) => type === 'sliding_attention').length
    : pattern.filter((kind) => kind === 1).length;
  if (slidingLayers && window) {
    spec.slidingLayers = slidingLayers;
    spec.slidingWindow = window;
  }
  let stateLayers = kdaLayers;
  if (layerTypes.length) {
    stateLayers = layerTypes.filter((type) => typeof type === 'string' && /linear|mamba|recurrent/.test(type)).length;
  } else if (overridePattern) {
    stateLayers = layers - [...overridePattern].filter((kind) => kind === '*').length;
    spec.stateKind = 'Mamba or feed-forward';
  } else if (interval && interval > 1) {
    stateLayers = layers - Math.floor(layers / interval);
  }
  if (stateLayers > 0) spec.stateLayers = stateLayers;
  if (config.kv_source_layer_ids || config.compress_ratios) {
    spec.kvNote =
      'This model shares and compresses its KV cache across layers, which the calculator does not model, so the KV cache figure is an upper bound.';
  }

  const maxContext = positive(config.max_position_embeddings);
  if (maxContext) spec.maxContext = maxContext;
  return spec;
}

/** Accepts "owner/model" or a huggingface.co URL. */
export function parseModelId(input: string): string | undefined {
  const id = input
    .trim()
    .replace(/^https?:\/\/(www\.)?huggingface\.co\//, '')
    .split(/[?#]/)[0]
    .replace(/\/(tree|blob)\/.*$/, '')
    .replace(/\/$/, '');
  return /^[\w.-]+\/[\w.-]+$/.test(id) ? id : undefined;
}

export async function loadFromHub(input: string): Promise<ModelSpec> {
  const id = parseModelId(input);
  if (!id) throw new Error('Enter a model id like Qwen/Qwen3-8B.');

  const [infoRes, configRes, filesRes] = await Promise.all([
    fetch(`${HUB}/api/models/${id}?expand[]=safetensors`),
    fetch(`${HUB}/${id}/resolve/main/config.json`),
    fetch(`${HUB}/api/models/${id}/tree/main`),
  ]);
  // The Hub answers 401, not 404, for ids that do not exist, so private repos stay hidden.
  if (infoRes.status === 404 || infoRes.status === 401) {
    throw new Error(`Hugging Face has no public model called ${id}. Check the spelling.`);
  }
  if (!infoRes.ok) throw new Error(`Hugging Face answered ${infoRes.status}. Try again in a moment.`);
  if (configRes.status === 401 || configRes.status === 403) {
    throw new Error(`${id} is gated, so its config.json needs a login. Enter its numbers under Advanced instead.`);
  }
  if (!configRes.ok) throw new Error(`${id} has no readable config.json (${configRes.status}).`);

  // The file listing only sharpens "as published"; the estimate works without it.
  const files: HubFile[] = filesRes.ok ? await filesRes.json().catch(() => []) : [];
  return specFromHub(id, await configRes.json(), await infoRes.json(), Array.isArray(files) ? files : []);
}
