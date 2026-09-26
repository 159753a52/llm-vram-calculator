import type { ModelSpec } from './vram.ts';

type HubConfig = Record<string, unknown>;
export type HubInfo = {
  safetensors?: { total?: number; parameters?: Record<string, number> };
};

const HUB = 'https://huggingface.co';

const positive = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : undefined;

/**
 * Reads the numbers that decide inference memory from a Hub config.json plus the parameter
 * counts in the safetensors metadata. Layers not listed as sliding-window or linear are
 * treated as full attention, which errs on the side of more memory.
 */
export function specFromHub(id: string, rawConfig: HubConfig, info: HubInfo): ModelSpec {
  // Multimodal checkpoints keep the language model's numbers in text_config.
  const config = positive(rawConfig.num_hidden_layers)
    ? rawConfig
    : { ...rawConfig, ...((rawConfig.text_config as HubConfig | undefined) ?? {}) };

  const layers = positive(config.num_hidden_layers) ?? positive(config.n_layer);
  const heads = positive(config.num_attention_heads) ?? positive(config.n_head);
  const hidden = positive(config.hidden_size) ?? positive(config.n_embd);
  if (!layers || !heads) throw new Error('config.json does not list layers and attention heads.');
  const headDim = positive(config.head_dim) ?? (hidden ? hidden / heads : undefined);
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

  const quantMethod = (config.quantization_config as HubConfig | undefined)?.quant_method;
  if (typeof quantMethod === 'string') spec.quantMethod = quantMethod;

  // DeepSeek-style MLA caches one compressed latent plus the rotary part of the key.
  const kvLoraRank = positive(config.kv_lora_rank);
  if (kvLoraRank) spec.mlaDim = kvLoraRank + (positive(config.qk_rope_head_dim) ?? 0);

  const layerTypes = Array.isArray(config.layer_types) ? config.layer_types : [];
  const window = positive(config.sliding_window);
  const slidingLayers = layerTypes.filter((type) => type === 'sliding_attention').length;
  if (slidingLayers && window) {
    spec.slidingLayers = slidingLayers;
    spec.slidingWindow = window;
  }
  const stateLayers = layerTypes.filter(
    (type) => typeof type === 'string' && /linear|mamba|recurrent/.test(type),
  ).length;
  if (stateLayers) spec.stateLayers = stateLayers;

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

  const [infoRes, configRes] = await Promise.all([
    fetch(`${HUB}/api/models/${id}?expand[]=safetensors`),
    fetch(`${HUB}/${id}/resolve/main/config.json`),
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

  return specFromHub(id, await configRes.json(), await infoRes.json());
}
