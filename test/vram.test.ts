import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PRESETS } from '../src/presets.ts';
import { parseModelId, specFromHub } from '../src/hf.ts';
import {
  GIB,
  WEIGHT_PRECISIONS,
  estimate,
  formatGib,
  gpusNeeded,
  kvCacheBytes,
  weightBytes,
  type Precision,
} from '../src/vram.ts';

const preset = (id: string) => PRESETS.find((p) => p.id === id)!;
const precision = (id: string): Precision => WEIGHT_PRECISIONS.find((p) => p.id === id)!;
const close = (actual: number, expected: number, tolerance = 1e-4) =>
  assert.ok(Math.abs(actual - expected) / expected < tolerance, `${actual} is not ~${expected}`);

test('KV cache: Llama 3.1 8B holds 128 KiB per token, 16 GiB at 128K context', () => {
  const llama = preset('meta-llama/Llama-3.1-8B-Instruct');
  assert.equal(kvCacheBytes(llama, 1, 1, 16), 131_072);
  assert.equal(kvCacheBytes(llama, 131_072, 1, 16), 16 * GIB);
  assert.equal(kvCacheBytes(llama, 131_072, 4, 8), 32 * GIB);
});

test('"As published" matches the real checkpoint sizes on Hugging Face', () => {
  // Byte totals of the .safetensors files (headers make up the small difference).
  close(weightBytes(preset('openai/gpt-oss-20b'), precision('native')), 13_761_316_904);
  close(weightBytes(preset('openai/gpt-oss-120b'), precision('native')), 65_248_893_184);
  close(weightBytes(preset('deepseek-ai/DeepSeek-V3'), precision('native')), 641.28 * GIB);
});

test('fixed precisions scale the parameter count', () => {
  const llama = preset('meta-llama/Llama-3.1-8B-Instruct');
  assert.equal(weightBytes(llama, precision('bf16')), 8_030_261_248 * 2);
  assert.equal(weightBytes(llama, precision('q4_k_m')), (8_030_261_248 * 4.84) / 8);
});

test('MoE models load every expert', () => {
  const moe = preset('Qwen/Qwen3-30B-A3B');
  assert.equal(weightBytes(moe, precision('bf16')), 30_532_122_624 * 2);
});

test('MLA caches the compressed latent: DeepSeek V3 needs 70,272 bytes per token', () => {
  assert.equal(kvCacheBytes(preset('deepseek-ai/DeepSeek-V3'), 1, 1, 16), 61 * 576 * 2);
});

test('sliding-window layers stop growing at the window', () => {
  const gpt = preset('openai/gpt-oss-20b');
  const perLayerToken = 2 * 8 * 64 * 2;
  assert.equal(kvCacheBytes(gpt, 100, 1, 16), 24 * 100 * perLayerToken);
  assert.equal(kvCacheBytes(gpt, 1000, 1, 16), (12 * 1000 + 12 * 128) * perLayerToken);
});

test('linear-attention layers keep no KV cache', () => {
  const hybrid = { ...preset('Qwen/Qwen3-8B'), stateLayers: 27 };
  assert.equal(kvCacheBytes(hybrid, 10, 1, 16), 9 * 10 * 2 * 8 * 128 * 2);
});

test('overhead is the CUDA context plus a share of weights and cache', () => {
  const est = estimate(preset('Qwen/Qwen3-8B'), precision('bf16'), 8192, 1, 16, 10);
  close(est.overhead, 0.5 * GIB + (est.weights + est.kvCache) * 0.1);
  close(est.total, est.weights + est.kvCache + est.overhead);
});

test('GPU count: Llama 3.1 70B in BF16 needs two 80 GB cards or eight 24 GB cards', () => {
  const est = estimate(preset('meta-llama/Llama-3.1-70B-Instruct'), precision('bf16'), 8192, 1, 16, 10);
  assert.equal(gpusNeeded(est, 10, 80), 2);
  assert.equal(gpusNeeded(est, 10, 24), 8);
  assert.equal(gpusNeeded(est, 10, 12), undefined);
});

test('specFromHub reads a plain GQA config and derives head_dim', () => {
  const spec = specFromHub(
    'Qwen/Qwen2.5-7B-Instruct',
    {
      num_hidden_layers: 28,
      hidden_size: 3584,
      num_attention_heads: 28,
      num_key_value_heads: 4,
      sliding_window: 131072,
      use_sliding_window: false,
      max_position_embeddings: 32768,
    },
    { safetensors: { total: 7_615_616_512, parameters: { BF16: 7_615_616_512 } } },
  );
  assert.deepEqual(
    { layers: spec.layers, kvHeads: spec.kvHeads, headDim: spec.headDim, sliding: spec.slidingLayers },
    { layers: 28, kvHeads: 4, headDim: 128, sliding: undefined },
  );
  assert.equal(spec.maxContext, 32768);
});

test('specFromHub picks up MLA, sliding windows, MXFP4 and nested text configs', () => {
  const info = { safetensors: { total: 1e9 } };
  const deepseek = specFromHub(
    'x/ds',
    { num_hidden_layers: 61, hidden_size: 7168, num_attention_heads: 128, kv_lora_rank: 512, qk_rope_head_dim: 64 },
    info,
  );
  assert.equal(deepseek.mlaDim, 576);

  const gpt = specFromHub(
    'x/gpt',
    {
      num_hidden_layers: 4,
      num_attention_heads: 64,
      num_key_value_heads: 8,
      head_dim: 64,
      sliding_window: 128,
      layer_types: ['sliding_attention', 'full_attention', 'sliding_attention', 'full_attention'],
      quantization_config: { quant_method: 'mxfp4' },
    },
    info,
  );
  assert.deepEqual([gpt.slidingLayers, gpt.slidingWindow, gpt.quantMethod], [2, 128, 'mxfp4']);

  const vision = specFromHub(
    'x/vlm',
    { model_type: 'vlm', text_config: { num_hidden_layers: 10, hidden_size: 1024, num_attention_heads: 8 } },
    info,
  );
  assert.deepEqual([vision.layers, vision.kvHeads, vision.headDim], [10, 8, 128]);

  assert.throws(() => specFromHub('x/empty', {}, info), /layers and attention heads/);
});

test('parseModelId accepts ids and Hub URLs only', () => {
  assert.equal(parseModelId(' Qwen/Qwen3-8B '), 'Qwen/Qwen3-8B');
  assert.equal(parseModelId('https://huggingface.co/openai/gpt-oss-20b/tree/main'), 'openai/gpt-oss-20b');
  assert.equal(parseModelId('llama'), undefined);
  assert.equal(parseModelId('a/b/c'), undefined);
});

test('formatGib', () => {
  assert.equal(formatGib(16 * GIB), '16.0 GB');
  assert.equal(formatGib(1.5 * GIB), '1.50 GB');
  assert.equal(formatGib(641.28 * GIB), '641 GB');
  assert.equal(formatGib(512 * 1024 ** 2), '512 MB');
});
