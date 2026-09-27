import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PRESETS } from '../src/presets.ts';
import { parseModelId, publishedWeightBytes, specFromHub } from '../src/hf.ts';
import {
  GIB,
  WEIGHT_PRECISIONS,
  estimate,
  formatGib,
  gpusNeeded,
  kvCacheBytes,
  kvGrowthPerToken,
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

test('IQ4_XS and IQ3_XXS stay within 10% of real GGUF downloads', () => {
  // unsloth file sizes on Hugging Face, 2026-09-27.
  const files: [string, string, number][] = [
    ['meta-llama/Llama-3.1-8B-Instruct', 'iq4_xs', 4.158],
    ['meta-llama/Llama-3.1-8B-Instruct', 'iq3_xxs', 3.094],
    ['Qwen/Qwen3-30B-A3B', 'iq4_xs', 15.253],
    ['Qwen/Qwen3-30B-A3B', 'iq3_xxs', 12.003],
    ['Qwen/Qwen3.6-27B', 'iq4_xs', 14.38],
    ['Qwen/Qwen3.6-27B', 'iq3_xxs', 11.171],
    ['Qwen/Qwen3.8-27B', 'iq4_xs', 13.27],
    ['Qwen/Qwen3.8-27B', 'iq3_xxs', 10.18],
  ];
  for (const [id, type, gib] of files) close(weightBytes(preset(id), precision(type)) / GIB, gib, 0.1);
  // Each sits between its neighbours, so the dropdown stays ordered by size.
  const bits = (id: string) => precision(id).bits;
  assert.ok(bits('q4_k_m') > bits('iq4_xs') && bits('iq4_xs') > bits('q3_k_m') && bits('q3_k_m') > bits('iq3_xxs'));
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

test('"As published" uses the real file size when the repository listing is known', () => {
  const files = [
    { type: 'file', path: 'model_pp0_ep0_shard0.safetensors', lfs: { size: 1_000 } },
    { type: 'file', path: 'model_mtp.safetensors', lfs: { size: 200 } },
    { type: 'file', path: 'consolidated-00001-of-00001.safetensors', lfs: { size: 1_150 } },
    { type: 'file', path: 'dflash/model.safetensors', lfs: { size: 300 } },
    { type: 'file', path: 'config.json', size: 5 },
  ];
  assert.equal(publishedWeightBytes(files), 1_200);
  assert.equal(publishedWeightBytes([{ type: 'file', path: 'README.md', size: 10 }]), undefined);

  // MiMo V2.6 Flash stores FP4 experts in U8 tensors while its quant_method says fp8.
  const mimo = specFromHub(
    'XiaomiMiMo/MiMo-V2.6-Flash-RL',
    { num_hidden_layers: 48, num_attention_heads: 64, num_key_value_heads: 4, head_dim: 192 },
    { safetensors: { total: 310_756_322_688, parameters: { U8: 302_795_194_368 } } },
    [{ type: 'file', path: 'model_pp0_ep0_shard0.safetensors', lfs: { size: 172_932_505_264 } }],
  );
  assert.equal(weightBytes(mimo, precision('native')), 172_932_505_264);
  assert.equal(weightBytes(mimo, precision('bf16')), 310_756_322_688 * 2);
});

test('keys and values of different widths: MiMo V2 caches 4 × (192 + 128) values per layer', () => {
  const spec = { ...preset('Qwen/Qwen3-8B'), layers: 1, kvHeads: 4, headDim: 192, vHeadDim: 128 };
  assert.equal(kvCacheBytes(spec, 1, 1, 16), 4 * (192 + 128) * 2);
});

test('Gemma 4: global layers cache 4 × 512 keys that double as values, sliding layers 16 × 256', () => {
  const gemma = specFromHub(
    'google/gemma-4-31B-it',
    {
      text_config: {
        num_hidden_layers: 60,
        num_attention_heads: 32,
        num_key_value_heads: 16,
        head_dim: 256,
        num_global_key_value_heads: 4,
        global_head_dim: 512,
        attention_k_eq_v: true,
        sliding_window: 1024,
        layer_types: [...Array(50).fill('sliding_attention'), ...Array(10).fill('full_attention')],
      },
    },
    { safetensors: { total: 31_273_088_876 } },
  );
  assert.deepEqual(
    [gemma.kvHeads, gemma.headDim, gemma.vHeadDim, gemma.slidingKvHeads, gemma.slidingHeadDim],
    [4, 512, 0, 16, 256],
  );
  // 262K tokens: 10 global layers × 2,048 values per token, plus 50 sliding layers × 1,024 tokens × 8,192.
  assert.equal(kvCacheBytes(gemma, 262_144, 1, 16), (10 * 262_144 * 2_048 + 50 * 1_024 * 8_192) * 2);
  // Past the window only the global layers grow: 40 KB per token.
  assert.equal(kvGrowthPerToken(gemma, 16), 10 * 2_048 * 2);
});

test('specFromHub reads the hybrid layouts of MiMo, Kimi and DeepSeek V4', () => {
  const info = { safetensors: { total: 1e9 } };
  const mimo = specFromHub(
    'x/mimo',
    {
      num_hidden_layers: 6,
      num_attention_heads: 64,
      num_key_value_heads: 4,
      head_dim: 192,
      v_head_dim: 128,
      sliding_window: 128,
      hybrid_layer_pattern: [0, 1, 1, 1, 1, 0],
      swa_num_key_value_heads: 8,
      swa_head_dim: 192,
      swa_v_head_dim: 128,
    },
    info,
  );
  assert.deepEqual([mimo.slidingLayers, mimo.slidingWindow, mimo.vHeadDim], [4, 128, 128]);
  assert.deepEqual([mimo.slidingKvHeads, mimo.slidingHeadDim, mimo.slidingVHeadDim], [8, 192, 128]);
  // 1,000 tokens: 2 global layers × 1,000 × 4 × 320 values, 4 sliding layers × 128 × 8 × 320.
  assert.equal(kvCacheBytes(mimo, 1_000, 1, 16), (2 * 1_000 * 1_280 + 4 * 128 * 2_560) * 2);

  const kimi = specFromHub(
    'x/kimi',
    {
      num_hidden_layers: 8,
      hidden_size: 7168,
      num_attention_heads: 96,
      kv_lora_rank: 512,
      qk_nope_head_dim: 128,
      qk_rope_head_dim: 64,
      v_head_dim: 128,
      linear_attn_config: { kda_layers: [1, 2, 3, 5, 6, 7], full_attn_layers: [4, 8] },
    },
    info,
  );
  assert.deepEqual([kimi.mlaDim, kimi.headDim, kimi.stateLayers, kimi.vHeadDim], [576, 192, 6, undefined]);

  const deepseek = specFromHub(
    'x/dsv4',
    { num_hidden_layers: 40, num_attention_heads: 64, num_key_value_heads: 1, head_dim: 512, kv_source_layer_ids: [2, 8] },
    info,
  );
  assert.match(deepseek.kvNote ?? '', /upper bound/);
  assert.equal(specFromHub('x/plain', { num_hidden_layers: 2, num_attention_heads: 2, head_dim: 64 }, info).kvNote, undefined);
});

test('DeepSeek sparse attention adds an FP8 indexer key per token, fewer when layers share it', () => {
  const info = { safetensors: { total: 1e9 } };
  const v32 = specFromHub(
    'x/dsv32',
    { num_hidden_layers: 61, num_attention_heads: 128, kv_lora_rank: 512, qk_rope_head_dim: 64, index_head_dim: 128 },
    info,
  );
  assert.deepEqual([v32.mlaDim, v32.indexDim, v32.indexLayers], [576, 128, undefined]);
  assert.equal(kvCacheBytes(v32, 1, 1, 16), 61 * (576 * 2 + 128));
  assert.equal(kvGrowthPerToken(v32, 8), 61 * (576 + 128));

  const glm = specFromHub(
    'x/glm5',
    {
      num_hidden_layers: 8,
      num_attention_heads: 64,
      kv_lora_rank: 512,
      qk_rope_head_dim: 64,
      index_head_dim: 128,
      indexer_types: ['full', 'full', 'shared', 'shared', 'full', 'shared', 'shared', 'shared'],
    },
    info,
  );
  assert.equal(glm.indexLayers, 3);
  assert.equal(kvCacheBytes(glm, 10, 1, 16), 10 * (8 * 576 * 2 + 3 * 128));
});

test('Gemma 4 E models: the last layers reuse earlier caches and keep none of their own', () => {
  const pattern = Array.from({ length: 12 }, (_, i) => (i % 6 === 5 ? 'full_attention' : 'sliding_attention'));
  const spec = specFromHub(
    'x/gemma-e',
    {
      num_hidden_layers: 12,
      num_attention_heads: 8,
      num_key_value_heads: 2,
      head_dim: 256,
      global_head_dim: 512,
      sliding_window: 512,
      num_kv_shared_layers: 6,
      layer_types: pattern,
    },
    { safetensors: { total: 1e9 } },
  );
  // Own caches in layers 0-5 only: five sliding layers and one global layer.
  assert.deepEqual([spec.sharedKvLayers, spec.slidingLayers, spec.kvHeads, spec.headDim], [6, 5, 2, 512]);
  assert.equal(kvCacheBytes(spec, 1_000, 1, 16), (1 * 1_000 * 2 * 1_024 + 5 * 512 * 2 * 512) * 2);
});

test('specFromHub counts attention layers in Qwen3-Next and Nemotron-H layouts', () => {
  const info = { safetensors: { total: 1e9 } };
  const next = specFromHub(
    'x/qwen3-next',
    { num_hidden_layers: 48, num_attention_heads: 16, num_key_value_heads: 2, head_dim: 256, full_attention_interval: 4 },
    info,
  );
  assert.equal(next.stateLayers, 36);
  const nemotron = specFromHub(
    'x/nemotron-h',
    { num_hidden_layers: 8, num_attention_heads: 32, num_key_value_heads: 2, head_dim: 128, hybrid_override_pattern: 'MEM*EME*' },
    info,
  );
  assert.deepEqual([nemotron.stateLayers, nemotron.stateKind], [6, 'Mamba or feed-forward']);
  assert.equal(kvCacheBytes(nemotron, 1_000, 1, 16), 2 * 1_000 * 2 * 2 * 128 * 2);
});

test('the presets behind the FAQ: linear layers in Qwen3.8 27B and Kimi K3', () => {
  assert.deepEqual([preset('Qwen/Qwen3.8-27B').stateLayers, preset('Qwen/Qwen3.8-27B').layers], [48, 64]);
  assert.deepEqual([preset('moonshotai/Kimi-K3').stateLayers, preset('moonshotai/Kimi-K3').layers], [69, 93]);
  assert.equal(preset('zai-org/GLM-5.3').mlaDim, 576);
  assert.equal(preset('deepseek-ai/DeepSeek-V3.2').indexDim, 128);
  assert.equal(preset('Qwen/Qwen3.6-35B-A3B').activeParams, 3e9);
  const gemma = preset('google/gemma-4-31B-it');
  assert.deepEqual([gemma.slidingLayers, gemma.layers, gemma.slidingWindow, gemma.kvHeads, gemma.headDim], [50, 60, 1024, 4, 512]);
  assert.match(preset('deepseek-ai/DeepSeek-V4.1-Flash').kvNote ?? '', /upper bound/);
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
