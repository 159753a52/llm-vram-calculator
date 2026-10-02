import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { PRESETS } from '../src/presets.ts';
import { specFromHub } from '../src/hf.ts';
import { kvCacheBytes, kvGrowthPerToken } from '../src/vram.ts';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/glm5-next-cache.json', import.meta.url), 'utf8'));
const preset = PRESETS.find((s) => s.id === 'zai-org/GLM-5.3-Flash')!;
const fromHub = specFromHub(preset.id, fixture.config, fixture.info);

// Allocation-shape regression, not a model run. Pinned llama.cpp sources:
// https://github.com/ggml-org/llama.cpp/blob/5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd/src/llama-memory-hybrid-idx.cpp#L53-L73
// https://github.com/ggml-org/llama.cpp/blob/5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd/src/models/glm5-next.cpp#L782-L794
// K-pool stores key | gate | pooled at type_k; no /4 reduction and no second V tensor.
test('GLM-5.3-Flash preset and official nested/text configs select the scoped k-pool layout', () => {
  for (const spec of [preset, fromHub, specFromHub(preset.id, fixture.config.text_config, fixture.info)]) {
    assert.equal(spec.indexCache, 'glm5-next-kpool');
    assert.deepEqual([spec.layers, spec.stateLayers, spec.mlaDim, spec.indexDim, spec.indexLayers], [45, 34, 512, 128, 45]);
    assert.equal(spec.kvNote, preset.kvNote);
    assert.match(spec.kvNote ?? '', /llama.cpp 5fc4f3c/);
    assert.match(spec.kvNote ?? '', /Fixed recurrent state, MTP and scratch/);
    assert.match(spec.kvNote ?? '', /do not establish kernel support or measured GPU memory/);
  }
});

// Q8_0: 32 values + 2-byte scale = 34 bytes. Q4_0: 16 packed bytes + 2-byte scale = 18.
// The latent (512) and packed indexer (384) widths are both exact multiples of 32.
for (const precision of [
  { name: 'FP16', bits: 16, latentRow: 512 * 2, indexRow: 384 * 2 },
  { name: 'Q8_0', bits: 8.5, latentRow: (512 / 32) * 34, indexRow: (384 / 32) * 34 },
  { name: 'Q4_0', bits: 4.5, latentRow: (512 / 32) * 18, indexRow: (384 / 32) * 18 },
  // Preserve the generic 8-bit storage input; this does not claim llama.cpp FP8 kernel support.
  { name: 'generic FP8 storage', bits: 8, latentRow: 512, indexRow: 384 },
]) {
  test(`GLM-5.3-Flash ${precision.name}: full context rows, GGML block sizes and request scaling`, () => {
    const perToken = 11 * (precision.latentRow + precision.indexRow);
    for (const spec of [preset, fromHub]) {
      assert.equal(kvGrowthPerToken(spec, precision.bits), perToken);
      for (const context of [1, 32_768, 131_072, 1_048_576]) {
        for (const requests of [1, 2, 4]) {
          const bytes = kvCacheBytes(spec, context, requests, precision.bits);
          assert.equal(bytes, context * requests * perToken);
          assert.equal(kvCacheBytes(spec, context + 1, requests, precision.bits) - bytes, requests * kvGrowthPerToken(spec, precision.bits));
        }
      }
      assert.equal(kvCacheBytes({ ...spec, indexLayers: 3 }, 1, 1, precision.bits), 11 * precision.latentRow + 3 * precision.indexRow);
      assert.equal(kvCacheBytes({ ...spec, indexLayers: 0 }, 1, 1, precision.bits), 11 * precision.latentRow);
      assert.equal(kvCacheBytes({ ...spec, stateLayers: 45 }, 1, 1, precision.bits), 0);
    }
  });
}

test('GLM-5.3-Flash FP16 context-growing cache is 616 MiB at 32K and 2464 MiB at 128K', () => {
  assert.equal(kvCacheBytes(preset, 32_768, 1, 16) / 1024 ** 2, 616);
  assert.equal(kvCacheBytes(preset, 131_072, 1, 16) / 1024 ** 2, 2464);
  assert.equal(kvCacheBytes(preset, 1_048_576, 1, 16) / 1024 ** 3, 19.25);
});

test('unknown pooling layouts remain approximate rather than borrowing GLM three-slot storage', () => {
  for (const changed of [
    { model_type: 'other_pooled_dsa' },
    { index_head_dim: 96 },
    { index_kpool: 8 },
    { index_kpool_compress: false },
  ]) {
    const spec = specFromHub('x/unknown', { ...fixture.config.text_config, ...changed }, fixture.info);
    assert.equal(spec.indexCache, undefined);
    assert.match(spec.kvNote ?? '', /not modelled.*approximation.*too low/);
  }
});

test('ordinary DSA and GLM-5.2/5.3 keep one FP8 indexer key regardless of KV precision', () => {
  for (const id of ['deepseek-ai/DeepSeek-V3.2', 'zai-org/GLM-5.2', 'zai-org/GLM-5.3']) {
    const spec = PRESETS.find((s) => s.id === id)!;
    assert.equal(spec.indexCache, undefined);
    for (const bits of [16, 8, 8.5, 4.5]) {
      const expected = spec.layers * spec.mlaDim! * bits / 8 + (spec.indexLayers ?? spec.layers) * spec.indexDim!;
      assert.equal(kvGrowthPerToken(spec, bits), expected);
      assert.equal(kvCacheBytes(spec, 32_768, 4, bits), expected * 32_768 * 4);
    }
  }
});

test('MiMo sliding-window and Nemotron recurrent layouts keep their existing cache arithmetic', () => {
  const info = { safetensors: { total: 1e9 } };
  const mimo = specFromHub('x/mimo', {
    num_hidden_layers: 48, num_attention_heads: 64, num_key_value_heads: 4, head_dim: 192, v_head_dim: 128,
    swa_num_key_value_heads: 8, swa_head_dim: 192, swa_v_head_dim: 128, sliding_window: 128,
    hybrid_layer_pattern: [...Array(9).fill(0), ...Array(39).fill(1)],
  }, info);
  const nemotron = specFromHub('x/nemotron', {
    num_hidden_layers: 52, num_attention_heads: 32, num_key_value_heads: 2, head_dim: 128,
    hybrid_override_pattern: '*'.repeat(6) + 'M'.repeat(23) + 'E'.repeat(23),
  }, info);
  for (const bits of [16, 8, 8.5, 4.5]) {
    for (const context of [8192, 131_072]) {
      for (const requests of [1, 4]) {
        const slidingCells = Math.min(context * requests, Math.ceil((128 * requests + 512) / 256) * 256);
        assert.equal(kvCacheBytes(mimo, context, requests, bits), (9 * context * requests * 4 * (192 + 128) + 39 * slidingCells * 8 * (192 + 128)) * bits / 8);
        assert.equal(kvCacheBytes(nemotron, context, requests, bits), 6 * context * requests * 2 * (128 + 128) * bits / 8);
      }
    }
    assert.equal(kvGrowthPerToken(mimo, bits), 9 * 4 * (192 + 128) * bits / 8);
    assert.equal(kvGrowthPerToken(nemotron, bits), 6 * 2 * (128 + 128) * bits / 8);
  }
});
