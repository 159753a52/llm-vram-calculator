import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PRESETS } from '../src/presets.ts';
import { WEIGHT_PRECISIONS, weightBytes } from '../src/vram.ts';
import { SPEED_GPUS, activeWeightBytes, bytesPerToken, formatRange, tokensPerSecond } from '../src/speed.ts';

const preset = (id: string) => PRESETS.find((p) => p.id === id)!;
const precision = (id: string) => WEIGHT_PRECISIONS.find((p) => p.id === id)!;
const gpu = (name: string) => SPEED_GPUS.find((g) => g.name === name)!;

test('dense models read every weight per token', () => {
  const llama = preset('meta-llama/Llama-3.1-8B-Instruct');
  assert.equal(activeWeightBytes(llama, precision('q4_k_m')), weightBytes(llama, precision('q4_k_m')));
});

test('MoE models read only their active share', () => {
  const qwen = preset('Qwen/Qwen3-30B-A3B');
  const share = 3.3e9 / qwen.params;
  assert.equal(activeWeightBytes(qwen, precision('q4_k_m')), weightBytes(qwen, precision('q4_k_m')) * share);
});

test('the KV cache is read for every token, so long contexts are slower', () => {
  const llama = preset('meta-llama/Llama-3.1-8B-Instruct');
  const short = bytesPerToken(llama, precision('q4_k_m'), 512, 16);
  const long = bytesPerToken(llama, precision('q4_k_m'), 131_072, 16);
  assert.equal(long - short, (131_072 - 512) * 131_072);
});

test('matches published llama.cpp speeds on an RTX 4090', () => {
  // Llama 3.1 8B Q4_K_M: about 130 tokens/s measured, 205 the bandwidth limit.
  const dense = tokensPerSecond(preset('meta-llama/Llama-3.1-8B-Instruct'), precision('q4_k_m'), 512, 16, gpu('RTX 4090'));
  assert.ok(Math.abs(dense.limit - 205) < 3, String(dense.limit));
  assert.ok(dense.low < 130 && dense.high > 130, formatRange(dense));
  // Qwen3-30B-A3B Q4_K_M: about 170 tokens/s measured.
  const moe = tokensPerSecond(preset('Qwen/Qwen3-30B-A3B'), precision('q4_k_m'), 512, 16, gpu('RTX 4090'));
  assert.ok(moe.low < 170 && moe.high > 170, formatRange(moe));
});

test('the fixed cost per token caps small models on fast GPUs', () => {
  const small = tokensPerSecond(preset('Qwen/Qwen3.6-35B-A3B'), precision('q4_k_m'), 4096, 16, gpu('H200'));
  assert.ok(small.high < 2000 / 1.5, formatRange(small));
  assert.ok(small.limit > 2 * small.high, 'the bandwidth limit alone would be far higher');
});

test('tensor parallel adds up the bandwidth of the cards', () => {
  const spec = preset('meta-llama/Llama-3.1-70B-Instruct');
  const one = tokensPerSecond(spec, precision('fp8'), 4096, 16, gpu('H100 SXM'));
  const two = tokensPerSecond(spec, precision('fp8'), 4096, 16, gpu('H100 SXM'), 2);
  assert.equal(two.limit, one.limit * 2);
  // Swapping results between the cards after every layer keeps it below twice as fast.
  assert.ok(two.high < one.high * 2 && two.high > one.high, formatRange(two));
});

test('the FAQ numbers', () => {
  const llama = preset('meta-llama/Llama-3.1-8B-Instruct');
  const weights = activeWeightBytes(llama, precision('q4_k_m'));
  assert.ok(weights > 4.8e9 && weights < 4.95e9, String(weights));
  // At 128K tokens the cache is 16 GiB, about 3.5 times the weights, and writing is ~4.5x slower.
  const ratio = bytesPerToken(llama, precision('q4_k_m'), 131_072, 16) / bytesPerToken(llama, precision('q4_k_m'), 512, 16);
  assert.ok(ratio > 4.3 && ratio < 4.6, String(ratio));
  const qwen = preset('Qwen/Qwen3-30B-A3B');
  const moe = tokensPerSecond(qwen, precision('q4_k_m'), 512, 16, gpu('RTX 4090'));
  const dense = tokensPerSecond(llama, precision('q4_k_m'), 512, 16, gpu('RTX 4090'));
  assert.ok(moe.low > dense.low, 'Qwen3-30B-A3B should write faster than Llama 3.1 8B');
  assert.deepEqual(
    ['M4 Max Mac (128 GB)', 'M3 Ultra Mac Studio (512 GB)', 'Ryzen AI Max+ 395 (128 GB)'].map((name) => gpu(name).bandwidth),
    [546, 819, 256],
  );
});

test('formatRange', () => {
  assert.equal(formatRange({ limit: 0, low: 112.4, high: 153.6, bytes: 0 }), '112–154');
  assert.equal(formatRange({ limit: 0, low: 3.14, high: 4.26, bytes: 0 }), '3.1–4.3');
});
