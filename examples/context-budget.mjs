// Reproduce the README chart offline; Node >=22.18, no model download or GPU.
import { PRESETS, WEIGHT_PRECISIONS, estimate } from '../src/index.ts';

const model = PRESETS.find((p) => p.id === 'meta-llama/Llama-3.1-8B-Instruct');
const precision = WEIGHT_PRECISIONS.find((p) => p.id === 'q4_k_m');
if (!model || !precision) throw new Error('The example model or precision is missing.');
const gib = (bytes) => Number((bytes / 1024 ** 3).toFixed(2));
const rows = [8192, 32768].map((contextTokens) => {
  const result = estimate(model, precision, contextTokens, 1, 16, 10);
  return {
    contextTokens,
    weightsGiB: gib(result.weights),
    kvCacheGiB: gib(result.kvCache),
    overheadGiB: gib(result.overhead),
    totalGiB: gib(result.total),
  };
});
console.log(JSON.stringify({
  model: model.id,
  weightPrecision: precision.id,
  parallelRequests: 1,
  kvBits: 16,
  overhead: '0.5 GiB + 10% of weights and KV cache',
  notice: 'Illustrative estimates, not hardware benchmarks. Runtime allocations can differ.',
  rows,
}, null, 2));
