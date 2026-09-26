// Writes data/vram-requirements.csv and .json from the presets, with the same functions the
// calculator uses:
//
//   npm run export-data
//
// Totals assume one request, an FP16 KV cache and the default overhead (0.5 GiB + 10%).
import { mkdirSync, writeFileSync } from 'node:fs';
import { PRESETS } from '../src/presets.ts';
import { GIB, WEIGHT_PRECISIONS, estimate, kvGrowthPerToken } from '../src/vram.ts';

const CONTEXT = 8192;
const OVERHEAD_PCT = 10;
const PRECISIONS = ['native', 'bf16', 'fp8', 'int4', 'q8_0', 'q6_k', 'q5_k_m', 'q4_k_m', 'q3_k_m', 'q2_k'];

const gib = (bytes) => Math.round((bytes / GIB) * 100) / 100;
const precision = (id) => WEIGHT_PRECISIONS.find((p) => p.id === id);

const rows = PRESETS.map((spec) => {
  const row = {
    model: spec.name,
    hf_id: spec.id,
    params: spec.params,
    experts: spec.experts ?? '',
    layers: spec.layers,
    max_context: spec.maxContext ?? '',
    kv_kib_per_token_fp16: Math.round((kvGrowthPerToken(spec, 16) / 1024) * 10) / 10,
  };
  for (const id of PRECISIONS) {
    const est = estimate(spec, precision(id), CONTEXT, 1, 16, OVERHEAD_PCT);
    row[`weights_gib_${id}`] = gib(est.weights);
    row[`total_gib_${id}_8k`] = gib(est.total);
  }
  const limit = spec.maxContext ?? CONTEXT;
  row.total_gib_q4_k_m_max_context = gib(estimate(spec, precision('q4_k_m'), limit, 1, 16, OVERHEAD_PCT).total);
  return row;
});

const columns = Object.keys(rows[0]);
const cell = (value) => (typeof value === 'string' && /[",\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : String(value));
const csv = [columns.join(','), ...rows.map((row) => columns.map((c) => cell(row[c])).join(','))].join('\n') + '\n';

mkdirSync('data', { recursive: true });
writeFileSync('data/vram-requirements.csv', csv);
writeFileSync(
  'data/vram-requirements.json',
  JSON.stringify({ context_tokens: CONTEXT, requests: 1, kv_cache: 'fp16', overhead: '0.5 GiB + 10%', unit: 'GiB', models: rows }, null, 2) + '\n',
);
console.log(`wrote data/vram-requirements.csv and .json (${rows.length} models, ${columns.length} columns)`);
