import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { estimate, GIB, WEIGHT_PRECISIONS } from '../src/vram.ts';
import { PRESETS } from '../src/presets.ts';

test('README visual examples match the calculation core and disclose their assumptions', () => {
  const spec = PRESETS.find((s) => s.id === 'meta-llama/Llama-3.1-8B-Instruct')!;
  const precision = WEIGHT_PRECISIONS.find((p) => p.id === 'q4_k_m')!;
  const a = estimate(spec, precision, 8192, 1, 16, 10);
  const b = estimate(spec, precision, 32768, 1, 16, 10);
  const hero = readFileSync('assets/readme-hero.svg', 'utf8');
  const chart = readFileSync('assets/vram-breakdown.svg', 'utf8');
  assert.ok(hero.includes((a.total / GIB).toFixed(2)));
  for (const result of [a, b]) assert.ok(chart.includes(`${(result.total / GIB).toFixed(2)} GiB`));
  assert.ok(chart.includes(`+${((b.total - a.total) / GIB).toFixed(2)} GiB`));
  assert.equal(a.weights, b.weights);
  assert.equal(b.kvCache / a.kvCache, 4);
  assert.ok(hero.includes('not hardware benchmarks'));
  assert.ok(chart.includes('Not a benchmark'));
  assert.ok(chart.includes('FP16 KV cache'));
  assert.ok(chart.includes('0.5 GiB + 10%'));
});

test('README art is self-contained static SVG with accessible descriptions and working local references', () => {
  const readme = readFileSync('README.md', 'utf8');
  for (const name of ['readme-hero', 'vram-breakdown']) {
    const path = `assets/${name}.svg`;
    const svg = readFileSync(path, 'utf8');
    assert.ok(readme.includes(path));
    assert.match(svg, /<title id="title">/);
    assert.match(svg, /<desc id="desc">/);
    assert.doesNotMatch(svg, /<script|<foreignObject|<image|onload=|href=/i);
  }
  assert.ok(readme.includes('https://modelvram.com/llm-vram-calculator/'));
});
