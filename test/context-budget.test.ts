import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

test('offline context example reproduces both README chart rows and labels assumptions', () => {
  const example = new URL('../examples/context-budget.mjs', import.meta.url);
  const output = JSON.parse(execFileSync(process.execPath, [fileURLToPath(example)], { encoding: 'utf8' }));
  assert.equal(output.parallelRequests, 1);
  assert.equal(output.kvBits, 16);
  assert.equal(output.weightPrecision, 'q4_k_m');
  assert.match(output.notice, /not hardware benchmarks/);
  assert.match(output.overhead, /0.5 GiB \+ 10%/);
  assert.deepEqual(output.rows, [
    { contextTokens: 8192, weightsGiB: 4.52, kvCacheGiB: 1, overheadGiB: 1.05, totalGiB: 6.58 },
    { contextTokens: 32768, weightsGiB: 4.52, kvCacheGiB: 4, overheadGiB: 1.35, totalGiB: 9.88 },
  ]);
  assert.ok(readFileSync('README.md', 'utf8').includes('node examples/context-budget.mjs'));
});
