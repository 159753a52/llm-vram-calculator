import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadFromHub } from '../src/hf.ts';

// Synthetic metadata, not a real model or a live Hub response.
const id = 'fixture/Synthetic-1B';
const config = { num_hidden_layers: 16, num_attention_heads: 16, num_key_value_heads: 4, hidden_size: 2048 };
const info = { safetensors: { total: 1_000_000_000, parameters: { BF16: 1_000_000_000 } } };
const endpoints = [
  `https://huggingface.co/api/models/${id}?expand[]=safetensors`,
  `https://huggingface.co/${id}/resolve/main/config.json`,
  `https://huggingface.co/api/models/${id}/tree/main`,
];

test('a gated config directs Node callers to an accessible repository or an offline preset', async (t) => {
  const calls: string[] = [];
  t.mock.method(globalThis, 'fetch', async (input: string) => {
    const index = endpoints.indexOf(String(input));
    assert.notEqual(index, -1, 'unexpected request must never reach the network');
    calls.push(String(input));
    return Response.json([info, config, []][index], { status: [200, 403, 200][index] });
  });
  await assert.rejects(loadFromHub(id), {
    message: `${id} is gated, so its config.json needs a login. Use an accessible model repository or estimate a built-in preset instead.`,
  });
  assert.deepEqual(calls, endpoints);
});

test('a file-list HTTP error still returns the model metadata without exact published bytes', async (t) => {
  const calls: string[] = [];
  t.mock.method(globalThis, 'fetch', async (input: string) => {
    const index = endpoints.indexOf(String(input));
    assert.notEqual(index, -1, 'unexpected request must never reach the network');
    calls.push(String(input));
    return Response.json([info, config, []][index], { status: [200, 200, 503][index] });
  });
  assert.deepEqual(await loadFromHub(id), {
    id, name: 'Synthetic-1B', params: 1_000_000_000, dtypes: { BF16: 1_000_000_000 },
    layers: 16, kvHeads: 4, headDim: 128,
  });
  assert.deepEqual(calls, endpoints);
});
