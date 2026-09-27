# LLM VRAM Calculator

Estimate how much GPU memory a large language model needs for inference (weights, KV cache and runtime
overhead) for any model on Hugging Face.

**Use it online: https://modelvram.com/llm-vram-calculator/**

This repository holds the calculation core behind that page: dependency-free TypeScript, plus tests that pin
it to real checkpoint sizes.

## Why another calculator

Most calculators know a fixed list of models, or treat every model as plain multi-head attention. This one
reads the model itself:

- `config.json` gives the layers, KV heads and head size;
- the safetensors metadata on the Hub gives the parameter count for each stored dtype.

So new models work the day they are uploaded, and newer architectures come out right:

| Case | What is modelled |
| --- | --- |
| MoE | Every expert counts toward weight memory: Qwen3-30B-A3B loads 30.5B parameters, not 3B. |
| MLA (DeepSeek V3, GLM-5.3, Kimi K3) | The cache holds `kv_lora_rank + qk_rope_head_dim` values per token per layer: 576 for DeepSeek V3, about 69 KB per token in BF16. |
| Sliding-window layers | Layers listed as `sliding_attention` in `layer_types`, or marked 1 in MiMo's `hybrid_layer_pattern`, cache only the window (gpt-oss: 128 tokens on half of its layers). |
| Different cache shapes per layer kind | Gemma 4's global layers cache 4 heads of 512 values that serve as both keys and values, its sliding layers 16 heads of 256; MiMo V2 caches 192-wide keys and 128-wide values. |
| Linear-attention and state-space layers | No KV cache that grows with context: 48 of Qwen3.8 27B's 64 layers, 69 of Kimi K3's 93 (from `linear_attn_config`). |
| Sparse-attention indexer (DeepSeek V3.2, GLM-5) | The indexer keeps its own FP8 key per token (`index_head_dim`, 128 values), in every layer or, for GLM-5, only in the layers listed as `full` in `indexer_types`. |
| Hybrid layouts named other ways | `full_attention_interval` (Qwen3-Next: one full-attention layer in four) and Nemotron-H's `hybrid_override_pattern` (only `*` layers are attention). |
| Layers that reuse an earlier cache (Gemma 4 E models) | `num_kv_shared_layers`: the last layers keep no cache of their own, 18 of Gemma 4 E4B's 42. |
| Cross-layer cache sharing (DeepSeek V4) | Not modelled; the model carries a note that its KV figure is an upper bound. |

"As published" is the size of the `.safetensors` files in the repository root (Mistral's duplicate
`consolidated.safetensors` excluded), so it matches the download whatever the storage format: MXFP4 in
gpt-oss, 4-bit experts in DeepSeek V4 and MiMo V2.6. Without the file listing it adds up the stored
dtypes, which gives 12.82 GB for gpt-oss-20b and 641.3 GB for DeepSeek V3.
(GB means GiB throughout, the unit GPU memory is sold in.)

The site has a page per model with every quantization, the cache at long context and the GPUs that fit,
for example [Gemma 4 31B](https://modelvram.com/llm-vram-calculator/gemma-4-31b-it/) and
[DeepSeek V4.1 Flash](https://modelvram.com/llm-vram-calculator/deepseek-v4.1-flash/).

## The formula

```text
weights  = parameters × bits per weight ÷ 8
KV cache = 2 × layers × KV heads × head dim × bytes per value × tokens × requests
           (MLA: layers × latent size × bytes per value × tokens × requests)
overhead = 0.5 GiB CUDA context + a share of weights and cache (10% by default)
```

A GPU fits the model with 1, 2, 4 or 8-way tensor parallelism when every card holds its share of the weights
and cache plus the overhead.

These are estimates. vLLM reserves a fixed share of GPU memory up front, and llama.cpp allocates the cache for
the full context when it starts, so leave some headroom.

## Data

[`data/vram-requirements.csv`](data/vram-requirements.csv) and [`.json`](data/vram-requirements.json) list the
weights and total GPU memory of every preset at ten precisions, with an 8,192-token FP16 cache, one request and the
default overhead; `npm run export-data` regenerates them. Totals in GiB:

| Model | Parameters | As published | Q8_0 | Q4_K_M | Q4_K_M at max context | KV cache per token (FP16) |
|---|---|---|---|---|---|---|
| DeepSeek V4 Flash 0731 | 304.2B | 172 | 332 | 190 | 284 (1M) | 86 KiB |
| Qwen3.6 35B-A3B (MoE) | 36.0B | 74.3 | 39.8 | 22.9 | 28.3 (256K) | 20 KiB |
| Qwen3.6 27B | 27.8B | 58.0 | 31.3 | 18.3 | 35.3 (256K) | 64 KiB |
| Qwen3.8 27B | 27.8B | 58.0 | 31.3 | 18.3 | 35.3 (256K) | 64 KiB |
| Qwen3.8 Flash Next (180B MoE) | 180.0B | 370 | 197 | 112 | 119 (256K) | 24 KiB |
| Qwen3.8 2.4T-A95B (MoE) | 2.45T | 5,013 | 2,664 | 1,517 | 1,542 (256K) | 92 KiB |
| Qwen3.5 9B | 9.7B | 20.6 | 11.3 | 6.8 | 15.3 (256K) | 32 KiB |
| Qwen3.5 122B-A10B (MoE) | 125.1B | 257 | 137 | 78.2 | 84.6 (256K) | 24 KiB |
| Qwen3-Coder-Next (80B MoE) | 79.7B | 164 | 87.4 | 50.1 | 56.5 (256K) | 24 KiB |
| DeepSeek V4.1 Flash | 763.2B | 524 | 832 | 474 | 562 (1M) | 80 KiB |
| DeepSeek V4 Flash | 290.9B | 165 | 318 | 182 | 275 (1M) | 86 KiB |
| DeepSeek V4 Pro | 1.60T | 887 | 1,742 | 992 | 1,126 (1M) | 122 KiB |
| DeepSeek V3.2 | 685.4B | 708 | 747 | 426 | 438 (160K) | 76.3 KiB |
| GLM-5.3 | 753.3B | 775 | 821 | 468 | 567 (1M) | 90.4 KiB |
| GLM-5.3 Flash | 321.3B | 337 | 350 | 200 | 213 (1M) | 12.4 KiB |
| GLM-5.2 | 753.3B | 1,545 | 821 | 468 | 567 (1M) | 90.4 KiB |
| GLM-4.7 Flash | 31.2B | 64.9 | 34.9 | 20.3 | 31.1 (198K) | 52.9 KiB |
| Gemma 4 31B | 31.3B | 65.8 | 35.7 | 21.1 | 31.7 (256K) | 40 KiB |
| Gemma 4 26B-A4B (MoE) | 25.8B | 53.7 | 28.9 | 16.8 | 19.5 (256K) | 10 KiB |
| Gemma 4 12B | 12.0B | 25.4 | 13.9 | 8.3 | 10.5 (256K) | 8 KiB |
| Gemma 4 E4B | 8.0B | 17.0 | 9.4 | 5.6 | 7.7 (128K) | 16 KiB |
| Kimi K3 | 2.78T | 1,600 | 3,027 | 1,724 | 1,753 (1M) | 27 KiB |
| MiniMax M3 | 427.0B | 877 | 466 | 266 | 397 (1M) | 120 KiB |
| MiniMax M2.7 | 228.7B | 238 | 252 | 144 | 196 (200K) | 248 KiB |
| MiMo V2.6 Flash | 310.8B | 178 | 339 | 193 | 218 (1M) | 22.5 KiB |
| MiMo V2.6 Pro | 1.02T | 581 | 1,116 | 636 | 690 (1M) | 50 KiB |
| Mistral Medium 3.5 128B | 127.7B | 140 | 143 | 82.7 | 176 (256K) | 352 KiB |
| Nemotron 3 Nano 4B | 4.0B | 8.8 | 5.0 | 3.1 | 7.4 (256K) | 16 KiB |
| Nemotron 3 Nano 30B-A3B (MoE) | 31.6B | 65.2 | 34.9 | 20.1 | 21.7 (256K) | 6 KiB |
| Nemotron 3 Super 120B-A12B (MoE) | 123.6B | 254 | 135 | 77.2 | 79.3 (256K) | 8 KiB |
| Xing 4.0 29B-A4B (MoE) | 31.2B | 64.8 | 34.9 | 20.2 | 32.2 (256K) | 45 KiB |
| Muse Glimmer 30B | 29.8B | 61.7 | 33.1 | 19.1 | 20.8 (128K) | 13 KiB |
| MiniCPM5 2B | 2.5B | 6.0 | 3.6 | 2.4 | 7.8 (128K) | 42 KiB |
| Llama 3.1 8B | 8.0B | 18.1 | 10.3 | 6.6 | 23.1 (128K) | 128 KiB |
| Llama 3.1 70B | 70.6B | 148 | 80.0 | 47.0 | 88.2 (128K) | 320 KiB |
| Qwen3 8B | 8.2B | 18.5 | 10.7 | 6.8 | 11.8 (40K) | 144 KiB |
| Qwen3 30B-A3B (MoE) | 30.5B | 63.9 | 34.6 | 20.2 | 23.6 (40K) | 96 KiB |
| gpt-oss-20b | 20.9B | 14.8 | 23.5 | 13.7 | 16.8 (128K) | 24 KiB |
| gpt-oss-120b | 116.8B | 67.7 | 128 | 73.2 | 77.9 (128K) | 36 KiB |
| DeepSeek V3 / R1 (671B) | 684.5B | 706 | 746 | 425 | 437 (160K) | 68.6 KiB |

DeepSeek V4's KV figures are upper bounds (see above). Free to reuse under the MIT license; a link back is
appreciated.

## Speed

`src/speed.ts` estimates how many tokens per second one request generates: every token reads the active
weights and the whole KV cache once, so the ceiling is memory bandwidth divided by those bytes. Engines reach
55–75% of it with dense models and 30–50% with mixture-of-experts models, plus 0.5–1.5 ms of fixed cost per
token, and 0.8–2 ms more when cards in tensor parallel exchange results after every layer. That puts Llama 3.1 8B at Q4_K_M on an RTX 4090 at 96–143 tokens/s (about 130 measured) and
Qwen3-30B-A3B at 121–219 (about 170 measured), with a 512-token context. Online: https://modelvram.com/llm-speed-calculator/

## Use it as a library

```ts
import { WEIGHT_PRECISIONS, estimate, formatGib, loadFromHub } from './src/index.ts';

const spec = await loadFromHub('Qwen/Qwen3-14B'); // or one of PRESETS
const q4 = WEIGHT_PRECISIONS.find((p) => p.id === 'q4_k_m')!;
// context tokens, parallel requests, KV cache bits, overhead %
const result = estimate(spec, q4, 32_768, 1, 16, 10);
console.log(formatGib(result.total));
```

## Adding a model

```bash
npm run model-spec -- Qwen/Qwen3.8-27B
```

prints a preset read the same way the page reads the Hub, and flags attention layer types the parser
does not know yet.

## Tests

```bash
npm test
```

Node 22.18 or newer runs the TypeScript tests directly; there is nothing to install.

## License

MIT
