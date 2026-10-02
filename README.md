# LLM VRAM Calculator

**How much GPU memory does this LLM need?** Weights, KV cache and runtime overhead for any model on
Hugging Face, read from the model's own `config.json` and safetensors metadata, so new models work the
day they are uploaded.

**Use it online, free, in your browser: [modelvram.com](https://modelvram.com/llm-vram-calculator/)**

[![ModelVRAM: an illustrated Llama 3.1 8B Q4_K_M memory estimate showing weights, KV cache and overhead](assets/readme-hero.svg)](https://modelvram.com/llm-vram-calculator/)

This repository is the calculation core behind [modelvram.com](https://modelvram.com): dependency-free
TypeScript, 44 model presets and tests pinned to real checkpoint sizes and llama.cpp buffer logs
(the site itself tracks 64 models as of 2026-09-30).

## See the memory budget

![For Llama 3.1 8B Q4_K_M, estimated memory rises from 6.58 GiB at 8K context to 9.88 GiB at 32K. Weights stay fixed while the FP16 KV cache grows from 1 to 4 GiB.](assets/vram-breakdown.svg)

Illustrative estimates from this repository’s own formula, with one request and default overhead.
These are not hardware benchmarks. [Try your own model and context →](https://modelvram.com/llm-vram-calculator/)

**Reproduce this chart offline:** after cloning the repository, run `node examples/context-budget.mjs`
(Node 22.18+). No GPU, model download or account is needed. The [runnable example](examples/context-budget.mjs)
prints both memory budgets and all assumptions as JSON, so you can inspect where the extra 3.30 GiB comes from.

## Why trust it

The estimates are checked against **20 public measurements** (llama.cpp, koboldcpp and vLLM logs, each with
a source link). **Median absolute error across these checks: 0.8%; largest error: +13.7%.**
The checks mix KV-cache buffers, weight buffers and whole-card totals; the 0.8% median is not a
whole-card accuracy guarantee. See the separate results below, or the full table, sources and every miss:
**[modelvram.com/accuracy](https://modelvram.com/accuracy/)**.

- **KV cache matches llama.cpp to the byte in 8 of 8 logs**, including Gemma 4's split global/sliding cache
  and gpt-oss's sliding windows.
- **Weights: median 1.5% off.** GGUF files that keep more tensors at high precision are where it misses
  (Gemma 4 26B Q4_K_M: 7.0% low); the download size is the exact figure when you have the file.
- **Whole-card totals come out 5.6%–8.1% high**, on purpose: the default 0.5 GiB + 10% overhead is
  conservative for one llama.cpp request.
- **vLLM:** the KV cache pool is checked against 6 public vLLM startup logs, all within ±6%
  ([vLLM calculator](https://modelvram.com/vllm-memory-calculator/#vllm-logs)).
- **Fine-tuning** (site calculator, not in this repo): 15 published runs, median error 1.6%, largest 7.5%
  ([fine-tuning calculator](https://modelvram.com/fine-tuning-vram-calculator/)).
- **Prompt speed / time to first token** (site calculator, not in this repo): 32 public llama-bench runs,
  all within ±28%, median 10% ([speed calculator](https://modelvram.com/llm-speed-calculator/#prefill-calibration)).
- **Image and video file sizes** (site calculator, not in this repo): 6 ComfyUI load logs, all within 0.53%
  ([image and video calculator](https://modelvram.com/image-video-vram-calculator/#calibration)).

The same functions in this repo (`weightBytes`, `kvCacheBytes`, `estimate`) produce the predictions on
that page at build time.

## The formula

Everything is in [`src/vram.ts`](src/vram.ts). GB means GiB (1024³ bytes) throughout.

```text
total    = weights + KV cache + overhead

weights  = parameters × bits per weight ÷ 8
KV cache = full layers    × cells         × values per token per layer × KV bits ÷ 8
         + sliding layers × sliding cells × values per token per layer × KV bits ÷ 8
         + cells × indexer layers × indexer bytes per row  (layout-specific; see below)
overhead = 0.5 GiB + (weights + KV cache) × overhead %   (10% by default)
```

**Weights.** Parameters include every expert of an MoE model (Qwen3-30B-A3B loads 30.5B, not 3B).
Bits per weight: 32 / 16 / 8 for FP32, BF16, FP8; 4.25 for INT4 (AWQ/GPTQ); llama.cpp's whole-model
averages for GGUF (Q8_0 8.5, Q6_K 6.56, Q5_K_M 5.67, Q4_K_M 4.84, IQ4_XS 4.35, Q3_K_M 3.91, Q2_K 3.35,
IQ3_XXS 3.3). "As published" is the size of the root `.safetensors` files; without a file listing it sums
the stored dtypes, counting MXFP4 `U8` tensors as 4.25 bits.

**KV cache.**

- *Values per token per layer* = `KV heads × (head dim + V head dim)`, or for MLA (DeepSeek V3, Kimi) the
  latent `kv_lora_rank + qk_rope_head_dim` (576 for DeepSeek V3).
- *Cells* = `context × parallel requests`.
- *Sliding cells* = `min(cells, pad256(window × requests + 512))`, which is how llama.cpp sizes
  sliding-window layers (`src/llama-kv-cache-iswa.cpp`, 512 = default ubatch). gpt-oss: 768 cells.
- Linear-attention / Mamba / recurrent layers and layers that reuse an earlier cache (Gemma 4 E models) hold
  no context-growing cache and are left out.
- KV bits: 16 (FP16/BF16), 8 (FP8), 8.5 (Q8_0), 4.5 (Q4_0).
- Ordinary DeepSeek-style indexers store one FP8 key per row. GLM-5.3-Flash uses the
  source-checked `glm5-next-kpool` layout: `3 × indexDim × KV bits ÷ 8` bytes per row,
  for key, gate and pooled values at the selected K-cache precision. Pooling does not divide
  the allocated context rows. Its 512-value MLA and 384-value indexer rows are both aligned to
  GGML's 32-value blocks: Q8_0 uses 34 bytes per block; Q4_0 uses 18.
  This follows [llama.cpp 5fc4f3c](https://github.com/ggml-org/llama.cpp/blob/5fc4f3c8c7103ffd0b7ff5ee4855bcc78a3ed5cd/src/llama-memory-hybrid-idx.cpp#L53-L73),
  not a universal runtime contract or a GPU measurement. With FP16 and one request,
  the context-growing cache is 616 MiB at 32K and 2,464 MiB at 128K. Fixed recurrent state,
  MTP and scratch buffers are separate; quantized storage arithmetic does not establish
  kernel support (including the generic FP8 option).

**Multi-GPU.** A GPU group of 1, 2, 4 or 8 cards fits when each card holds
`(weights + KV cache) ÷ n × (1 + overhead %) + 0.5 GiB`.

**Speed** ([`src/speed.ts`](src/speed.ts)). Tokens/s ceiling = memory bandwidth ÷ (active weights + KV cache)
per token; engines reach 55–75% of it for dense models and 30–50% for MoE, plus 0.5–1.5 ms fixed cost per
token (0.8–2 ms more in tensor parallel). Llama 3.1 8B Q4_K_M on an RTX 4090: 96–143 tokens/s (about 130
measured). Online: [LLM speed calculator](https://modelvram.com/llm-speed-calculator/).

These are estimates: vLLM reserves a fixed share of GPU memory up front, and llama.cpp allocates the cache
for the full context at startup, so leave some headroom.

## Quick start

Requires Node 22.18 or newer (it runs the TypeScript directly). Nothing to install.

```bash
git clone https://github.com/159753a52/llm-vram-calculator.git
cd llm-vram-calculator
npm test
```

Estimate a model: save this as `example.mjs` in the repository root and run `node example.mjs`.

```js
import { GPUS, PRESETS, WEIGHT_PRECISIONS, estimate, formatGib, gpusNeeded, loadFromHub } from './src/index.ts';

const q4 = WEIGHT_PRECISIONS.find((p) => p.id === 'q4_k_m');

// Offline: one of the built-in presets
const llama = PRESETS.find((p) => p.id === 'meta-llama/Llama-3.1-8B-Instruct');
// context tokens, parallel requests, KV cache bits, overhead %
const r = estimate(llama, q4, 32_768, 1, 16, 10);
console.log(`weights ${formatGib(r.weights)}, KV ${formatGib(r.kvCache)}, overhead ${formatGib(r.overhead)}, total ${formatGib(r.total)}`);
console.log('RTX 3060 cards needed:', gpusNeeded(r, 10, GPUS.find((g) => g.name === 'RTX 3060').gib));

// Online: any public model on the Hugging Face Hub
const qwen = await loadFromHub('Qwen/Qwen3-14B');
console.log(formatGib(estimate(qwen, q4, 32_768, 1, 16, 10).total));
```

If `loadFromHub` cannot read a gated model's `config.json`, use an accessible repository
or run the offline preset example above. The calculation core does not sign in to Hugging Face.

```text
weights 4.52 GB, KV 4.00 GB, overhead 1.35 GB, total 9.88 GB
RTX 3060 cards needed: 1
15.2 GB
```

Generation speed:

```js
import { PRESETS, SPEED_GPUS, WEIGHT_PRECISIONS, formatRange, tokensPerSecond } from './src/index.ts';

const q4 = WEIGHT_PRECISIONS.find((p) => p.id === 'q4_k_m');
const llama = PRESETS.find((p) => p.id === 'meta-llama/Llama-3.1-8B-Instruct');
const rtx4090 = SPEED_GPUS.find((g) => g.name === 'RTX 4090');
console.log(formatRange(tokensPerSecond(llama, q4, 512, 16, rtx4090)), 'tokens/s'); // 96–143 tokens/s
```

Read a Hugging Face model into a preset (flags attention layer types the parser does not know yet):

```bash
npm run model-spec -- Qwen/Qwen3-14B
```

Regenerate the data tables below from the presets:

```bash
npm run export-data
```

## Coverage

| Case | What is modelled |
| --- | --- |
| Dense / GQA | `num_key_value_heads × head_dim`, K and V; plain attention matches llama.cpp to the byte. |
| MoE | Every expert counts toward weight memory; active parameters drive the speed estimate. |
| MLA (DeepSeek V3, GLM-5.3, Kimi K3) | Compressed latent `kv_lora_rank + qk_rope_head_dim`: about 69 KB per token in BF16 for DeepSeek V3. |
| Sliding-window layers | `layer_types`, MiMo's `hybrid_layer_pattern`, Limite's `global_layers`; cells sized as llama.cpp does. |
| Per-layer cache shapes | Gemma 4's global layers (4 × 512, K and V stored separately even with `attention_k_eq_v`) vs sliding (16 × 256); MiMo V2's 192-wide K / 128-wide V. |
| Linear attention / state-space | No context-growing cache: Qwen3-Next's `full_attention_interval`, Kimi's `linear_attn_config`, Nemotron-H's `hybrid_override_pattern`. |
| KDA recurrent layers (AliceAI 80B-A3B) | Fixed state excluded, so real serving memory can be higher. |
| Sparse-attention indexer (DeepSeek V3.2, GLM-5) | Its own FP8 key per token (`index_head_dim`), in every or only `full` layers; GLM-5.3-Flash instead stores three vectors in the selected K dtype, as described above. |
| KV sharing (Gemma 4 E models) | `num_kv_shared_layers` keep no cache of their own. |
| Cross-layer compression (DeepSeek V4) | Not modelled: the KV figure is flagged as an upper bound. |
| Multimodal checkpoints | Language model numbers read from `text_config`. |
| Storage formats | MXFP4 (gpt-oss), FP8, 4-bit experts in I8 (DeepSeek V4): "as published" uses the real file size. |

Weight precisions: as published, FP32, FP16/BF16, FP8/INT8, INT4 (AWQ/GPTQ), GGUF Q8_0, Q6_K, Q5_K_M,
Q4_K_M, IQ4_XS, Q3_K_M, IQ3_XXS, Q2_K. KV cache: FP16/BF16, FP8, Q8_0, Q4_0. GPUs: consumer RTX cards to
H200/B200, Apple Silicon and other unified-memory machines ([`src/speed.ts`](src/speed.ts)).

Not covered: training and fine-tuning memory, CPU offload, different K and V cache types
(`-ctk q8_0 -ctv q4_0`). Fine-tuning and MoE offload have their own calculators on
[modelvram.com](https://modelvram.com).

### Data

[`data/vram-requirements.csv`](data/vram-requirements.csv) and [`.json`](data/vram-requirements.json) list
weights and total GPU memory for all 44 presets at ten precisions (8,192-token FP16 cache, one request,
default overhead). A few rows, totals in GiB:

| Model | Parameters | As published | Q8_0 | Q4_K_M | Q4_K_M at max context | KV cache per token (FP16) |
|---|---|---|---|---|---|---|
| Llama 3.1 8B | 8.0B | 18.1 | 10.3 | 6.6 | 23.1 (128K) | 128 KiB |
| Llama 3.1 70B | 70.6B | 148 | 80.0 | 47.0 | 88.2 (128K) | 320 KiB |
| Qwen3 30B-A3B (MoE) | 30.5B | 63.9 | 34.6 | 20.2 | 23.6 (40K) | 96 KiB |
| Qwen3.6 27B | 27.8B | 58.0 | 31.3 | 18.3 | 35.3 (256K) | 64 KiB |
| Gemma 4 31B | 31.3B | 66.6 | 36.5 | 21.9 | 43.2 (256K) | 80 KiB |
| gpt-oss-20b | 20.9B | 14.8 | 23.5 | 13.7 | 16.8 (128K) | 24 KiB |
| gpt-oss-120b | 116.8B | 67.7 | 128 | 73.2 | 77.9 (128K) | 36 KiB |
| DeepSeek V3 / R1 (671B) | 684.5B | 706 | 746 | 425 | 437 (160K) | 68.6 KiB |

Per-model pages with every quantization and the GPUs that fit, e.g.
[Gemma 4 31B](https://modelvram.com/llm-vram-calculator/gemma-4-31b-it/) and
[DeepSeek V4.1 Flash](https://modelvram.com/llm-vram-calculator/deepseek-v4.1-flash/).

## Contributing

**Real measurements are the most useful contribution.** Ran a model with llama.cpp, vLLM, Ollama or
another engine? [Submit a measurement](https://github.com/159753a52/llm-vram-calculator/issues/new?template=measurement.yml):
model, file/quant, engine version, GPU, context, parallel slots, KV type and the log lines as printed
(llama.cpp's `KV buffer size` / `model buffer size`, vLLM's KV cache lines, or an `nvidia-smi` reading).
Each one is checked by hand and added to the [accuracy table](https://modelvram.com/accuracy/).

Code changes: a model the parser gets wrong, a new attention layout, a missing quant. Run
`npm run model-spec -- <hf-id>` to see what the parser reads, add a test in [`test/`](test/), and make sure
`npm test` passes. If you change a preset, run `npm run export-data` and commit the regenerated `data/`.

## License

[MIT](LICENSE). The data files are free to reuse; a link back to [modelvram.com](https://modelvram.com) is
appreciated.
