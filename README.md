# LLM VRAM Calculator

Estimate how much GPU memory a large language model needs for inference (weights, KV cache and runtime
overhead) for any model on Hugging Face.

**Use it online: https://toolsite-static.pages.dev/llm-vram-calculator/**

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
| Cross-layer cache sharing (DeepSeek V4) | Not modelled; the model carries a note that its KV figure is an upper bound. |

"As published" is the size of the `.safetensors` files in the repository root (Mistral's duplicate
`consolidated.safetensors` excluded), so it matches the download whatever the storage format: MXFP4 in
gpt-oss, 4-bit experts in DeepSeek V4 and MiMo V2.6. Without the file listing it adds up the stored
dtypes, which gives 12.82 GB for gpt-oss-20b and 641.3 GB for DeepSeek V3.
(GB means GiB throughout, the unit GPU memory is sold in.)

The site has a page per model with every quantization, the cache at long context and the GPUs that fit,
for example [Gemma 4 31B](https://toolsite-static.pages.dev/llm-vram-calculator/gemma-4-31b-it/) and
[DeepSeek V4.1 Flash](https://toolsite-static.pages.dev/llm-vram-calculator/deepseek-v4.1-flash/).

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
