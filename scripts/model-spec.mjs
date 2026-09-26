// Prints calculator presets for Hugging Face models, read the same way the page reads them:
//
//   npm run model-spec -- Qwen/Qwen3.8-27B deepseek-ai/DeepSeek-V4.1-Flash
//
// Paste the output into src/presets.ts and give each a readable name.
// It also lists config fields the parser ignores, so a new kind of attention layer gets noticed
// before the page shows a wrong KV cache.
import { loadFromHub } from '../src/hf.ts';

const HANDLED_LAYER_TYPES = /^(full_attention|sliding_attention|attention|deepseek_sparse_attention|.*linear.*|.*mamba.*|.*recurrent.*)$/;

for (const id of process.argv.slice(2)) {
  try {
    const spec = await loadFromHub(id);
    const config = await fetch(`https://huggingface.co/${id}/resolve/main/config.json`).then((res) => res.json());
    const text = config.text_config ?? config;
    const layerTypes = [...new Set(text.layer_types ?? [])];
    const unknown = layerTypes.filter((type) => !HANDLED_LAYER_TYPES.test(type));
    const notes = [
      `model_type=${text.model_type ?? config.model_type}`,
      layerTypes.length ? `layer_types=${layerTypes.join('/')}` : '',
      text.num_experts ?? text.n_routed_experts ? `experts=${text.num_experts ?? text.n_routed_experts}` : '',
      unknown.length ? `UNHANDLED layer types: ${unknown.join(', ')}` : '',
    ].filter(Boolean);
    console.log(`  // ${notes.join('; ')}`);
    console.log(`  ${JSON.stringify(spec, null, 2).replace(/\n/g, '\n  ')},`);
  } catch (error) {
    console.log(`  // ${id}: ${error.message}`);
  }
}
