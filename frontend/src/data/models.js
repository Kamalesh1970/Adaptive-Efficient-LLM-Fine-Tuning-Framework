// Foundation model catalog (pre-populated with verified catalog & dynamic backend sync)
export const DEFAULT_MODELS = [
  { id: "google/gemma-2-2b", n: "Gemma-2 2B", p: 2.0, t: "Text generation, instruction tuning", ctx: 8192, note: "Lightweight, fits on 4GB-8GB VRAM with LoRA/QLoRA." },
  { id: "microsoft/Phi-3-mini-4k-instruct", n: "Phi-3 Mini 3.8B", p: 3.8, t: "Reasoning, code, instruction tuning", ctx: 4096, note: "High performance small model." },
  { id: "meta-llama/Meta-Llama-3-8B", n: "Llama-3 8B", p: 8.0, t: "General instruction tuning, chat", ctx: 8192, note: "Standard 8B foundation model, requires 8GB+ GPU with QLoRA." },
  { id: "mistralai/Mistral-7B-v0.3", n: "Mistral-7B v0.3", p: 7.2, t: "Text generation, coding, chat", ctx: 32768, note: "Strong 7B foundation model with sliding window attention." },
  { id: "Qwen/Qwen2.5-7B", n: "Qwen-2.5 7B", p: 7.6, t: "Multilingual, coding, math", ctx: 32768, note: "Excellent general-purpose capability." }
];

export let MODELS = [...DEFAULT_MODELS];

export function setCatalogModels(backendModels) {
  if (Array.isArray(backendModels) && backendModels.length > 0) {
    MODELS = backendModels.map(m => {
      let pSize = 7.0;
      if (m.parameter_count) {
        const match = m.parameter_count.match(/([0-9.]+)/);
        if (match) pSize = parseFloat(match[1]);
      }
      return {
        id: m.id,
        n: m.name || m.id,
        p: pSize,
        t: m.task_type || "Causal LM fine-tuning",
        ctx: m.context_length || 4096,
        note: m.compatibility_notes || (m.is_gated ? "Gated model (requires HF token)." : "Open weights model."),
        is_gated: !!m.is_gated,
        supported_methods: m.supported_methods || ["lora", "qlora"],
      };
    });
    if (typeof window !== 'undefined') {
      window.MODELS = MODELS;
    }
  }
}

if (typeof window !== 'undefined') {
  window.MODELS = MODELS;
  window.setCatalogModels = setCatalogModels;
}
