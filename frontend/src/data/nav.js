// Sidebar groups and page titles
export const NAV = [
  ["Overview", [
    ["dashboard", "Dashboard"],
  ]],
  ["Data & Models", [
    ["dataset", "Dataset Management"],
    ["models", "Model Hub"],
    ["compat", "Compatibility & Preflight"],
  ]],
  ["Training & Execution", [
    ["config", "Training Configuration"],
    ["runs", "Training Runs & Logs"],
    ["eval", "Model Evaluation"],
    ["quant", "Quantization Jobs"],
  ]],
  ["Artifacts & Serving", [
    ["artifacts", "Artifact Registry"],
    ["serving", "Model Serving & vLLM"],
  ]],
  ["Infrastructure", [
    ["distributed", "Multi-GPU & Distributed"],
    ["settings", "Settings & Diagnostics"],
  ]],
];

export const TITLES = {};
NAV.forEach(g => g[1].forEach(p => (TITLES[p[0]] = p[1])));

if (typeof window !== 'undefined') {
  window.NAV = NAV;
  window.TITLES = TITLES;
}
