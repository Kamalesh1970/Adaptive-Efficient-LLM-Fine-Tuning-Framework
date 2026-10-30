// "Ask the pilot": rule-based and backend telemetry assistance
import { GL } from '../data/glossary.js';
import { TITLES } from '../data/nav.js';
import { MODELS } from '../data/models.js';
import { S } from '../state.js';
import { est, compat } from '../logic/compatRules.js';
import { go } from '../router.js';

const $ = s => document.querySelector(s);

export function pilot(c) {
  const [w, ...r] = c.trim().toLowerCase().split(/\s+/);
  const a = r.join(" ");

  if (w === "help") {
    return "Commands: status, check, hardware, models, runs, explain <term>, goto <page>, clear. Terms: " + Object.keys(GL).join(", ");
  }
  if (w === "status") {
    const hwDesc = S.hardwareInfo?.cuda_available
      ? `GPU ${S.hardwareInfo.primary_gpu_name} (${S.hardwareInfo.primary_gpu_vram_gb} GB VRAM)`
      : "CPU Mode (Active Fallback)";
    return `Project: ${S.proj || "none"} | Dataset: ${S.ds ? S.ds.name : "none"} | Model: ${
      S.model ? S.model.n : "none"
    } | Method: ${S.method.toUpperCase()} | Hardware: ${hwDesc} | Declared memory: ${S.hw} GB | Backend: ${
      S.backendOnline ? "Online" : "Connecting..."
    }`;
  }
  if (w === "hardware") {
    if (!S.hardwareInfo) return "Hardware probe in progress...";
    return S.hardwareInfo.cuda_available
      ? `GPU detected: ${S.hardwareInfo.primary_gpu_name} with ${S.hardwareInfo.primary_gpu_vram_gb} GB VRAM. PyTorch: ${S.hardwareInfo.pytorch_version}`
      : `Host operating in CPU fallback mode (CUDA runtime unavailable). PyTorch: ${S.hardwareInfo.pytorch_version}. Training runs safely on CPU.`;
  }
  if (w === "models") {
    return "Available models:\n" + MODELS.map(m => ` - ${m.n} (${m.p}B parameters, ctx ${m.ctx})`).join("\n");
  }
  if (w === "runs") {
    if (!S.runs.length) return "No runs executed yet. Save a config and launch on the Training Runs page.";
    return "Recent runs:\n" + S.runs.slice(0, 5).map(x => ` - ${x.id}: ${x.m} [${x.status || (x.done ? 'COMPLETED' : 'RUNNING')}]`).join("\n");
  }
  if (w === "check") {
    return compat().map(x => `[${x[0].toUpperCase()}] ${x[1]}: ${x[2]}`).join("\n");
  }
  if (w === "explain") {
    return GL[a] || "No entry for '" + a + "'. Try: " + Object.keys(GL).join(", ");
  }
  if (w === "goto") {
    if (TITLES[a]) {
      go(a);
      return "Opening " + TITLES[a];
    }
    return "Pages: " + Object.keys(TITLES).join(", ");
  }
  return 'Unknown command. Type "help" to see available options.';
}

export function tlHTML() {
  return S.tlog.map(t => `<div class="b ${t.startsWith("$ ") ? "me" : ""}">${t.replace(/^\$ /, "").replace(/&/g, "&amp;").replace(/</g, "&lt;")}</div>`).join("");
}

export function ask(v) {
  v = (v || "").trim();
  if (!v) return;
  const i = $("#ti");
  if (i) i.value = "";
  if (v === "clear") S.tlog = [];
  else S.tlog.push("$ " + v, pilot(v));
  const l = $("#tl");
  if (l) {
    l.innerHTML = tlHTML();
    l.scrollTop = 1e9;
  }
}

if (typeof window !== 'undefined') {
  window.pilot = pilot;
  window.tlHTML = tlHTML;
  window.ask = ask;
}
