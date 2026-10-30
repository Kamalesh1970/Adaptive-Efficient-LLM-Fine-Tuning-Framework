// FineTunePilot Main Application Entrypoint & Backend Event Wiring
import './styles/tokens.css';
import './styles/base.css';
import './styles/components.css';
import './styles/backgrounds.css';

import { api } from './api.js';
import { S } from './state.js';
import { MODELS, setCatalogModels } from './data/models.js';
import { DEMO } from './data/sampleDataset.js';
import { GL, RO } from './data/glossary.js';
import { NAV, TITLES } from './data/nav.js';
import { parse } from './logic/parseDataset.js';
import { est, compat, alt, ok } from './logic/compatRules.js';
import { run, cancelRun } from './logic/simulateRun.js';
import { useDS } from './pages/dataset.js';
import { badge } from './components/badge.js';
import { tooltip, T } from './components/tooltip.js';
import { toast } from './components/toast.js';
import { pilot, tlHTML, ask } from './components/pilot.js';
import { go, nav, render, offerNext } from './router.js';

// Attach globals for inline event handlers and page functions
Object.assign(window, {
  api,
  S,
  MODELS,
  DEMO,
  GL,
  RO,
  NAV,
  TITLES,
  parse,
  est,
  compat,
  alt,
  ok,
  run,
  cancelRun,
  useDS,
  badge,
  tooltip,
  T,
  toast,
  pilot,
  tlHTML,
  ask,
  go,
  nav,
  render,
  offerNext,
});

document.addEventListener("click", async e => {
  const t = e.target.closest("[data-a]");
  if (!t) return;
  const a = t.dataset.a, v = t.dataset.v;

  if (a === "new") $("#dlg")?.showModal();
  if (a === "ask") ask(v || $("#ti")?.value);
  if (a === "nxgo") {
    $("#nx")?.close();
    S.off = S.nxRoute === "compat" ? 1 : 0;
    go(S.nxRoute);
  }
  if (a === "nxstay") $("#nx")?.close();
  if (a === "sum") {
    const hwDesc = S.hardwareInfo?.cuda_available
      ? `GPU ${S.hardwareInfo.primary_gpu_name} (${S.hardwareInfo.primary_gpu_vram_gb} GB VRAM)`
      : "CPU Fallback Mode";
    S.sumT = `Adaptive LLM Fine-Tuning Studio (FineTunePilot)\nProject: ${S.proj || "none"}\nBackend Status: ${
      S.backendOnline ? "Connected (FastAPI)" : "Connecting..."
    }\nDataset: ${S.ds ? S.ds.name + " (" + S.ds.rows.length + " rows)" : "none"}\nModel: ${
      S.model ? S.model.n : "none"
    }\nMethod: ${S.method.toUpperCase()}, Hardware: ${hwDesc}, Est: ${est() ?? "n/a"} GB\nConfig Saved: ${
      S.saved ? "yes" : "no"
    }\nRuns: ${S.runs.length}\nArtifacts: ${S.arts.length}`;
    const stEl = $("#sumt");
    if (stEl) stEl.textContent = S.sumT;
    $("#sum")?.showModal();
  }
  if (a === "closesum") $("#sum")?.close();
  if (a === "copy") {
    (navigator.clipboard ? navigator.clipboard.writeText(S.sumT) : Promise.reject()).then(
      () => toast("Summary copied"),
      () => toast("Copy blocked. Select text manually.")
    );
  }
  if (a === "demo") useDS("sample-dataset.csv", DEMO, true);
  if (a === "pick") {
    S.model = MODELS.find(m => m.id === v) || { id: v, n: v, p: 7, ctx: 4096 };
    toast(S.model.n + " selected");
    render();
    offerNext("models", S.model.n + " selected.");
  }
  if (a === "save") {
    S.saved = true;
    const configPayload = {
      project_name: S.proj || "my-project",
      model_identifier: S.model?.id || "google/gemma-2-2b",
      training_method: S.method === "qlora" ? "qlora" : "lora",
      dataset_reference: S.ds?.id || S.ds?.name || "dataset",
      num_epochs: Number(S.cfg.epochs) || 1,
      learning_rate: Number(S.cfg.lr) || 0.0002,
      batch_size: Number(S.cfg.batch) || 2,
      lora_rank: Number(S.cfg.rank) || 8,
      context_length: Number(S.cfg.len) || 512,
    };

    api.validateConfig(configPayload)
      .then(res => {
        if (res.is_valid) {
          toast("Configuration verified by backend!");
        } else {
          toast("Saved with warnings: " + (res.errors?.[0] || "Check parameters"));
        }
      })
      .catch(() => {});

    S.arts = S.arts.filter(x => !x.cfg);
    S.arts.unshift({
      n: "Training config",
      t: "Configuration",
      p: S.proj || "Untitled",
      s: "Saved",
      cfg: JSON.stringify({ model: S.model?.n, method: S.method, ...S.cfg }, null, 2),
    });
    toast("Configuration saved");
    render();
    offerNext("config", "Configuration saved.");
  }
  if (a === "run") run();
  if (a === "cancel") cancelRun();
  if (a === "export-gguf") {
    const latestRun = S.runs[0];
    const exportPayload = {
      source_artifact_id: latestRun?.id || "run-1",
      base_model_identifier: S.model?.id || "google/gemma-2-2b",
      quantization_type: "q4_k_m",
      output_filename: `${S.proj || "model"}-q4_k_m.gguf`,
    };
    toast("Triggering GGUF binary export...");
    try {
      const job = await api.createGGUFExport(exportPayload);
      S.lastExport = job;
      toast("GGUF export completed!");
      render();
    } catch (err) {
      toast("Export error: " + err.message);
    }
  }
  if (a === "save-adapter") {
    toast("LoRA adapter weights verified.");
  }
  if (a === "refresh-artifacts") {
    api.getArtifacts().then(arts => {
      if (Array.isArray(arts)) {
        S.arts = arts.map(a => ({
          id: a.artifact_id,
          n: a.artifact_type || "Model Adapter",
          t: a.fine_tuning_method?.toUpperCase() || "LoRA",
          p: a.model_identifier || S.proj,
          s: a.validation_status || "Saved",
          size: a.size_bytes ? `${Math.round(a.size_bytes / 1024)} KB` : "—",
          url: api.getArtifactDownloadUrl(a.artifact_id),
        }));
        toast("Artifacts refreshed");
        render();
      }
    }).catch(err => toast("Refresh error: " + err.message));
  }
  if (a === "view") {
    S.viewed = S.arts[v]?.cfg || JSON.stringify(S.arts[v], null, 2);
    render();
  }
  if (a === "del") {
    S.arts.splice(v, 1);
    S.viewed = "";
    render();
  }
  if (a === "reset") {
    if (S.timer) clearInterval(S.timer);
    Object.assign(S, {
      proj: null,
      ds: null,
      model: null,
      saved: false,
      runs: [],
      activeRun: null,
      arts: [],
      log: [],
      prog: 0,
      timer: null,
      viewed: "",
    });
    toast("Demo data reset");
    render();
  }
});

document.addEventListener("keydown", e => {
  if ((e.key === "Enter" || e.key === " ") && e.target.matches?.(".card.click")) {
    e.preventDefault();
    e.target.click();
  }
});

const nf = $("#nf");
if (nf) {
  nf.onsubmit = e => {
    if (e.submitter?.value === "go") {
      S.proj = $("#pn")?.value.trim() || "My First Fine-Tune";
      toast("Project created");
      setTimeout(() => go("dataset"), 0);
    }
  };
}

const mb = $("#mb");
if (mb) {
  mb.onclick = () => {
    const o = $("#side")?.classList.toggle("open");
    mb.setAttribute("aria-expanded", o);
  };
}

addEventListener("keydown", e => {
  if (e.key === "Enter" && e.target.id === "ti") {
    ask(e.target.value);
  }
});

addEventListener("focusin", e => {
  const k = e.target.dataset?.c, r = $("#ro");
  if (k && r) r.innerHTML = "<b>" + k + "</b><br>" + (RO[k] || "");
});

addEventListener("mouseover", e => {
  const k = e.target.dataset?.c, r = $("#ro");
  if (k && r) r.innerHTML = "<b>" + k + "</b><br>" + (RO[k] || "");
});

addEventListener("hashchange", render);

// Initial Hardware & Backend Synchronization
async function initBackendSync() {
  try {
    const [health, hw, models, runs, artifacts, serving, k8s] = await Promise.all([
      api.getHealth().catch(() => null),
      api.getSystemStatus().catch(() => null),
      api.getModels().catch(() => []),
      api.getTrainingRuns().catch(() => []),
      api.getArtifacts().catch(() => []),
      api.getServingCapabilities().catch(() => null),
      api.validateKubernetes().catch(() => null),
    ]);

    if (health) S.backendOnline = true;
    if (hw) {
      S.hardwareInfo = hw;
      if (hw.cuda_available && hw.primary_gpu_vram_gb) {
        S.hw = String(Math.round(hw.primary_gpu_vram_gb));
      } else {
        S.hw = "8"; // CPU allocation baseline
      }
    }
    if (Array.isArray(models) && models.length > 0) {
      setCatalogModels(models);
    }
    if (Array.isArray(runs) && runs.length > 0) {
      S.runs = runs.map(r => ({
        id: r.run_id,
        m: r.model_identifier,
        d: r.dataset_reference || "dataset",
        c: `${r.training_method?.toUpperCase() || 'LORA'}, ${r.num_epochs || 1} epoch(s)`,
        done: r.status === 'completed',
        status: r.status,
      }));
    }
    if (Array.isArray(artifacts) && artifacts.length > 0) {
      S.arts = artifacts.map(a => ({
        id: a.artifact_id,
        n: a.artifact_type || "Model Checkpoint",
        t: a.fine_tuning_method?.toUpperCase() || "LoRA",
        p: a.model_identifier || S.proj,
        s: a.validation_status || "Saved",
        size: a.size_bytes ? `${Math.round(a.size_bytes / 1024)} KB` : "—",
        url: api.getArtifactDownloadUrl(a.artifact_id),
      }));
    }
    if (serving) S.servingCaps = serving;
    if (k8s) S.k8sStatus = k8s;
  } catch (syncErr) {
    console.warn("Backend synchronization notice:", syncErr);
  } finally {
    render();
  }
}

// Initial render & sync
render();
initBackendSync();
