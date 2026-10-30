// FineTunePilot Main Application Entrypoint & Backend Event Wiring
import './styles/tokens.css';
import './styles/base.css';
import './styles/components.css';
import './styles/backgrounds.css';

import { $ } from './utils.js';
import { api } from './api.js';
import { S } from './state.js';
import { MODELS, setCatalogModels } from './data/models.js';
import { DEMO } from './data/sampleDataset.js';
import { GL, RO } from './data/glossary.js';
import { NAV, TITLES } from './data/nav.js';
import { parse } from './logic/parseDataset.js';
import { est, compat, alt, ok } from './logic/compatRules.js';
import { run, cancelRun } from './logic/simulateRun.js';
import { badge } from './components/badge.js';
import { tooltip, T } from './components/tooltip.js';
import { toast } from './components/toast.js';
import { pilot, tlHTML, ask } from './components/pilot.js';
import { go, nav, render, offerNext } from './router.js';

// Import all page modules to register their routes into P
import { P, hdr, next } from './pages/shared.js';
import './pages/dashboard.js';
import './pages/dataset.js';
import './pages/models.js';
import './pages/compat.js';
import './pages/config.js';
import './pages/runs.js';
import './pages/eval.js';
import './pages/quant.js';
import './pages/artifacts.js';
import './pages/serving.js';
import './pages/distributed.js';
import './pages/settings.js';
import { useDS } from './pages/dataset.js';

// Attach globals for inline event handlers and page functions
Object.assign(window, {
  $,
  P,
  hdr,
  next,
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

  // General & Modal Controls
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
      S.model ? S.model.n || S.model.id : "none"
    }\nMethod: ${S.method.toUpperCase()}, Hardware: ${hwDesc}, Est: ${est() ?? "n/a"} GB\nConfig Saved: ${
      S.saved ? "yes" : "no"
    }\nRuns: ${S.runs.length}\nArtifacts: ${S.arts.length}\nServing Instances: ${S.servingInstances.length}`;
    const stEl = $("#sumt");
    if (stEl) stEl.textContent = S.sumT;
    $("#sum")?.showModal();
  }
  if (a === "closesum") $("#sum")?.close();
  if (a === "copy") {
    (navigator.clipboard ? navigator.clipboard.writeText(S.sumT) : Promise.reject()).then(
      () => toast("Summary copied to clipboard"),
      () => toast("Copy blocked. Select text manually.")
    );
  }

  // Dataset Actions
  if (a === "demo") useDS("sample-dataset.csv", DEMO, true);
  if (a === "refresh-datasets") {
    try {
      const list = await api.getDatasets();
      if (Array.isArray(list)) S.datasetsList = list;
      toast("Server dataset catalog refreshed");
      render();
    } catch (err) {
      toast("Error loading datasets: " + err.message);
    }
  }
  if (a === "select-server-dataset") {
    const target = S.datasetsList.find(d => (d.dataset_id || d.id) === v);
    if (target) {
      try {
        const details = await api.getDataset(v);
        S.ds = {
          id: details.dataset_id || details.id,
          name: details.filename || details.name || v,
          cols: details.columns || details.detected_columns || ["instruction", "response"],
          rows: details.sample_rows || [{ instruction: "Loaded from server dataset repository", response: "OK" }],
          miss: [],
          dup: 0,
          issues: 0,
          schemaType: details.schema_type || "instruction-response",
        };
        toast(`Selected dataset ${S.ds.name}`);
        render();
        offerNext("dataset", `Dataset ${S.ds.name} loaded.`);
      } catch (err) {
        toast("Failed to load dataset details: " + err.message);
      }
    }
  }
  if (a === "inspect-splits") {
    try {
      const splits = await api.getDatasetSplits(v);
      S.activeDatasetSplits = splits;
      toast("Dataset splits loaded");
      render();
    } catch (err) {
      toast("Splits error: " + err.message);
    }
  }
  if (a === "close-splits") {
    S.activeDatasetSplits = null;
    render();
  }
  if (a === "validate-server-dataset") {
    try {
      toast("Validating dataset on backend...");
      const rep = await api.validateDataset(v);
      S.datasetValidationReport = rep;
      toast("Backend dataset validation completed");
      render();
    } catch (err) {
      toast("Validation error: " + err.message);
    }
  }
  if (a === "close-val-report") {
    S.datasetValidationReport = null;
    render();
  }

  // Model Selection & Validation
  if (a === "pick") {
    S.model = MODELS.find(m => m.id === v) || S.modelsList.find(m => (m.model_id || m.id) === v) || { id: v, n: v, p: 7, ctx: 4096 };
    toast((S.model.n || S.model.id) + " selected");
    render();
    offerNext("models", (S.model.n || S.model.id) + " selected.");
  }
  if (a === "validate-model-api") {
    try {
      toast("Validating model compatibility with backend...");
      const rep = await api.validateModel(v, S.method);
      S.modelValidation = rep;
      toast("Model validation verified!");
      render();
    } catch (err) {
      toast("Model validation failed: " + err.message);
    }
  }

  // Compatibility & Preflight
  if (a === "trigger-preflight") {
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
    try {
      toast("Running preflight checks on backend...");
      const rep = await api.runPreflight(configPayload);
      S.preflightReport = rep;
      toast("Preflight check complete!");
      render();
    } catch (err) {
      toast("Preflight error: " + err.message);
    }
  }
  if (a === "trigger-preview") {
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
    try {
      toast("Generating Run Plan preview...");
      const plan = await api.previewPlan(configPayload);
      S.runPlan = plan;
      toast("Run Plan preview ready");
      render();
    } catch (err) {
      toast("Preview error: " + err.message);
    }
  }

  // Configuration Actions
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
      cfg: JSON.stringify({ model: S.model?.n || S.model?.id, method: S.method, ...S.cfg }, null, 2),
    });
    toast("Configuration saved");
    render();
    offerNext("config", "Configuration saved.");
  }
  if (a === "export-yaml") {
    try {
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
      const res = await api.exportConfigYaml(configPayload);
      S.yamlPreview = res.yaml_content;
      toast("YAML serialized successfully");
      render();
    } catch (err) {
      toast("YAML export error: " + err.message);
    }
  }
  if (a === "close-yaml") {
    S.yamlPreview = null;
    render();
  }
  if (a === "reset-config") {
    try {
      const defaults = await api.resetConfig();
      if (defaults) {
        S.cfg.epochs = defaults.num_epochs || 1;
        S.cfg.lr = String(defaults.learning_rate || "0.0002");
        S.cfg.batch = defaults.batch_size || 2;
        S.cfg.rank = defaults.lora_rank || 8;
        S.cfg.len = defaults.context_length || 512;
      }
      toast("Configuration reset to defaults");
      render();
    } catch (err) {
      toast("Reset error: " + err.message);
    }
  }

  // Training Execution & Monitoring
  if (a === "run") run();
  if (a === "cancel") cancelRun();
  if (a === "refresh-runs") {
    try {
      const runs = await api.getTrainingRuns();
      if (Array.isArray(runs)) {
        S.runs = runs.map(r => ({
          id: r.run_id,
          m: r.model_identifier,
          d: r.dataset_reference || "dataset",
          c: `${r.training_method?.toUpperCase() || 'LORA'}, ${r.num_epochs || 1} epoch(s)`,
          done: r.status === 'completed',
          status: r.status,
          prog: r.progress_percent,
        }));
      }
      toast("Training runs updated");
      render();
    } catch (err) {
      toast("Error loading runs: " + err.message);
    }
  }
  if (a === "inspect-run") {
    try {
      const [runDetails, events, checkpoints] = await Promise.all([
        api.getTrainingRun(v).catch(() => null),
        api.getTrainingEvents(v).catch(() => []),
        api.getCheckpoints(v).catch(() => []),
      ]);
      if (runDetails) S.activeRun = runDetails;
      S.activeEvents = Array.isArray(events) ? events : [];
      S.activeCheckpoints = Array.isArray(checkpoints) ? checkpoints : [];
      toast(`Inspecting run ${v}`);
      render();
    } catch (err) {
      toast("Error inspecting run: " + err.message);
    }
  }
  if (a === "cancel-run-item") {
    try {
      await api.cancelTrainingRun(v);
      toast(`Run ${v} cancellation requested`);
      const runs = await api.getTrainingRuns();
      if (Array.isArray(runs)) S.runs = runs;
      render();
    } catch (err) {
      toast("Cancellation error: " + err.message);
    }
  }

  // Model Evaluation Actions
  if (a === "start-evaluation") {
    const latestRun = S.runs.find(x => x.done || x.status === "completed") || S.runs[0];
    const evalPayload = {
      model_identifier: latestRun?.m || S.model?.id || "google/gemma-2-2b",
      dataset_reference: S.ds?.id || S.ds?.name || "validation_set",
      batch_size: Number($("#ev_batch")?.value) || 2,
    };
    try {
      toast("Launching model evaluation benchmark...");
      const ev = await api.createEvaluation(evalPayload);
      toast("Evaluation launched: " + (ev.evaluation_id || "job"));
      const evals = await api.getEvaluations();
      if (Array.isArray(evals)) S.evals = evals;
      render();
    } catch (err) {
      toast("Evaluation error: " + err.message);
    }
  }
  if (a === "refresh-evals") {
    try {
      const evals = await api.getEvaluations();
      if (Array.isArray(evals)) S.evals = evals;
      toast("Evaluation history refreshed");
      render();
    } catch (err) {
      toast("Error loading evals: " + err.message);
    }
  }
  if (a === "view-eval-results") {
    try {
      const results = await api.getEvaluationResults(v);
      S.activeEvalResults = results;
      toast("Evaluation results loaded");
      render();
    } catch (err) {
      toast("Results error: " + err.message);
    }
  }
  if (a === "close-eval-results") {
    S.activeEvalResults = null;
    render();
  }

  // Quantization Actions
  if (a === "submit-quantization") {
    const qbits = $("#q_bits")?.value || "4bit";
    const qout = $("#q_out")?.value || "model-quantized";
    const payload = {
      source_model: S.model?.id || "google/gemma-2-2b",
      quantization_type: qbits,
      output_artifact_id: qout,
    };
    try {
      toast("Launching post-training quantization job...");
      const qjob = await api.createQuantizationJob(payload);
      toast("Quantization job dispatched!");
      const qjobs = await api.getQuantizationJobs();
      if (Array.isArray(qjobs)) S.quantJobs = qjobs;
      render();
    } catch (err) {
      toast("Quantization error: " + err.message);
    }
  }
  if (a === "refresh-quant-jobs") {
    try {
      const qjobs = await api.getQuantizationJobs();
      if (Array.isArray(qjobs)) S.quantJobs = qjobs;
      toast("Quantization jobs refreshed");
      render();
    } catch (err) {
      toast("Error: " + err.message);
    }
  }
  if (a === "cancel-quant-job") {
    try {
      await api.cancelQuantizationJob(v);
      toast("Quantization job cancelled");
      const qjobs = await api.getQuantizationJobs();
      if (Array.isArray(qjobs)) S.quantJobs = qjobs;
      render();
    } catch (err) {
      toast("Cancel error: " + err.message);
    }
  }

  // GGUF Export Actions
  if (a === "export-gguf") {
    const latestRun = S.runs[0];
    const qtype = $("#gguf_qtype")?.value || "q4_k_m";
    const outName = $("#gguf_outname")?.value || `${S.proj || "model"}-${qtype}.gguf`;
    const exportPayload = {
      source_artifact_id: latestRun?.id || "run-1",
      base_model_identifier: S.model?.id || "google/gemma-2-2b",
      quantization_type: qtype,
      output_filename: outName,
    };
    toast("Triggering GGUF binary export...");
    try {
      const job = await api.createGGUFExport(exportPayload);
      S.lastExport = job;
      toast("GGUF export completed!");
      api.getGGUFExports().then(list => {
        if (Array.isArray(list)) S.ggufExports = list;
        render();
      }).catch(() => {});
      render();
    } catch (err) {
      toast("Export error: " + err.message);
    }
  }
  if (a === "refresh-exports") {
    try {
      const list = await api.getGGUFExports();
      if (Array.isArray(list)) S.ggufExports = list;
      toast("Export records updated");
      render();
    } catch (err) {
      toast("Exports error: " + err.message);
    }
  }

  // Artifact Registry Actions
  if (a === "refresh-artifacts") {
    try {
      const arts = await api.getArtifacts();
      if (Array.isArray(arts)) {
        S.arts = arts.map(art => ({
          id: art.artifact_id,
          n: art.artifact_type || "Model Adapter",
          t: art.fine_tuning_method?.toUpperCase() || "LoRA",
          p: art.model_identifier || S.proj,
          s: art.validation_status || "Saved",
          size: art.size_bytes ? `${Math.round(art.size_bytes / 1024)} KB` : "—",
          url: api.getArtifactDownloadUrl(art.artifact_id),
        }));
      }
      toast("Artifacts catalog updated");
      render();
    } catch (err) {
      toast("Error: " + err.message);
    }
  }
  if (a === "validate-artifact-item") {
    try {
      toast("Validating artifact integrity and checksum...");
      const rep = await api.validateArtifact(v);
      toast("Artifact validation: " + (rep.is_valid ? "VALID" : "INVALID"));
      render();
    } catch (err) {
      toast("Validation error: " + err.message);
    }
  }
  if (a === "export-artifact-item") {
    try {
      toast("Exporting artifact to external storage...");
      await api.exportArtifact(v);
      toast("Artifact export completed!");
      render();
    } catch (err) {
      toast("Export error: " + err.message);
    }
  }
  if (a === "view") {
    S.viewed = S.arts[v]?.cfg || JSON.stringify(S.arts[v], null, 2);
    render();
  }

  // Inference & Model Serving Actions
  if (a === "launch-serving") {
    const srvModel = $("#srv_model")?.value || S.model?.id || "google/gemma-2-2b";
    const srvPort = Number($("#srv_port")?.value) || 8000;
    const srvTp = Number($("#srv_tp")?.value) || 1;
    try {
      toast("Starting model serving instance...");
      const inst = await api.startServing({
        model_identifier: srvModel,
        port: srvPort,
        tensor_parallel_size: srvTp,
      });
      toast("Serving instance active: " + (inst.instance_id || "ready"));
      const instances = await api.getServingInstances();
      if (Array.isArray(instances)) S.servingInstances = instances;
      render();
    } catch (err) {
      toast("Serving startup error: " + err.message);
    }
  }
  if (a === "refresh-serving") {
    try {
      const instances = await api.getServingInstances();
      if (Array.isArray(instances)) S.servingInstances = instances;
      toast("Serving instances updated");
      render();
    } catch (err) {
      toast("Error: " + err.message);
    }
  }
  if (a === "stop-serving-instance") {
    try {
      await api.stopServing(v);
      toast(`Serving instance ${v} stopped`);
      const instances = await api.getServingInstances();
      if (Array.isArray(instances)) S.servingInstances = instances;
      render();
    } catch (err) {
      toast("Stop error: " + err.message);
    }
  }
  if (a === "prepare-smoke-test") {
    const el = $("#smoke_prompt");
    if (el) {
      el.focus();
      toast(`Ready to test instance ${v}`);
    }
  }
  if (a === "run-smoke-test") {
    const promptText = $("#smoke_prompt")?.value || "What is low-rank adaptation?";
    const firstInst = S.servingInstances[0];
    const instId = firstInst?.instance_id || firstInst?.id || "default";
    try {
      toast("Sending prompt to model inference engine...");
      const resp = await api.smokeTestServing(instId, promptText);
      S.servingTestResponse = resp;
      toast("Inference response received!");
      render();
    } catch (err) {
      toast("Inference error: " + err.message);
    }
  }

  // Multi-GPU & Distributed Actions
  if (a === "refresh-multi-gpu") {
    try {
      const mgpu = await api.getMultiGPUStatus();
      S.multiGpuStatus = mgpu;
      toast("Multi-GPU topology probed");
      render();
    } catch (err) {
      toast("Probe error: " + err.message);
    }
  }
  if (a === "validate-distributed") {
    const strat = $("#dist_strategy")?.value || "ddp";
    const world = Number($("#dist_world")?.value) || 1;
    try {
      const val = await api.validateMultiGPUStrategy({ strategy: strat, world_size: world });
      S.multiGpuValidation = val;
      toast("Strategy validation complete!");
      render();
    } catch (err) {
      toast("Validation error: " + err.message);
    }
  }

  // Settings & System Actions
  if (a === "refresh-hardware") {
    try {
      const [hw, health] = await Promise.all([
        api.getSystemStatus(),
        api.getHealth(),
      ]);
      S.hardwareInfo = hw;
      if (health) S.backendOnline = true;
      toast("System diagnostics refreshed");
      render();
    } catch (err) {
      toast("Diagnostics error: " + err.message);
    }
  }
  if (a === "validate-k8s") {
    try {
      toast("Verifying Kubernetes manifests against cluster schemas...");
      const k8s = await api.validateKubernetes();
      S.k8sStatus = k8s;
      toast("Kubernetes validation completed!");
      render();
    } catch (err) {
      toast("K8s audit error: " + err.message);
    }
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
      activeEvents: [],
      activeCheckpoints: [],
      arts: [],
      evals: [],
      activeEvalResults: null,
      quantJobs: [],
      ggufExports: [],
      lastExport: null,
      servingInstances: [],
      servingTestResponse: null,
      log: [],
      prog: 0,
      timer: null,
      viewed: "",
    });
    toast("Workspace reset to initial state");
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

// Initial Hardware, Diagnostics & Backend Synchronization
async function initBackendSync() {
  try {
    const [health, hw, models, datasets, runs, artifacts, serving, k8s, quantCaps, quantJobs, evals, multiGpu] = await Promise.all([
      api.getHealth().catch(() => null),
      api.getSystemStatus().catch(() => null),
      api.getModels().catch(() => []),
      api.getDatasets().catch(() => []),
      api.getTrainingRuns().catch(() => []),
      api.getArtifacts().catch(() => []),
      api.getServingCapabilities().catch(() => null),
      api.validateKubernetes().catch(() => null),
      api.getQuantizationCapabilities().catch(() => null),
      api.getQuantizationJobs().catch(() => []),
      api.getEvaluations().catch(() => []),
      api.getMultiGPUStatus().catch(() => null),
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
      S.modelsList = models;
    }
    if (Array.isArray(datasets) && datasets.length > 0) {
      S.datasetsList = datasets;
    }
    if (Array.isArray(runs) && runs.length > 0) {
      S.runs = runs.map(r => ({
        id: r.run_id,
        m: r.model_identifier,
        d: r.dataset_reference || "dataset",
        c: `${r.training_method?.toUpperCase() || 'LORA'}, ${r.num_epochs || 1} epoch(s)`,
        done: r.status === 'completed',
        status: r.status,
        prog: r.progress_percent,
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
    if (quantCaps) S.quantCaps = quantCaps;
    if (Array.isArray(quantJobs)) S.quantJobs = quantJobs;
    if (Array.isArray(evals)) S.evals = evals;
    if (multiGpu) S.multiGpuStatus = multiGpu;
  } catch (syncErr) {
    console.warn("Backend synchronization notice:", syncErr);
  } finally {
    render();
  }
}

// Initial render & sync
render();
initBackendSync();
