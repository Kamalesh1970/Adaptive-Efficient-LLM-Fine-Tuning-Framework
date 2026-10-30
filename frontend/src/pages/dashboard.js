import { S } from '../state.js';
import { P } from './shared.js';
import { badge } from '../components/badge.js';
import { tlHTML } from '../components/pilot.js';
import { ok } from '../logic/compatRules.js';

P.dashboard = function () {
  const hw = S.hardwareInfo;
  const isRunning = !!S.timer;
  const recentRuns = S.runs.slice(0, 4);

  const st = [
    ["dataset", "Dataset Management", !!S.ds],
    ["models", "Model Hub", !!S.model],
    ["compat", "Compatibility & Preflight", !!S.model && !!S.ds && ok()],
    ["config", "Training Configuration", S.saved],
    ["runs", "Training Runs & Logs", S.runs.length > 0],
    ["eval", "Model Evaluation", S.runs.some(r => r.done || r.status === "completed")],
    ["quant", "Quantization Jobs", S.quantJobs.length > 0],
    ["artifacts", "Artifact Registry", S.arts.length > 0],
    ["serving", "Model Serving & vLLM", S.servingInstances.length > 0],
  ];

  return (
    `<section class="hero hb hb-contour"><div>
      <h1>Fine-tuning, without the guesswork.</h1>
      <p>Prepare datasets, validate model compatibility, configure LoRA/QLoRA, monitor training, evaluate perplexity, and export GGUF binaries connected directly to your FastAPI backend.</p>
      <div class="row">
        <button class="btn p" data-a="new">Start a Project</button>
        <a class="btn" href="#dataset">Upload Dataset</a>
        <a class="btn" href="#models">Model Hub</a>
        <a class="btn" href="#runs">Active Runs</a>
      </div>
    </div>
    <div class="pipe" id="workflow">
      <b style="font-family:var(--serif)">${S.proj ? S.proj : "Studio Workflow Pipeline"}</b>
      ${st
        .map(
          (s, i) =>
            `<a class="stage" href="#${s[0]}"><span class="dot ${s[2] ? "d" : ""}"></span><span><b>0${i + 1} &mdash; ${
              s[1]
            }</b><br><small style="color:var(--grey)">${s[2] ? "Ready" : "Pending"}</small></span></a>`
        )
        .join("")}
    </div></section>

    <div class="grid">
      <div class="card">
        <h3>Platform &amp; Connectivity</h3>
        <p><b>Status:</b> ${S.backendOnline ? badge("ok", "FastAPI Connected") : badge("warn", "Connecting...")}</p>
        <p class="mu">Project: <b>${S.proj || "Default Workspace"}</b></p>
        <p class="mu">Registered Models: <b>${S.modelsList.length || 7}</b> &bull; Datasets: <b>${S.datasetsList.length}</b></p>
      </div>

      <div class="card">
        <h3>Hardware Diagnostics</h3>
        <p><b>Accelerator:</b> ${
          hw?.cuda_available
            ? badge("ok", `GPU: ${hw.primary_gpu_name || "NVIDIA CUDA"}`)
            : badge("warn", "CPU Mode (Active Fallback)")
        }</p>
        <p class="mu">VRAM Allocated: <b>${hw?.primary_gpu_vram_gb ? `${hw.primary_gpu_vram_gb} GB` : `${S.hw} GB (System Memory)`}</b></p>
        <p class="mu">PyTorch: <b>${hw?.pytorch_version || "2.13.0"}</b> &bull; OS: <b>${hw?.os_platform || "Linux"}</b></p>
      </div>

      <div class="card">
        <h3>Training Lifecycle</h3>
        <p><b>Execution Status:</b> ${isRunning ? badge("warn", "Training Active") : badge("", "Idle / Standby")}</p>
        <p class="mu">Total Runs: <b>${S.runs.length}</b> &bull; Checkpoints: <b>${S.activeCheckpoints.length}</b></p>
        <p class="mu">Artifacts: <b>${S.arts.length}</b> &bull; GGUF Exports: <b>${S.ggufExports.length}</b></p>
      </div>
    </div>

    ${
      isRunning
        ? `<div class="card" style="margin-top:20px;border-color:var(--gold)">
            <div class="row" style="justify-content:space-between">
              <h3>Active Training Run: ${S.activeRun?.run_id || "Job"}</h3>
              <button class="btn" data-a="cancel" style="background:#5c2424;color:#ffcccc;border-color:#8b3a3a">Cancel Run</button>
            </div>
            <div class="bar" style="margin:12px 0 6px"><i style="width:${S.prog}%"></i></div>
            <p class="mu">${S.prog}% completed &bull; Step ${S.activeRun?.current_step || 0} / ${S.activeRun?.total_steps || 100} &bull; Loss: ${S.activeRun?.current_loss ?? "Calculating..."}</p>
            <div class="row" style="margin-top:10px">
              <a href="#runs" class="btn p">Open Run Monitor &amp; Logs</a>
            </div>
          </div>`
        : ""
    }

    <div class="card pilot" style="margin-top:22px">
      <h3>Ask the pilot</h3>
      <p class="mu">Interactive assistant querying real backend telemetry, hardware diagnostics, glossary terms, and project status.</p>
      <div class="chat" id="tl" aria-live="polite">${tlHTML()}</div>
      <div class="tl row" style="flex-wrap:nowrap">
        <input id="ti" aria-label="Ask the pilot" placeholder="Try: status, hardware, models, runs, or explain lora" autocomplete="off">
        <button class="btn p" data-a="ask">Ask</button>
      </div>
      <div class="row" style="margin-top:10px">
        ${["status", "hardware", "models", "runs", "check", "explain lora", "help"]
          .map(c => `<button class="btn" data-a="ask" data-v="${c}">${c}</button>`)
          .join("")}
      </div>
    </div>

    ${
      recentRuns.length
        ? `<h2 style="margin-top:28px">Recent Training Runs</h2>
          <div class="card scroll">
            <table><thead><tr><th>Run ID</th><th>Model</th><th>Dataset</th><th>Status</th><th>Progress</th><th>Action</th></tr></thead>
            <tbody>
              ${recentRuns
                .map(
                  r =>
                    `<tr>
                      <td class="font-mono"><b>${r.id}</b></td>
                      <td>${r.m}</td>
                      <td>${r.d}</td>
                      <td>${badge(r.done || r.status === 'completed' ? 'ok' : r.status === 'failed' || r.status === 'cancelled' ? 'bad' : 'warn', r.status ? r.status.toUpperCase() : 'COMPLETED')}</td>
                      <td>${r.prog !== undefined ? `${r.prog}%` : r.done ? '100%' : '—'}</td>
                      <td><a href="#runs" class="btn" style="padding:4px 8px;font-size:11px">Inspect</a></td>
                    </tr>`
                )
                .join("")}
            </tbody></table>
          </div>`
        : ""
    }`
  );
};
