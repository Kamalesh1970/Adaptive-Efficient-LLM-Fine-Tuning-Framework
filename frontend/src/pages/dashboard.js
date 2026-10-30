import { S } from '../state.js';
import { P } from './shared.js';
import { badge } from '../components/badge.js';
import { tlHTML } from '../components/pilot.js';
import { ok } from '../logic/compatRules.js';

P.dashboard = function () {
  const st = [
    ["dataset", "Inspect dataset", !!S.ds],
    ["models", "Select model", !!S.model],
    ["compat", "Check compatibility", !!S.model && !!S.ds && ok()],
    ["config", "Save configuration", S.saved],
    ["runs", "Execute training", S.runs.length > 0],
    ["eval", "Review evaluation", S.runs.some(r => r.done || r.status === "completed")],
  ];

  const hw = S.hardwareInfo;

  return (
    `<section class="hero hb hb-contour"><div><h1>Fine-tuning, without the guesswork.</h1><p>Prepare datasets, validate model compatibility, configure training, and manage fine-tuning workflows with guidance designed for beginners.</p>
<div class="row"><button class="btn p" data-a="new">Start a Project</button><a class="btn" href="#workflow">Explore the Workflow</a></div></div>
<div class="pipe" id="workflow"><b style="font-family:var(--serif)">${
      S.proj ? S.proj : "Workflow preview"
    }</b>${st
      .map(
        (s, i) =>
          `<a class="stage" href="#${s[0]}"><span class="dot ${s[2] ? "d" : ""}"></span><span><b>0${i + 1} &mdash; ${
            s[1]
          }</b><br><small style="color:var(--grey)">${s[2] ? "Done" : "Not started"}</small></span></a>`
      )
      .join("")}</div></section>
<div class="grid"><div class="card"><h3>Project &amp; Backend</h3>${
      S.proj ? `<p><b>${S.proj}</b></p>` : '<div class="empty">No projects active yet. Click "Start a Project".</div>'
    }<p class="mu" style="margin-top:8px">Backend Status: ${
      S.backendOnline ? badge("ok", "FastAPI Connected") : badge("warn", "Checking API...")
    }</p></div>
<div class="card"><h3>Dataset and Model</h3><p>${
      S.ds ? S.ds.name + " (" + S.ds.rows.length + " rows)" : "No dataset loaded"
    }<br>${S.model ? S.model.n : "No model selected"}</p></div>
<div class="card"><h3>Hardware &amp; Execution</h3><p>${
      hw?.cuda_available
        ? badge("ok", `GPU: ${hw.primary_gpu_name}`)
        : badge("warn", "CPU Mode (Host Operating in CPU Fallback)")
    }<br><small class="mu">${S.runs.length} run(s) executed &bull; ${S.arts.length} artifact(s)</small></p></div></div>
<div class="card pilot" style="margin-top:22px"><h3>Ask the pilot</h3><p class="mu">Ask for help, your project status, or a plain-language explanation of a term. Answers come from built-in rules, real backend telemetry, and a glossary.</p><div class="chat" id="tl" aria-live="polite">${tlHTML()}</div><div class="tl row" style="flex-wrap:nowrap"><input id="ti" aria-label="Ask the pilot" placeholder="Try: status or explain lora" autocomplete="off"><button class="btn p" data-a="ask">Ask</button></div><div class="row" style="margin-top:10px">${[
      "status",
      "check",
      "hardware",
      "explain lora",
      "help",
    ]
      .map(c => `<button class="btn" data-a="ask" data-v="${c}">${c}</button>`)
      .join("")}</div></div>
<div class="note" style="margin-top:22px"><b>System Status:</b> Connected to FastAPI backend. Supervised fine-tuning with LoRA/QLoRA, GGUF export, and artifact management fully operational.</div>`
  );
};
