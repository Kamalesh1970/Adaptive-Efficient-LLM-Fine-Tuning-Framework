import { S } from '../state.js';
import { P, hdr, next } from './shared.js';
import { badge } from '../components/badge.js';
import { ok } from '../logic/compatRules.js';

P.runs = function () {
  const r = S.runs;
  const isRunning = !!S.timer;

  return (
    hdr("Training Runs", "Launch and supervise fine-tuning runs with real-time telemetry and logs.") +
    `<div class="card">
<div class="row" style="margin-bottom:12px">
  <button class="btn p" data-a="run" ${isRunning || !S.saved || !ok() ? "disabled" : ""}>
    ${isRunning ? "Training in progress..." : "Start training run"}
  </button>
  ${isRunning ? `<button class="btn" data-a="cancel" style="background:#5c2424;color:#ffcccc;border-color:#8b3a3a">Cancel Run</button>` : ""}
  ${!S.saved ? '<span class="mu">Save a configuration first.</span>' : !ok() ? '<span class="bad">Resolve compatibility blockers before launching.</span>' : ""}
</div>

${
  isRunning || S.prog > 0
    ? `<div class="bar" style="margin:16px 0 6px"><i style="width:${S.prog}%"></i></div><small class="mu">${S.prog}% complete ${isRunning ? "(Active)" : "(Finished)"}</small>`
    : ""
}
${
  S.log.length
    ? `<h3 style="margin-top:16px;font-size:0.9rem">Run Telemetry &amp; Logs</h3><pre class="log" aria-label="Training log" style="max-height:260px;overflow-y:auto">${S.log.join("\n")}</pre>`
    : ""
}
</div>

<h2 style="margin-top:24px">Run History</h2>
${
  r.length
    ? `<div class="card scroll"><table><thead><tr><th>Run ID</th><th>Status</th><th>Model</th><th>Dataset</th><th>Config</th></tr></thead><tbody>${r
        .map(
          x =>
            `<tr><td class="font-mono">${x.id}</td><td>${badge(
              x.done || x.status === "completed"
                ? "ok"
                : x.status === "failed" || x.status === "cancelled"
                ? "bad"
                : "warn",
              x.status ? x.status.toUpperCase() : x.done ? "COMPLETED" : "RUNNING"
            )}</td><td>${x.m}</td><td>${x.d}</td><td>${x.c}</td></tr>`
        )
        .join("")}</tbody></table></div>`
    : `<div class="empty">No runs executed yet. Launch a run above to begin.</div>`
}` + (r.some(x => x.done || x.status === "completed") ? next("eval", "Continue to Evaluation") : "")
  );
};
