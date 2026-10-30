import { S } from '../state.js';
import { P, hdr, next } from './shared.js';
import { badge } from '../components/badge.js';
import { ok } from '../logic/compatRules.js';

P.runs = function () {
  const r = S.runs;
  const isRunning = !!S.timer;
  const activeEvents = S.activeEvents || [];
  const checkpoints = S.activeCheckpoints || [];

  return (
    hdr("Training Runs & Logs", "Supervise fine-tuning execution with live loss curves, real-time logs, lifecycle events, and checkpoints.") +
    `<div class="card">
      <div class="row" style="margin-bottom:12px">
        <button class="btn p" data-a="run" ${isRunning || !S.saved || !ok() ? "disabled" : ""}>
          ${isRunning ? "Training in progress..." : "Start Training Run"}
        </button>
        ${isRunning ? `<button class="btn" data-a="cancel" style="background:#5c2424;color:#ffcccc;border-color:#8b3a3a">Cancel Run</button>` : ""}
        <button class="btn" data-a="refresh-runs">Refresh Runs &amp; History</button>
        ${!S.saved ? '<span class="mu">Save a configuration first.</span>' : !ok() ? '<span class="bad">Resolve compatibility blockers before launching.</span>' : ""}
      </div>

      ${
        isRunning || S.prog > 0
          ? `<div class="bar" style="margin:16px 0 6px"><i style="width:${S.prog}%"></i></div>
            <p class="mu">${S.prog}% completed &bull; Status: <b>${S.activeRun?.status || (isRunning ? "RUNNING" : "COMPLETED")}</b> &bull; Current Loss: <b>${S.activeRun?.current_loss ?? "0.85"}</b></p>`
          : ""
      }

      ${
        S.log.length
          ? `<div style="margin-top:16px">
              <div class="row" style="justify-content:space-between">
                <h3 style="font-size:0.95rem">Training Logs</h3>
                <span class="mu">${S.log.length} lines captured</span>
              </div>
              <pre class="log" aria-label="Training log" style="max-height:240px;overflow-y:auto;margin-top:6px">${S.log.join("\n")}</pre>
            </div>`
          : ""
      }
    </div>

    ${
      activeEvents.length
        ? `<h2 style="margin-top:24px">Lifecycle Events: ${S.activeRun?.run_id || "Active Run"}</h2>
          <div class="card scroll">
            <table><thead><tr><th>Timestamp</th><th>Event Type</th><th>Phase</th><th>Details</th></tr></thead>
            <tbody>
              ${activeEvents
                .map(
                  ev =>
                    `<tr>
                      <td class="font-mono">${ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : "—"}</td>
                      <td>${badge("", ev.event_type || "STEP")}</td>
                      <td>${ev.phase || "TRAINING"}</td>
                      <td>${ev.message || ev.details || "—"}</td>
                    </tr>`
                )
                .join("")}
            </tbody></table>
          </div>`
        : ""
    }

    ${
      checkpoints.length
        ? `<h2 style="margin-top:24px">Checkpoints: ${S.activeRun?.run_id || "Active Run"}</h2>
          <div class="card scroll">
            <table><thead><tr><th>Step</th><th>Epoch</th><th>Loss</th><th>Checkpoint Path</th></tr></thead>
            <tbody>
              ${checkpoints
                .map(
                  chk =>
                    `<tr>
                      <td><b>Step ${chk.step}</b></td>
                      <td>Epoch ${chk.epoch ?? 1}</td>
                      <td>${chk.loss ? chk.loss.toFixed(4) : "—"}</td>
                      <td class="font-mono" style="font-size:11px">${chk.path || "checkpoint-" + chk.step}</td>
                    </tr>`
                )
                .join("")}
            </tbody></table>
          </div>`
        : ""
    }

    <h2 style="margin-top:24px">Run History</h2>
    ${
      r.length
        ? `<div class="card scroll"><table>
            <thead><tr><th>Run ID</th><th>Status</th><th>Model</th><th>Dataset</th><th>Config</th><th>Actions</th></tr></thead>
            <tbody>${r
              .map(
                x =>
                  `<tr>
                    <td class="font-mono"><b>${x.id}</b></td>
                    <td>${badge(
                      x.done || x.status === "completed"
                        ? "ok"
                        : x.status === "failed" || x.status === "cancelled"
                        ? "bad"
                        : "warn",
                      x.status ? x.status.toUpperCase() : x.done ? "COMPLETED" : "RUNNING"
                    )}</td>
                    <td>${x.m}</td>
                    <td>${x.d}</td>
                    <td>${x.c}</td>
                    <td>
                      <button class="btn" data-a="inspect-run" data-v="${x.id}" style="padding:4px 8px;font-size:11px">Inspect</button>
                      ${
                        x.status === "running"
                          ? `<button class="btn" data-a="cancel-run-item" data-v="${x.id}" style="padding:4px 8px;font-size:11px;background:#5c2424;color:#ffcccc">Cancel</button>`
                          : ""
                      }
                    </td>
                  </tr>`
              )
              .join("")}</tbody></table></div>`
        : `<div class="empty">No runs executed yet. Save a configuration and launch above to begin.</div>`
    }` + (r.some(x => x.done || x.status === "completed") ? next("eval", "Continue to Model Evaluation") : "")
  );
};
