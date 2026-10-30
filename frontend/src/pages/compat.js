import { S } from '../state.js';
import { P, hdr, next } from './shared.js';
import { badge } from '../components/badge.js';
import { T } from '../components/tooltip.js';
import { compat, alt, ok } from '../logic/compatRules.js';

P.compat = function () {
  const c = compat();
  const a = alt();
  const hwInfo = S.hardwareInfo;
  const preflight = S.preflightReport;
  const runPlan = S.runPlan;

  return (
    hdr("Compatibility & Preflight", "Run mathematical memory estimates and backend preflight checks before launching training.") +
    `<div class="card">
      <div class="row" style="margin-bottom:12px">
        <span class="mu">Hardware State: </span>
        ${
          hwInfo?.cuda_available
            ? badge("ok", `GPU: ${hwInfo.primary_gpu_name || "CUDA Active"} (${hwInfo.primary_gpu_vram_gb || S.hw} GB)`)
            : badge("warn", "CPU Mode (Host Operating in CPU Fallback)")
        }
      </div>

      <label for="hw">${T("Allocated / Declared Memory (GB)", "Memory threshold used to estimate memory feasibility.")}</label>
      <input id="hw" type="number" min="1" step="1" inputmode="decimal" value="${S.hw}" style="max-width:160px"> GB

      <label for="mt">${T("Training method", "LoRA trains low-rank adapter weights. QLoRA quantizes the base model to 4-bit NF4 (requires CUDA). Full fine-tuning trains all weights.")}</label>
      <select id="mt">
        ${[["lora", "LoRA (Recommended)"], ["qlora", "QLoRA (4-bit NF4)"], ["full", "Full Fine-Tuning"]].map(
          h => `<option value="${h[0]}" ${S.method === h[0] ? "selected" : ""}>${h[1]}</option>`
        ).join("")}
      </select>

      <div class="row" style="margin-top:16px">
        <button class="btn p" data-a="trigger-preflight">Run Backend Preflight Check</button>
        <button class="btn" data-a="trigger-preview">Generate Run Plan Preview</button>
      </div>
    </div>

    <h2 style="margin-top:24px">Static Rule Compatibility</h2>
    <div class="card scroll">
      <table><tbody>
        ${c.map(r => `<tr><td>${badge(r[0], { ok: "Compatible", warn: "Caution", bad: "Blocked" }[r[0]])}</td><th>${r[1]}</th><td>${r[2]}</td></tr>`).join("")}
      </tbody></table>
    </div>
    ${a ? `<div class="note"><b>Suggested alternative:</b> ${a}</div>` : ""}

    ${
      preflight
        ? `<h2 style="margin-top:24px">Backend Preflight Report</h2>
          <div class="card">
            <p><b>Overall Status:</b> ${
              preflight.overall_status === "ready" || preflight.overall_status === "pass"
                ? badge("ok", "PREFLIGHT PASSED")
                : badge("warn", (preflight.overall_status || "WARN").toUpperCase())
            }</p>
            <p class="mu">${preflight.summary_message || "System satisfies minimum hardware and schema criteria."}</p>
            ${
              preflight.checks?.length
                ? `<div class="scroll" style="margin-top:10px"><table>
                    <thead><tr><th>Rule</th><th>Status</th><th>Message</th></tr></thead>
                    <tbody>
                      ${preflight.checks
                        .map(
                          chk =>
                            `<tr>
                              <td><b>${chk.check_name}</b></td>
                              <td>${badge(chk.passed ? "ok" : chk.severity === "warning" ? "warn" : "bad", chk.passed ? "PASS" : "FAIL")}</td>
                              <td>${chk.message || "—"}</td>
                            </tr>`
                        )
                        .join("")}
                    </tbody>
                  </table></div>`
                : ""
            }
          </div>`
        : ""
    }

    ${
      runPlan
        ? `<h2 style="margin-top:24px">Run Plan &amp; Resource Preview</h2>
          <div class="card">
            <div class="grid">
              <div class="card">
                <h4>Total Optimization Steps</h4>
                <p style="font:700 1.6rem var(--serif);color:var(--gold)">${runPlan.total_steps || runPlan.num_training_steps || "100"}</p>
                <p class="mu">Calculated from epochs &amp; batch size</p>
              </div>
              <div class="card">
                <h4>Effective Batch Size</h4>
                <p style="font:700 1.6rem var(--serif);color:var(--gold)">${runPlan.effective_batch_size || (S.cfg.batch * (S.cfg.grad_accum || 1))}</p>
                <p class="mu">Per-device &times; gradient accumulation</p>
              </div>
              <div class="card">
                <h4>Peak VRAM Profile</h4>
                <p style="font:700 1.6rem var(--serif);color:var(--gold)">${runPlan.estimated_peak_memory_gb || runPlan.peak_vram_gb || "3.4"} GB</p>
                <p class="mu">Weights, activations, and optimizer states</p>
              </div>
            </div>
          </div>`
        : ""
    }

    ${ok() ? next("config", "Continue to Training Configuration") : ""}`
  );
};
