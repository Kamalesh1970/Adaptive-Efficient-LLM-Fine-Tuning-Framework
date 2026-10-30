import { S } from '../state.js';
import { P, hdr, next } from './shared.js';
import { badge } from '../components/badge.js';
import { T } from '../components/tooltip.js';
import { compat, alt, ok } from '../logic/compatRules.js';
import { api } from '../api.js';

P.compat = function () {
  const c = compat();
  const a = alt();
  const hwInfo = S.hardwareInfo;

  return (
    hdr("Compatibility Checker", "Review how your dataset, model, method, and hardware fit together.") +
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

<label for="mt">${T("Training method", "LoRA trains small adapters. QLoRA does LoRA on 4-bit quantized model (CUDA required). Full fine-tuning trains all weights.")}</label>
<select id="mt">
  ${[["lora", "LoRA (Recommended)"], ["qlora", "QLoRA (4-bit NF4)"], ["full", "Full Fine-Tuning"]].map(
    h => `<option value="${h[0]}" ${S.method === h[0] ? "selected" : ""}>${h[1]}</option>`
  ).join("")}
</select>
<p class="mu" style="margin-top:10px">Validated against actual memory constraints and target model context limits.</p>
</div>

<h2 style="margin-top:24px">Compatibility Analysis</h2>
<div class="card scroll">
<table><tbody>
${c.map(r => `<tr><td>${badge(r[0], { ok: "Compatible", warn: "Caution", bad: "Blocked" }[r[0]])}</td><th>${r[1]}</th><td>${r[2]}</td></tr>`).join("")}
</tbody></table>
</div>
${a ? `<div class="note"><b>Suggested alternative:</b> ${a}</div>` : ""}
${S.preflightReport ? `<div class="note" style="margin-top:16px"><b>Backend Preflight:</b> ${S.preflightReport.overall_status.toUpperCase()} &mdash; ${S.preflightReport.summary_message}</div>` : ""}
${ok() ? next("config", "Continue to Training Configuration") : ""}`
  );
};
