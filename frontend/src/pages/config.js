import { S } from '../state.js';
import { P, hdr, next } from './shared.js';
import { badge } from '../components/badge.js';
import { T } from '../components/tooltip.js';
import { compat } from '../logic/compatRules.js';

P.config = function () {
  const c = S.cfg;
  const w = compat().filter(x => x[0] !== "ok" && x[1] !== "Dataset");

  return (
    hdr("Training Configuration", "Configure hyperparameters for fine-tuning. Hover or focus fields for plain-language assistance.") +
    `<div class="card">
      <h3>Hyperparameters &amp; Training Arguments</h3>
      <div class="grid">
        <div>
          <label for="c_epochs">${T("Epochs", "Number of complete passes over the training dataset. Typically 1 to 3.")}</label>
          <input id="c_epochs" data-c="epochs" type="number" min="1" step="1" value="${c.epochs}">
        </div>
        <div>
          <label for="c_lr">${T("Learning rate", "Step size for optimizer weight updates. Recommended 0.0002 for LoRA.")}</label>
          <input id="c_lr" data-c="lr" value="${c.lr}">
        </div>
        <div>
          <label for="c_batch">${T("Batch size", "Per-device training batch size. Lower to reduce memory consumption.")}</label>
          <input id="c_batch" data-c="batch" type="number" min="1" step="1" value="${c.batch}">
        </div>
        <div>
          <label for="c_grad_accum">${T("Gradient Accumulation", "Number of forward passes to accumulate gradients before executing an optimizer step.")}</label>
          <input id="c_grad_accum" data-c="grad_accum" type="number" min="1" step="1" value="${c.grad_accum || 1}">
        </div>
        <div>
          <label for="c_rank">${T("LoRA Rank (r)", "Rank dimension of the low-rank adapter matrices. Common values: 8, 16, 32.")}</label>
          <input id="c_rank" data-c="rank" type="number" min="1" step="1" value="${c.rank}">
        </div>
        <div>
          <label for="c_alpha">${T("LoRA Alpha", "Scaling factor for low-rank updates. Often 2x the rank dimension.")}</label>
          <input id="c_alpha" data-c="alpha" type="number" min="1" step="1" value="${c.alpha || 16}">
        </div>
        <div>
          <label for="c_dropout">${T("LoRA Dropout", "Dropout probability applied to adapter layers to prevent overfitting.")}</label>
          <input id="c_dropout" data-c="dropout" value="${c.dropout || "0.05"}">
        </div>
        <div>
          <label for="c_len">${T("Max Sequence Length", "Maximum tokens per input sequence. Long inputs will be truncated.")}</label>
          <input id="c_len" data-c="len" type="number" min="64" step="64" value="${c.len}">
        </div>
        <div>
          <label for="c_target">${T("Target Modules", "Comma-separated linear projection layers to apply LoRA adapters to.")}</label>
          <input id="c_target" data-c="target_modules" value="${c.target_modules || "q_proj,v_proj"}">
        </div>
        <div>
          <label for="c_precision">${T("Precision", "Floating point format: FP16 (CUDA), BF16 (Ampere+), or FP32 (CPU).")}</label>
          <select id="c_precision" data-c="precision">
            ${["fp16", "bf16", "fp32"].map(p => `<option value="${p}" ${c.precision === p ? "selected" : ""}>${p.toUpperCase()}</option>`).join("")}
          </select>
        </div>
        <div>
          <label for="c_optimizer">${T("Optimizer", "Optimization algorithm: standard AdamW or memory-efficient 8-bit Adam.")}</label>
          <select id="c_optimizer" data-c="optimizer">
            ${["adamw_torch", "adamw_8bit", "sgd"].map(o => `<option value="${o}" ${c.optimizer === o ? "selected" : ""}>${o}</option>`).join("")}
          </select>
        </div>
        <div>
          <label for="c_scheduler">${T("LR Scheduler", "Decay schedule: linear, cosine, or constant.")}</label>
          <select id="c_scheduler" data-c="scheduler">
            ${["linear", "cosine", "constant"].map(s => `<option value="${s}" ${c.scheduler === s ? "selected" : ""}>${s}</option>`).join("")}
          </select>
        </div>
      </div>

      <div class="note" id="ro" aria-live="polite" style="margin-top:16px">
        <b>Parameter readout</b><br>Hover or focus a field to view parameter documentation and memory impact.
      </div>

      <p class="mu" style="margin-top:12px">
        Selected Method: <b>${S.method.toUpperCase()}</b> &bull; Target Model: <b>${S.model ? S.model.n || S.model.id : "None selected"}</b>
      </p>

      ${w.map(x => `<p>${badge(x[0], x[0] === "bad" ? "Error" : "Warning")} ${x[2]}</p>`).join("")}

      <div class="row" style="margin-top:16px">
        <button class="btn p" data-a="save">Save &amp; Validate Configuration</button>
        <button class="btn" data-a="export-yaml">Export to YAML</button>
        <button class="btn" data-a="reset-config">Reset to Defaults</button>
        ${S.saved ? badge("ok", "Saved & Validated") : ""}
      </div>
    </div>

    ${
      S.yamlPreview
        ? `<div class="card" style="margin-top:20px">
            <div class="row" style="justify-content:space-between">
              <h3>Configuration YAML</h3>
              <button class="btn" data-a="close-yaml">Close</button>
            </div>
            <pre class="log" style="margin-top:10px">${S.yamlPreview}</pre>
          </div>`
        : ""
    }

    ${S.saved ? next("runs", "Continue to Training Runs") : ""}`
  );
};
