import { S } from '../state.js';
import { P, hdr, next } from './shared.js';
import { badge } from '../components/badge.js';
import { T } from '../components/tooltip.js';
import { compat } from '../logic/compatRules.js';
import { api } from '../api.js';

P.config = function () {
  const c = S.cfg;
  const w = compat().filter(x => x[0] !== "ok" && x[1] !== "Dataset");

  return (
    hdr("Training Configuration", "Configure hyperparameters for fine-tuning. Hover or focus fields for plain-language assistance.") +
    `<div class="card"><div class="grid">${[
      ["epochs", "Epochs", "Number of complete passes over the training dataset. Typically 2 to 3."],
      ["lr", "Learning rate", "Step size for parameter updates. Usually 0.0002 for LoRA."],
      ["batch", "Batch size", "Number of training examples per forward step. Lower to reduce memory."],
      ["rank", "LoRA rank (r)", "Rank dimension of the low-rank adapter matrices. Common values: 8, 16, 32."],
      ["len", "Max sequence length", "Maximum tokens per input sequence. Long inputs will be truncated."],
    ].map(p => `<div><label for="c_${p[0]}">${T(p[1], p[2])}</label><input id="c_${p[0]}" data-c="${p[0]}" value="${c[p[0]]}"></div>`).join("")}</div>
<div class="note" id="ro" aria-live="polite"><b>Parameter readout</b><br>Hover or focus a field to see guidance and memory impact.</div>
<p class="mu" style="margin-top:12px">Method: ${S.method.toUpperCase()}. Model: ${S.model ? S.model.n : "None selected"}.</p>
${w.map(x => `<p>${badge(x[0], x[0] === "bad" ? "Error" : "Warning")} ${x[2]}</p>`).join("")}
<button class="btn p" data-a="save">Save configuration</button> ${S.saved ? badge("ok", "Saved") : ""}</div>` +
    (S.saved ? next("runs", "Continue to Training Runs") : "")
  );
};
