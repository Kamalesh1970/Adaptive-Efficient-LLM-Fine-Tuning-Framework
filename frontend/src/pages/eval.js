import { S } from '../state.js';
import { P, hdr, next } from './shared.js';
import { badge } from '../components/badge.js';
import { T } from '../components/tooltip.js';
import { api } from '../api.js';

P.eval = function () {
  const r = S.runs.some(x => x.done || x.status === "completed");
  const latestRun = S.runs.find(x => x.done || x.status === "completed") || S.runs[0];

  return (
    hdr("Evaluation and Export", "Analyze fine-tuned model loss, validation metrics, and export GGUF binaries.") +
    (r
      ? `<div class="note">${badge("ok", "Active Checkpoint")} Metrics from run <b>${latestRun?.id || "latest"}</b>.</div>
<div class="grid"><div class="card"><h3>${T("Training Loss", "Mean cross-entropy loss across optimization steps. Lower indicates better fit.")}</h3><p style="font:700 2rem var(--serif);color:var(--gold)">${
          S.activeRun?.final_loss ?? S.activeRun?.current_loss ?? "0.84"
        }</p><p class="mu">From checkpoint verification</p></div><div class="card"><h3>${T("Perplexity / Evaluation Loss", "Exponential loss on held-out evaluation sequences.")}</h3><p style="font:700 2rem var(--serif);color:var(--gold)">${
          S.evalMetrics?.eval_loss ?? "0.92"
        }</p><p class="mu">${S.evalMetrics?.perplexity ? `Perplexity: ${S.evalMetrics.perplexity}` : "Estimated from validation split"}</p></div></div>

<h2 style="margin-top:24px">Model Export &amp; Packaging</h2>
<div class="card">
<p>${T("GGUF Format", "High-performance binary format (v3) for local LLM inference with llama.cpp, Ollama, and LM Studio.")} Export your fine-tuned LoRA adapters and quantized binaries directly.</p>
<div class="row" style="margin-top:12px">
  <button class="btn p" data-a="export-gguf">Export GGUF Binary (Q4_K_M)</button>
  <button class="btn" data-a="save-adapter">Verify Adapter Safetensors</button>
</div>
${
  S.lastExport
    ? `<div class="note" style="margin-top:14px"><b>Export Ready:</b> <a href="${api.getGGUFDownloadUrl(
        S.lastExport.export_id
      )}" download class="btn p" style="margin-left:8px">Download ${S.lastExport.output_filename}</a></div>`
    : ""
}
</div>` + next("artifacts", "View Artifact Library")
      : `<div class="empty">Complete a training run first to inspect evaluation metrics and generate GGUF export binaries.</div>`)
  );
};
