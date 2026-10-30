import { S } from '../state.js';
import { P, hdr, next } from './shared.js';
import { badge } from '../components/badge.js';
import { MODELS } from '../data/models.js';

P.models = function () {
  const modelsList = S.modelsList.length ? S.modelsList : MODELS;
  const selectedModel = S.model;
  const validation = S.modelValidation;

  return (
    hdr("Model Hub", "Inspect verified foundation models, parameter counts, context limits, and validate compatibility.") +
    `<div class="note">
      ${badge("ok", "Verified Catalog")} 
      Foundation models registered for Parameter-Efficient Fine-Tuning (LoRA / QLoRA) and causal language modeling.
    </div>

    <div class="grid" style="margin-top:16px">
      ${modelsList
        .map(m => {
          const isSel = selectedModel?.id === (m.model_id || m.id);
          const pCount = m.param_count_b || m.parameters_billions || m.p;
          const ctxLimit = m.max_context_length || m.context_length || m.ctx;
          const reqVram = m.min_vram_gb || m.min_gpu_vram_gb || Math.ceil(pCount * 1.5);

          return `
            <div class="card click ${isSel ? "sel" : ""}" tabindex="0" role="button" data-a="pick" data-v="${m.model_id || m.id}">
              <div class="row" style="justify-content:space-between">
                <h3>${m.name || m.n || m.model_id || m.id}</h3>
                ${isSel ? badge("ok", "Selected") : badge("", "Select")}
              </div>
              <p class="mu"><b>${pCount}B</b> parameters &bull; <b>${ctxLimit}</b> token context</p>
              <p class="mu">Base Architecture: <code>${m.architecture_family || m.architecture || "decoder-only"}</code></p>
              <p class="mu">Min VRAM Estimate: <b>${reqVram} GB</b></p>
              <p style="font-size:12px;margin-top:6px">${m.description || m.note || "Optimized for instruction and conversational adaptation."}</p>
            </div>
          `;
        })
        .join("")}
    </div>

    ${
      selectedModel
        ? `<div class="card" style="margin-top:24px;border-color:var(--gold)">
            <h3>Selected Model Details: ${selectedModel.n || selectedModel.name || selectedModel.id}</h3>
            <div class="grid" style="margin-top:10px">
              <div>
                <p><b>Model Identifier:</b> <code>${selectedModel.id}</code></p>
                <p><b>Context Window:</b> ${selectedModel.ctx || selectedModel.max_context_length || 4096} tokens</p>
                <p><b>Parameters:</b> ${selectedModel.p || selectedModel.param_count_b || 7} Billion</p>
              </div>
              <div>
                <p><b>Fine-Tuning Method:</b> ${badge("ok", S.method.toUpperCase())}</p>
                <p><b>Tokenizer:</b> PreTrained Byte-Pair / SentencePiece</p>
                <button class="btn p" data-a="validate-model-api" data-v="${selectedModel.id}" style="margin-top:8px">Run Backend Validation</button>
              </div>
            </div>
            ${
              validation
                ? `<div class="note ${validation.is_compatible ? "" : "bad"}" style="margin-top:14px">
                    <b>Backend Validation:</b> ${validation.is_compatible ? badge("ok", "COMPATIBLE") : badge("bad", "INCOMPATIBLE")}<br>
                    ${validation.message || validation.summary || "Model is ready for fine-tuning."}
                  </div>`
                : ""
            }
          </div>`
        : ""
    }

    ${selectedModel ? next("compat", "Continue to Compatibility & Preflight") : ""}`
  );
};
