import { S } from '../state.js';
import { P, hdr, next } from './shared.js';
import { badge } from '../components/badge.js';
import { MODELS } from '../data/models.js';

P.models = function () {
  return (
    hdr("Model Hub", "Pick a verified foundation model for your project.") +
    `<div class="note">${badge("ok", "Verified Catalog")} Models verified for causal language modeling and parameter-efficient fine-tuning (LoRA / QLoRA).</div>
<div class="grid">${MODELS.map(
      m =>
        `<div class="card click ${S.model?.id === m.id ? "sel" : ""}" tabindex="0" role="button" data-a="pick" data-v="${m.id}"><h3>${
          m.n
        }</h3><p class="mu">${m.p}B parameters. Context ${m.ctx} tokens.</p><p>Task: ${m.t}</p><p class="mu">${m.note}</p>${
          S.model?.id === m.id ? badge("ok", "Selected") : badge("", "Select")
        }</div>`
    ).join("")}</div>` +
    (S.model ? next("compat", "Check compatibility") : "")
  );
};
