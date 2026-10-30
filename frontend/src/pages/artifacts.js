import { S } from '../state.js';
import { P, hdr } from './shared.js';
import { badge } from '../components/badge.js';
import { api } from '../api.js';

P.artifacts = function () {
  const a = [...S.arts];
  return (
    hdr("Artifact Library", "Inspect and download fine-tuned LoRA adapters, model checkpoints, and exported GGUF binaries.") +
    `<div class="row" style="margin-bottom:14px">
<button class="btn" data-a="sum">Open project summary</button>
<button class="btn" data-a="refresh-artifacts">Refresh Artifacts</button>
</div>` +
    (a.length
      ? `<div class="card scroll"><table><thead><tr><th>Artifact Name</th><th>Type</th><th>Model / Reference</th><th>Status</th><th>Size</th><th>Action</th></tr></thead><tbody>${a
          .map(
            (x, i) =>
              `<tr><td><b>${x.n}</b></td><td>${badge("", x.t)}</td><td>${x.p}</td><td>${badge("ok", x.s)}</td><td>${
                x.size || "—"
              }</td><td>${
                x.url
                  ? `<a href="${x.url}" download class="btn p" style="padding:4px 8px;font-size:11px">Download</a>`
                  : x.cfg
                  ? `<button class="btn" data-a="view" data-v="${i}">View</button>`
                  : ""
              }</td></tr>`
          )
          .join("")}</tbody></table></div>${
          S.viewed ? `<pre class="log" style="margin-top:12px">${S.viewed}</pre>` : ""
        }`
      : `<div class="empty">No artifacts registered yet. Complete a training run or GGUF export to populate artifacts.</div>`)
  );
};
