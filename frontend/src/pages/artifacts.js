import { S } from '../state.js';
import { P, hdr, next } from './shared.js';
import { badge } from '../components/badge.js';

P.artifacts = function () {
  const a = [...S.arts];

  return (
    hdr("Artifact Registry", "Inspect, validate checksums, download model checkpoints, LoRA adapters, and exported GGUF binaries.") +
    `<div class="row" style="margin-bottom:14px">
      <button class="btn" data-a="sum">Open Project Summary</button>
      <button class="btn" data-a="refresh-artifacts">Refresh Artifacts</button>
    </div>` +
    (a.length
      ? `<div class="card scroll"><table>
          <thead><tr><th>Artifact Name</th><th>Type</th><th>Base Model</th><th>Status</th><th>Size</th><th>Actions</th></tr></thead>
          <tbody>${a
            .map(
              (x, i) =>
                `<tr>
                  <td><b>${x.n}</b></td>
                  <td>${badge("", x.t)}</td>
                  <td>${x.p}</td>
                  <td>${badge(x.s === "valid" || x.s === "Saved" ? "ok" : "warn", x.s)}</td>
                  <td>${x.size || "—"}</td>
                  <td>
                    ${
                      x.url
                        ? `<a href="${x.url}" download class="btn p" style="padding:4px 8px;font-size:11px">Download</a>`
                        : ""
                    }
                    ${
                      x.id
                        ? `<button class="btn" data-a="validate-artifact-item" data-v="${x.id}" style="padding:4px 8px;font-size:11px">Validate</button>
                           <button class="btn" data-a="export-artifact-item" data-v="${x.id}" style="padding:4px 8px;font-size:11px">Export</button>`
                        : x.cfg
                        ? `<button class="btn" data-a="view" data-v="${i}" style="padding:4px 8px;font-size:11px">View</button>`
                        : ""
                    }
                  </td>
                </tr>`
            )
            .join("")}</tbody></table></div>${
            S.viewed ? `<pre class="log" style="margin-top:12px">${S.viewed}</pre>` : ""
          }`
      : `<div class="empty">No artifacts registered yet. Complete a training run, quantization job, or GGUF export to populate artifacts.</div>`) +
    next("serving", "Continue to Model Serving & vLLM")
  );
};
