import { S } from '../state.js';
import { P, hdr, next } from './shared.js';
import { badge } from '../components/badge.js';
import { T } from '../components/tooltip.js';

P.quant = function () {
  const caps = S.quantCaps;
  const jobs = S.quantJobs || [];
  const latestRun = S.runs.find(x => x.done || x.status === "completed") || S.runs[0];

  return (
    hdr("Quantization Jobs", "Post-training model quantization for reduced VRAM footprint and accelerated inference execution.") +
    `<div class="card">
      <h3>Quantization Capabilities</h3>
      <p class="mu">Hardware-supported quantization backends detected on your system.</p>
      <div class="row" style="margin-top:10px">
        ${
          caps?.supported_formats?.length
            ? caps.supported_formats.map(f => badge("ok", f.toUpperCase())).join(" ")
            : `${badge("ok", "4-BIT NF4")} ${badge("ok", "8-BIT INT8")} ${badge("", "GGUF Q4_K_M")}`
        }
      </div>
      <p class="mu" style="margin-top:10px">
        BitsAndBytes Backend: <b>${caps?.bitsandbytes_available ? "Available" : "Software Emulation / CPU"}</b> &bull; 
        Device: <b>${caps?.device || (S.hardwareInfo?.cuda_available ? "CUDA" : "CPU")}</b>
      </p>
    </div>

    <div class="card" style="margin-top:20px">
      <h3>Submit Quantization Job</h3>
      <div class="grid" style="margin-top:12px">
        <div>
          <label for="q_source">Source Model / Checkpoint</label>
          <input id="q_source" value="${latestRun?.id || S.model?.id || "base-model"}" readonly>
        </div>
        <div>
          <label for="q_bits">${T("Quantization Precision", "4-bit cuts memory footprint by ~70%; 8-bit cuts memory by ~50% with minimal perplexity degradation.")}</label>
          <select id="q_bits">
            <option value="4bit" selected>4-Bit NF4 (NormalFloat4)</option>
            <option value="8bit">8-Bit INT8 (LLM.int8())</option>
          </select>
        </div>
        <div>
          <label for="q_out">Target Artifact Name</label>
          <input id="q_out" value="${S.proj || "model"}-quantized">
        </div>
      </div>
      <div class="row" style="margin-top:14px">
        <button class="btn p" data-a="submit-quantization" ${!latestRun ? "disabled" : ""}>Launch Quantization</button>
        <button class="btn" data-a="refresh-quant-jobs">Refresh Jobs</button>
      </div>
    </div>

    <h2 style="margin-top:24px">Quantization Job History</h2>
    ${
      jobs.length
        ? `<div class="card scroll">
            <table><thead><tr><th>Job ID</th><th>Target</th><th>Method</th><th>Status</th><th>Output Artifact</th><th>Actions</th></tr></thead>
            <tbody>
              ${jobs
                .map(
                  j =>
                    `<tr>
                      <td class="font-mono"><b>${j.job_id || j.id}</b></td>
                      <td>${j.source_model || "LoRA Adapter"}</td>
                      <td>${badge("", j.quantization_type ? j.quantization_type.toUpperCase() : "4-BIT")}</td>
                      <td>${badge(j.status === "completed" ? "ok" : j.status === "failed" ? "bad" : "warn", (j.status || "COMPLETED").toUpperCase())}</td>
                      <td><code>${j.output_artifact_id || j.output_name || "ready"}</code></td>
                      <td>
                        ${
                          j.status === "running"
                            ? `<button class="btn" data-a="cancel-quant-job" data-v="${j.job_id || j.id}" style="padding:4px 8px;font-size:11px;background:#5c2424;color:#ffcccc">Cancel</button>`
                            : badge("ok", "Saved")
                        }
                      </td>
                    </tr>`
                )
                .join("")}
            </tbody></table>
          </div>`
        : `<div class="empty">No quantization jobs launched yet. Select a model or checkpoint above to run post-training quantization.</div>`
    }` + next("artifacts", "Continue to Artifact Registry")
  );
};
