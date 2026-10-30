import { S } from '../state.js';
import { P, hdr, next } from './shared.js';
import { badge } from '../components/badge.js';
import { T } from '../components/tooltip.js';

P.serving = function () {
  const caps = S.servingCaps;
  const instances = S.servingInstances || [];
  const testResp = S.servingTestResponse;

  return (
    hdr("Model Serving & vLLM", "Deploy fine-tuned models for high-throughput inference serving with interactive test prompt execution.") +
    `<div class="card">
      <h3>Inference Engine Capabilities</h3>
      <p><b>vLLM Engine:</b> ${
        caps?.can_serve_vllm
          ? badge("ok", "Available &amp; Ready")
          : badge("warn", "CPU Emulation (CUDA Required for Full vLLM)")
      }</p>
      <p class="mu">${caps?.unsupported_reason || "PagedAttention and high-throughput vLLM OpenAI-compatible server ready."}</p>
      <p class="mu">Max Supported Context: <b>${caps?.max_model_len || 4096} tokens</b> &bull; Tensor Parallelism: <b>${caps?.max_tensor_parallel_size || 1}</b></p>
    </div>

    <div class="card" style="margin-top:20px">
      <h3>Start Model Serving Instance</h3>
      <div class="grid" style="margin-top:12px">
        <div>
          <label for="srv_model">Model or Checkpoint ID</label>
          <input id="srv_model" value="${S.model?.id || "google/gemma-2-2b"}">
        </div>
        <div>
          <label for="srv_port">Port</label>
          <input id="srv_port" type="number" value="8000">
        </div>
        <div>
          <label for="srv_tp">${T("Tensor Parallel Size", "Number of GPUs to shard the model weights across.")}</label>
          <input id="srv_tp" type="number" min="1" value="1">
        </div>
      </div>
      <div class="row" style="margin-top:14px">
        <button class="btn p" data-a="launch-serving">Start Serving Instance</button>
        <button class="btn" data-a="refresh-serving">Refresh Instances</button>
      </div>
    </div>

    <h2 style="margin-top:24px">Active Serving Instances</h2>
    ${
      instances.length
        ? `<div class="card scroll">
            <table><thead><tr><th>Instance ID</th><th>Model</th><th>Port</th><th>Status</th><th>Actions</th></tr></thead>
            <tbody>
              ${instances
                .map(
                  inst =>
                    `<tr>
                      <td class="font-mono"><b>${inst.instance_id || inst.id}</b></td>
                      <td>${inst.model_name || inst.model}</td>
                      <td><code>${inst.port || 8000}</code></td>
                      <td>${badge(inst.status === "healthy" || inst.status === "running" ? "ok" : "warn", (inst.status || "RUNNING").toUpperCase())}</td>
                      <td>
                        <button class="btn" data-a="prepare-smoke-test" data-v="${inst.instance_id || inst.id}" style="padding:4px 8px;font-size:11px">Test Prompt</button>
                        <button class="btn" data-a="stop-serving-instance" data-v="${inst.instance_id || inst.id}" style="padding:4px 8px;font-size:11px;background:#5c2424;color:#ffcccc">Stop</button>
                      </td>
                    </tr>`
                )
                .join("")}
            </tbody></table>
          </div>`
        : `<div class="empty">No serving instances running. Start an instance above to deploy your model for real-time inference.</div>`
    }

    <h2 style="margin-top:28px">Interactive Smoke Test &amp; Inference</h2>
    <div class="card">
      <p class="mu">Submit an inference prompt to verify the response latency and generation quality of your served model.</p>
      <label for="smoke_prompt">Test Prompt</label>
      <textarea id="smoke_prompt" rows="3" style="width:100%;font-family:var(--sans);background:var(--spruce);color:var(--sage);border:1px solid var(--line);border-radius:8px;padding:8px" placeholder="Explain the concept of low-rank adaptation in language models."></textarea>
      
      <div class="row" style="margin-top:10px">
        <button class="btn p" data-a="run-smoke-test">Send Inference Request</button>
      </div>

      ${
        testResp
          ? `<div class="note" style="margin-top:14px">
              <b>Model Completion:</b>
              <p style="white-space:pre-wrap;margin-top:6px">${testResp.generated_text || testResp.output || testResp}</p>
              <small class="mu">Tokens: ${testResp.tokens_generated || "—"} &bull; Latency: ${testResp.latency_ms ? `${testResp.latency_ms} ms` : "Instant"}</small>
            </div>`
          : ""
      }
    </div>` + next("distributed", "Continue to Multi-GPU & Distributed")
  );
};
