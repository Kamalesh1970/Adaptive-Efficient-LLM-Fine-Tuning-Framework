import { S } from '../state.js';
import { P, hdr, next } from './shared.js';
import { badge } from '../components/badge.js';
import { T } from '../components/tooltip.js';

P.distributed = function () {
  const mgpu = S.multiGpuStatus;
  const val = S.multiGpuValidation;
  const hw = S.hardwareInfo;

  return (
    hdr("Multi-GPU & Distributed Computing", "Inspect cluster device topology, validate DDP/FSDP configurations, and check hardware constraints.") +
    `<div class="card">
      <h3>Hardware Discovery &amp; Device Topology</h3>
      <div class="row" style="margin-top:10px">
        <span class="mu">Visible GPU Devices: </span>
        <b>${mgpu?.device_count ?? (hw?.cuda_available ? 1 : 0)} detected</b>
        ${mgpu?.device_count > 1 ? badge("ok", "MULTI-GPU READY") : badge("warn", "SINGLE DEVICE / CPU MODE")}
      </div>
      <p class="mu" style="margin-top:8px">Interconnect: <b>${mgpu?.interconnect_type || "PCIe / Emulated Bus"}</b> &bull; NVLink: <b>${mgpu?.has_nvlink ? "Active" : "None"}</b></p>
      
      ${
        mgpu?.devices?.length
          ? `<div class="scroll" style="margin-top:14px">
              <table><thead><tr><th>Device Index</th><th>Name</th><th>Total VRAM</th><th>Free VRAM</th><th>Compute Capability</th></tr></thead>
              <tbody>
                ${mgpu.devices
                  .map(
                    d =>
                      `<tr>
                        <td><b>GPU ${d.index}</b></td>
                        <td>${d.name}</td>
                        <td>${d.total_memory_gb} GB</td>
                        <td>${d.free_memory_gb} GB</td>
                        <td>SM ${d.compute_capability || "8.6"}</td>
                      </tr>`
                  )
                  .join("")}
              </tbody></table>
            </div>`
          : `<p class="mu" style="margin-top:10px">No secondary GPU detected. Operating on primary accelerator or CPU fallback.</p>`
      }

      <div class="row" style="margin-top:14px">
        <button class="btn" data-a="refresh-multi-gpu">Probe Topology</button>
      </div>
    </div>

    <div class="card" style="margin-top:20px">
      <h3>Distributed Strategy Validator</h3>
      <p class="mu">Verify feasibility of PyTorch DistributedDataParallel (DDP) or Fully Sharded Data Parallel (FSDP).</p>
      <div class="grid" style="margin-top:12px">
        <div>
          <label for="dist_strategy">${T("Distributed Backend", "DDP replicates model across GPUs; FSDP shards parameters, gradients, and optimizer states.")}</label>
          <select id="dist_strategy">
            <option value="ddp">DDP (DistributedDataParallel)</option>
            <option value="fsdp">FSDP (Fully Sharded Data Parallel)</option>
            <option value="deepspeed">DeepSpeed ZeRO-3</option>
          </select>
        </div>
        <div>
          <label for="dist_world">${T("World Size (GPU Count)", "Number of parallel workers/GPUs to allocate.")}</label>
          <input id="dist_world" type="number" min="1" max="8" value="${Math.max(1, mgpu?.device_count || 1)}">
        </div>
      </div>

      <div class="row" style="margin-top:14px">
        <button class="btn p" data-a="validate-distributed">Validate Strategy</button>
      </div>

      ${
        val
          ? `<div class="note ${val.is_valid ? "" : "bad"}" style="margin-top:14px">
              <b>Validation Result:</b> ${val.is_valid ? badge("ok", "STRATEGY VALID") : badge("bad", "STRATEGY BLOCKED")}<br>
              ${val.message || val.summary || "Configuration checked against physical cluster limits."}
              ${val.warnings?.length ? `<br><small class="mu">Notice: ${val.warnings.join(", ")}</small>` : ""}
            </div>`
          : ""
      }
    </div>` + next("settings", "Continue to Settings & Diagnostics")
  );
};
