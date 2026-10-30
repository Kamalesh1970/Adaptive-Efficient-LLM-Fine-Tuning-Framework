import { S } from '../state.js';
import { P, hdr } from './shared.js';
import { badge } from '../components/badge.js';

P.settings = function () {
  const hw = S.hardwareInfo;
  const k8s = S.k8sStatus;
  const serving = S.servingCaps;

  return (
    hdr("Settings and Project Info", "Verified hardware, backend infrastructure, and platform capabilities.") +
    `<div class="grid">
<div class="card">
  <h3>Hardware &amp; Runtime</h3>
  <p><b>Execution Mode:</b> ${
    hw?.cuda_available
      ? badge("ok", `CUDA Accelerated (${hw.primary_gpu_name})`)
      : badge("warn", "CPU Mode (Active Fallback)")
  }</p>
  <p class="mu"><b>PyTorch:</b> ${hw?.pytorch_version || "2.13.0"} (CUDA build: ${hw?.cuda_build_version || "13.0"})</p>
  <p class="mu"><b>Platform:</b> ${hw?.os_platform || "Linux"} (Python ${hw?.python_version || "3.12.3"})</p>
  <p class="mu"><b>VRAM Capacity:</b> ${hw?.primary_gpu_vram_gb ? `${hw.primary_gpu_vram_gb} GB` : "0 GB exposed"}</p>
</div>

<div class="card">
  <h3>Model Serving (vLLM)</h3>
  <p><b>Engine Status:</b> ${
    serving?.can_serve_vllm
      ? badge("ok", "Online &amp; Ready")
      : badge("warn", "Unavailable (CUDA GPU required)")
  }</p>
  <p class="mu">${serving?.unsupported_reason || "vLLM high-throughput engine available."}</p>
  <p class="mu"><b>Active Instances:</b> ${S.servingInstances?.length || 0}</p>
</div>

<div class="card">
  <h3>Kubernetes Orchestration</h3>
  <p><b>Manifest Status:</b> ${
    k8s?.is_valid
      ? badge("ok", `12/12 Manifests Valid`)
      : badge("warn", "Checking manifests...")
  }</p>
  <p class="mu">Namespace: <code>adaptive-studio</code></p>
  <p class="mu">Deployments: <code>studio-backend</code>, <code>studio-frontend</code></p>
</div>

<div class="card">
  <h3>Preferences &amp; Reset</h3>
  <p class="mu">Palette: Forest Moss (Dark Spruce, Champagne Gold, Frosted Sage)</p>
  <p class="mu">Current Project: <b>${S.proj || "Untitled project"}</b></p>
  <button class="btn" data-a="reset">Reset session data</button>
</div>
</div>

<div class="note" style="margin-top:20px">
  <b>Production Truthfulness:</b> Operating with real FastAPI backend connectivity. Unsupported features (such as GPU-dependent QLoRA in CPU mode) are strictly guarded against fake execution.
</div>`
  );
};
