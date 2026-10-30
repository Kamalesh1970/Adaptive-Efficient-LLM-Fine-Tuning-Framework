/**
 * Centralized API Client for Adaptive LLM Fine-Tuning Studio
 * Directly connects the new FineTunePilot frontend to the FastAPI backend.
 */

const API_BASE = '/api';

async function request(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const headers = { ...options.headers };
  if (!(options.body instanceof FormData) && !headers['Content-Type'] && options.method && options.method !== 'GET') {
    headers['Content-Type'] = 'application/json';
  }

  try {
    const res = await fetch(url, { ...options, headers });
    if (!res.ok) {
      let errDetail = `HTTP ${res.status}: ${res.statusText}`;
      try {
        const errJson = await res.json();
        errDetail = errJson.detail || errJson.message || JSON.stringify(errJson);
      } catch (_) {
        const text = await res.text();
        if (text) errDetail = text;
      }
      const error = new Error(errDetail);
      error.status = res.status;
      throw error;
    }

    const contentType = res.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      return await res.json();
    }
    return await res.text();
  } catch (err) {
    console.error(`API Error on ${url}:`, err);
    throw err;
  }
}

export const api = {
  // 1. System & Hardware Diagnostics
  getHealth: () => request('/health'),
  getSystemStatus: () => request('/system/status'),
  getMultiGPUStatus: () => request('/system/multi-gpu'),
  validateKubernetes: () => request('/system/kubernetes/validate'),

  // 2. Models
  getModels: async () => {
    const data = await request('/models');
    return data.models || data;
  },
  getModel: (modelId) => request(`/models/${encodeURIComponent(modelId)}`),
  validateModel: (modelId, trainingMethod = 'lora', hfToken = null) =>
    request('/models/validate', {
      method: 'POST',
      body: JSON.stringify({
        model_id: modelId,
        training_method: trainingMethod,
        hf_token: hfToken,
      }),
    }),

  // 3. Datasets
  uploadDataset: async (file, filename = null) => {
    const formData = new FormData();
    formData.append('file', file, filename || file.name);
    return request('/datasets/upload', {
      method: 'POST',
      body: formData,
    });
  },
  uploadDatasetContent: (filename, content) =>
    request('/datasets/upload', {
      method: 'POST',
      body: JSON.stringify({ filename, content }),
    }),
  getDatasets: async () => {
    const data = await request('/datasets');
    return data.datasets || data;
  },
  getDataset: (datasetId) => request(`/datasets/${datasetId}`),
  validateDataset: (datasetId) => request(`/datasets/${datasetId}/validate`, { method: 'POST' }),
  getDatasetSplits: (datasetId) => request(`/datasets/${datasetId}/splits`),

  // 4. Configuration & Preflight
  getConfigDefaults: () => request('/training/config/defaults'),
  validateConfig: (config) =>
    request('/training/config/validate', {
      method: 'POST',
      body: JSON.stringify(config),
    }),
  exportConfigYaml: (config) =>
    request('/training/config/yaml/export', {
      method: 'POST',
      body: JSON.stringify(config),
    }),
  parseConfigYaml: (yamlContent) =>
    request('/training/config/yaml/parse', {
      method: 'POST',
      body: JSON.stringify({ yaml_content: yamlContent }),
    }),
  runPreflight: (config) =>
    request('/training/preflight', {
      method: 'POST',
      body: JSON.stringify({ config }),
    }),
  previewPlan: (config) =>
    request('/training/preview', {
      method: 'POST',
      body: JSON.stringify({ config }),
    }),

  // 5. Training Lifecycle & Monitoring
  createTrainingRun: (config) =>
    request('/training/runs', {
      method: 'POST',
      body: JSON.stringify({ config }),
    }),
  getTrainingRuns: async () => {
    const data = await request('/training/runs');
    return data.runs || data;
  },
  getTrainingRun: (runId) => request(`/training/runs/${runId}`),
  getTrainingEvents: async (runId) => {
    const data = await request(`/training/runs/${runId}/events`);
    return data.events || data;
  },
  getTrainingLogs: async (runId) => {
    const data = await request(`/training/runs/${runId}/logs`);
    return data.logs || data;
  },
  getCheckpoints: async (runId) => {
    const data = await request(`/training/runs/${runId}/checkpoints`);
    return data.checkpoints || data;
  },
  cancelTrainingRun: (runId) =>
    request(`/training/runs/${runId}/cancel`, {
      method: 'POST',
    }),

  // 6. Artifacts
  getArtifacts: async () => {
    const data = await request('/artifacts');
    return data.artifacts || data;
  },
  getArtifact: (artifactId) => request(`/artifacts/${artifactId}`),
  validateArtifact: (artifactId) => request(`/artifacts/${artifactId}/validate`, { method: 'POST' }),
  exportArtifact: (artifactId, format = 'safetensors') =>
    request(`/artifacts/${artifactId}/export`, {
      method: 'POST',
      body: JSON.stringify({ format }),
    }),
  getArtifactDownloadUrl: (artifactId) => `${API_BASE}/artifacts/${artifactId}/download`,

  // 7. Evaluations
  getEvaluations: async () => {
    const data = await request('/evaluations');
    return data.evaluations || data;
  },
  createEvaluation: (config) =>
    request('/evaluations', {
      method: 'POST',
      body: JSON.stringify(config),
    }),
  getEvaluation: (evalId) => request(`/evaluations/${evalId}`),
  getEvaluationResults: (evalId) => request(`/evaluations/${evalId}/results`),
  cancelEvaluation: (evalId) => request(`/evaluations/${evalId}/cancel`, { method: 'POST' }),

  // 8. Advanced Quantization
  getQuantizationCapabilities: () => request('/quantization/capabilities'),
  createQuantizationJob: (config) =>
    request('/quantization/jobs', {
      method: 'POST',
      body: JSON.stringify(config),
    }),
  getQuantizationJobs: () => request('/quantization/jobs'),
  getQuantizationJob: (jobId) => request(`/quantization/jobs/${jobId}`),
  cancelQuantizationJob: (jobId) => request(`/quantization/jobs/${jobId}/cancel`, { method: 'POST' }),

  // 9. GGUF Export
  getGGUFExports: () => request('/export/gguf'),
  createGGUFExport: (config) =>
    request('/export/gguf', {
      method: 'POST',
      body: JSON.stringify(config),
    }),
  getGGUFExport: (exportId) => request(`/export/gguf/${exportId}`),
  getGGUFDownloadUrl: (exportId) => `${API_BASE}/export/gguf/${exportId}/download`,

  // 10. Model Serving (vLLM)
  getServingCapabilities: () => request('/serving/capabilities'),
  getServingInstances: () => request('/serving/instances'),
  startServing: (config) =>
    request('/serving/instances', {
      method: 'POST',
      body: JSON.stringify(config),
    }),
  stopServing: (instanceId) => request(`/serving/instances/${instanceId}/stop`, { method: 'POST' }),
  smokeTestServing: (instanceId, prompt, maxTokens = 64, temperature = 0.7) =>
    request(`/serving/instances/${instanceId}/smoke-test`, {
      method: 'POST',
      body: JSON.stringify({ prompt, max_tokens: maxTokens, temperature }),
    }),
};

if (typeof window !== 'undefined') {
  window.api = api;
}
