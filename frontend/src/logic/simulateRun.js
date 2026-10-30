// Real Training Run Execution & Polling Service
import { api } from '../api.js';

export async function run() {
  if (S.timer) return;
  if (!S.saved || !ok()) {
    toast("Please save a compatible configuration first.");
    return;
  }

  const c = S.cfg;
  const projectName = S.proj || "fine-tuning-project";
  const modelId = S.model?.id || "google/gemma-2-2b";
  const datasetRef = S.ds?.id || S.ds?.name || "dataset";
  const method = S.method === "qlora" ? "qlora" : "lora";

  const payload = {
    project_name: projectName,
    model_identifier: modelId,
    training_method: method,
    dataset_reference: datasetRef,
    num_epochs: Number(c.epochs) || 1,
    learning_rate: Number(c.lr) || 0.0002,
    batch_size: Number(c.batch) || 2,
    lora_rank: Number(c.rank) || 8,
    context_length: Number(c.len) || 512,
    lora_alpha: Number(c.rank) * 2,
    gradient_accumulation_steps: 1,
  };

  S.prog = 0;
  S.log = [`[INIT] Preparing training run for ${modelId}...`];
  if (location.hash === "#runs") render();

  try {
    const job = await api.createTrainingRun(payload);
    const runId = job.run_id;
    S.activeRun = job;

    const runItem = {
      id: runId,
      m: S.model.n,
      d: S.ds?.name || "-",
      c: `${method.toUpperCase()}, ${c.epochs} epoch(s)`,
      done: false,
      status: job.status,
    };
    S.runs.unshift(runItem);
    S.log.push(`[LAUNCH] Run ${runId} queued. Device: ${job.device || "cpu"}`);
    toast(`Training run ${runId} launched!`);
    render();

    // Poll backend every 1000ms for real status and logs
    S.timer = setInterval(async () => {
      try {
        const [updated, logs] = await Promise.all([
          api.getTrainingRun(runId).catch(() => null),
          api.getTrainingLogs(runId).catch(() => []),
        ]);

        if (updated) {
          S.activeRun = updated;
          runItem.status = updated.status;
          const totalSteps = updated.total_steps || 1;
          const curStep = updated.current_step || 0;
          S.prog = Math.min(100, Math.round((curStep / totalSteps) * 100));

          if (Array.isArray(logs) && logs.length > 0) {
            S.log = logs.map(l => (typeof l === 'string' ? l : `[${l.level || 'INFO'}] ${l.message}`));
          } else if (updated.current_loss !== null && updated.current_loss !== undefined) {
            S.log.push(`[STEP ${curStep}/${totalSteps}] Loss: ${updated.current_loss.toFixed(4)}`);
          }

          if (updated.status === 'completed') {
            clearInterval(S.timer);
            S.timer = null;
            runItem.done = true;
            S.prog = 100;
            S.log.push(`[COMPLETED] Training completed successfully. Final loss: ${updated.final_loss ?? updated.current_loss ?? 'N/A'}`);
            toast("Training run complete!");
            offerNext("runs", "Training run complete.");

            // Refresh artifacts
            api.getArtifacts().then(arts => {
              if (Array.isArray(arts)) {
                S.arts = arts.map(a => ({
                  id: a.artifact_id,
                  n: a.artifact_type || "Model Adapter",
                  t: a.fine_tuning_method?.toUpperCase() || "LoRA",
                  p: a.model_identifier || S.proj,
                  s: a.validation_status || "Saved",
                  size: a.size_bytes ? `${Math.round(a.size_bytes / 1024)} KB` : "—",
                  url: api.getArtifactDownloadUrl(a.artifact_id),
                }));
              }
            }).catch(() => {});

            if (location.hash === "#runs") render();
          } else if (updated.status === 'failed' || updated.status === 'cancelled') {
            clearInterval(S.timer);
            S.timer = null;
            S.log.push(`[TERMINATED] Run ended with status: ${updated.status}. ${updated.error_message || ''}`);
            toast(`Run ${updated.status}: ${updated.error_message || ''}`);
            if (location.hash === "#runs") render();
          } else {
            if (location.hash === "#runs") render();
          }
        }
      } catch (pollErr) {
        console.error("Polling error:", pollErr);
      }
    }, 1200);
  } catch (err) {
    clearInterval(S.timer);
    S.timer = null;
    S.log.push(`[ERROR] Failed to start run: ${err.message}`);
    toast(`Launch error: ${err.message}`);
    render();
  }
}

export async function cancelRun(runId) {
  const targetId = runId || S.activeRun?.run_id || (S.runs[0] && S.runs[0].id);
  if (!targetId) return;
  try {
    await api.cancelTrainingRun(targetId);
    toast(`Cancellation requested for ${targetId}`);
    if (S.timer) {
      clearInterval(S.timer);
      S.timer = null;
    }
    if (S.runs[0]) S.runs[0].status = 'cancelled';
    render();
  } catch (err) {
    toast(`Cancel error: ${err.message}`);
  }
}

if (typeof window !== 'undefined') {
  window.run = run;
  window.cancelRun = cancelRun;
}
