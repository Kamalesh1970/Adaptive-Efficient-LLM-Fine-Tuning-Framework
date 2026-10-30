// Rule-based memory estimate and compatibility checks
export function est() {
  const m = S.model;
  if (!m) return null;
  const b = { full: 16, lora: 2.4, qlora: 0.9 }[S.method] || 2.4;
  return +(m.p * b + 1).toFixed(1);
}

export function compat() {
  const r = [];
  const m = S.model;
  const e = est();

  // 1. Dataset check
  if (S.ds) {
    if (S.ds.issues) {
      r.push(["warn", "Dataset", `Loaded, but ${S.ds.issues} issue(s) were found. Fix them for cleaner training.`]);
    } else {
      r.push(["ok", "Dataset", `Loaded with no detected issues (${S.ds.rows?.length || 0} samples).`]);
    }
  } else {
    r.push(["bad", "Dataset", "No dataset loaded. Open the Dataset Inspector to upload or choose data."]);
  }

  // 2. Model check
  if (m) {
    r.push(["ok", "Model", `${m.n} supports text-based causal fine-tuning.`]);
  } else {
    r.push(["bad", "Model", "No model selected. Pick one in the Model Hub."]);
  }

  // 3. Hardware & Memory check
  if (m) {
    const hw = +S.hw || 16;
    if (S.hardwareInfo && !S.hardwareInfo.cuda_available) {
      r.push(["warn", "Hardware", "Operating in CPU mode. Training will execute via CPU fallback (GPU unavailable)."]);
    } else if (e <= hw) {
      r.push(["ok", "Memory", `Estimated memory ${e} GB fits within available ${hw} GB.`]);
    } else {
      r.push(["bad", "Memory", `Estimated memory ${e} GB exceeds available ${hw} GB.`]);
    }

    // 4. Sequence length check
    if (S.cfg.len > m.ctx) {
      r.push(["bad", "Sequence length", `${S.cfg.len} tokens exceeds model context limit of ${m.ctx}.`]);
    } else {
      r.push(["ok", "Sequence length", `Within model ${m.ctx}-token context limit.`]);
    }

    // 5. Method check
    if (S.method === "full" && m.p > 3) {
      r.push(["warn", "Method", `Full fine-tuning of a ${m.p}B model is heavy. LoRA or QLoRA trains small adapters and saves VRAM.`]);
    }
    if (S.method === "qlora" && S.hardwareInfo && !S.hardwareInfo.cuda_available) {
      r.push(["bad", "Method", "QLoRA requires active CUDA GPU acceleration with 4-bit bitsandbytes support."]);
    }
  }

  return r;
}

export function alt() {
  const m = S.model;
  if (!m) return "";
  const hw = +S.hw || 16;
  if (est() <= hw) return "";
  for (const k of ["lora", "qlora"]) {
    const b = { lora: 2.4, qlora: 0.9 }[k];
    const e = m.p * b + 1;
    if (e <= hw) {
      return `Try <b>${k === "lora" ? "LoRA" : "QLoRA"}</b> (about ${e.toFixed(1)} GB). It freezes the base model and trains small adapters.`;
    }
  }
  const s = MODELS.filter(x => x.p * 0.9 + 1 <= hw).pop();
  return s ? `Try a smaller model such as <b>${s.n}</b> with QLoRA.` : "Choose a larger GPU option or a smaller model.";
}

export const ok = () => compat().every(c => c[0] !== "bad");

if (typeof window !== 'undefined') {
  window.est = est;
  window.compat = compat;
  window.alt = alt;
  window.ok = ok;
}
