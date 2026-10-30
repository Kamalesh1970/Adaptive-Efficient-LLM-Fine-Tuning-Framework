import { api } from '../api.js';

P.dataset = function () {
  const d = S.ds;
  return (
    hdr("Dataset Inspector", "Upload a CSV, JSON, or JSONL file. Client-side preview and backend schema validation.") +
    `<div class="drop"><p style="margin:0 auto 12px">Choose a file or load sample data.</p><div class="row" style="justify-content:center"><input type="file" id="f" accept=".csv,.json,.jsonl" aria-label="Dataset file"><button class="btn" data-a="demo">Load sample data</button></div></div>${
      S.dsErr ? `<div class="note bad">${S.dsErr}</div>` : ""
    }` +
    (d
      ? `<h2 style="margin-top:28px">${d.name} ${d.demo ? badge("warn", "Sample data") : badge("ok", "Uploaded")} ${
          d.schemaType ? badge("", d.schemaType.toUpperCase()) : ""
        }</h2>
<div class="grid"><div class="card"><h3>Detected fields</h3><p>${d.cols.map(c => `<code>${c}</code>`).join(", ")}</p><p class="mu">${
          d.rows.length
        } records</p></div>
<div class="card"><h3>Checks</h3>${d.miss.map(m => `<p style="margin:0 0 6px">${badge(m[1] ? "warn" : "ok", m[1] ? m[1] + " missing" : "OK")} ${m[0]}</p>`).join("")}<p style="margin:0">${badge(d.dup ? "warn" : "ok", d.dup ? d.dup + " duplicate(s)" : "No duplicates")} Duplicate rows</p></div></div>
${
  d.issues
    ? `<div class="note"><b>What to do:</b> ${d.dup ? "Remove duplicate rows so the model does not over-learn them. " : ""}${
        d.miss.some(m => m[1]) ? "Fill or delete rows with empty fields; blank answers teach the model nothing." : ""
      }</div>`
    : `<div class="note">No issues found. Dataset is ready for tokenization.</div>`
}
<div class="scroll card"><table><thead><tr>${d.cols.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>${d.rows
          .slice(0, 8)
          .map(r => `<tr>${d.cols.map(c => `<td>${r[c] ?? ""}</td>`).join("")}</tr>`)
          .join("")}</tbody></table></div>` + next("models", "Continue to Model Hub")
      : `<div class="empty" style="margin-top:20px">No dataset yet.</div>`)
  );
};

export function useDS(name, txt, demo) {
  try {
    S.ds = parse(name, txt);
    S.ds.demo = demo;
    S.dsErr = "";
    toast("Dataset loaded");

    // Asynchronously register and validate on FastAPI backend
    if (typeof api !== 'undefined') {
      api.uploadDatasetContent(name, txt)
        .then(rep => {
          if (rep && rep.dataset_id) {
            S.ds.id = rep.dataset_id;
            S.ds.schemaType = rep.schema_type;
            S.ds.backendReport = rep;
            if (location.hash === "#dataset") render();
          }
        })
        .catch(err => {
          console.warn("Backend dataset upload notification:", err.message);
        });
    }

    go("dataset");
    offerNext(
      "dataset",
      S.ds.issues
        ? "Dataset loaded with " + S.ds.issues + " issue(s) to review."
        : "Dataset loaded with no issues."
    );
  } catch (e) {
    S.dsErr = "Could not read " + name + ": " + e.message + ". Check that the file is valid CSV, JSON, or JSONL.";
    render();
  }
}

if (typeof window !== 'undefined') {
  window.useDS = useDS;
}
