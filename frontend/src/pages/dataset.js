import { S } from '../state.js';
import { P, hdr, next } from './shared.js';
import { badge } from '../components/badge.js';
import { toast } from '../components/toast.js';
import { parse } from '../logic/parseDataset.js';
import { go, render, offerNext } from '../router.js';
import { api } from '../api.js';

P.dataset = function () {
  const d = S.ds;
  const datasetsList = S.datasetsList || [];
  const splits = S.activeDatasetSplits;
  const valReport = S.datasetValidationReport;

  return (
    hdr("Dataset Management", "Upload, inspect, validate schemas, and analyze train/val splits with the backend dataset service.") +
    `<div class="drop">
      <p style="margin:0 auto 12px">Choose a local dataset file (.csv, .json, .jsonl) or load the fine-tuning sample.</p>
      <div class="row" style="justify-content:center">
        <input type="file" id="f" accept=".csv,.json,.jsonl" aria-label="Dataset file">
        <button class="btn" data-a="demo">Load sample dataset</button>
        <button class="btn" data-a="refresh-datasets">Refresh Server Datasets</button>
      </div>
    </div>
    ${S.dsErr ? `<div class="note bad">${S.dsErr}</div>` : ""}

    ${
      datasetsList.length
        ? `<h2 style="margin-top:28px">Server Dataset Repository</h2>
          <div class="card scroll">
            <table><thead><tr><th>Dataset ID</th><th>Filename</th><th>Format</th><th>Samples</th><th>Size</th><th>Action</th></tr></thead>
            <tbody>
              ${datasetsList
                .map(
                  dsItem =>
                    `<tr>
                      <td class="font-mono"><b>${dsItem.dataset_id || dsItem.id}</b></td>
                      <td>${dsItem.filename || dsItem.name}</td>
                      <td>${badge("", dsItem.format ? dsItem.format.toUpperCase() : "CSV/JSON")}</td>
                      <td>${dsItem.sample_count || dsItem.num_samples || "—"}</td>
                      <td>${dsItem.size_bytes ? `${Math.round(dsItem.size_bytes / 1024)} KB` : "—"}</td>
                      <td>
                        <button class="btn ${S.ds?.id === (dsItem.dataset_id || dsItem.id) ? "p" : ""}" data-a="select-server-dataset" data-v="${dsItem.dataset_id || dsItem.id}">
                          ${S.ds?.id === (dsItem.dataset_id || dsItem.id) ? "Selected" : "Select"}
                        </button>
                        <button class="btn" data-a="inspect-splits" data-v="${dsItem.dataset_id || dsItem.id}">Splits</button>
                        <button class="btn" data-a="validate-server-dataset" data-v="${dsItem.dataset_id || dsItem.id}">Validate</button>
                      </td>
                    </tr>`
                )
                .join("")}
            </tbody></table>
          </div>`
        : ""
    }

    ${
      splits
        ? `<div class="card" style="margin-top:20px">
            <div class="row" style="justify-content:space-between">
              <h3>Dataset Splits: ${splits.dataset_id || "Active"}</h3>
              <button class="btn" data-a="close-splits">Close Splits</button>
            </div>
            <div class="grid" style="margin-top:12px">
              <div class="card">
                <h4>Training Split</h4>
                <p style="font:700 1.5rem var(--serif);color:var(--gold)">${splits.train_samples ?? splits.train_count ?? "80%"}</p>
                <p class="mu">Primary optimization samples</p>
              </div>
              <div class="card">
                <h4>Validation Split</h4>
                <p style="font:700 1.5rem var(--serif);color:var(--gold)">${splits.val_samples ?? splits.validation_count ?? "10%"}</p>
                <p class="mu">Generalization &amp; checkpoint tracking</p>
              </div>
              <div class="card">
                <h4>Test Split</h4>
                <p style="font:700 1.5rem var(--serif);color:var(--gold)">${splits.test_samples ?? splits.test_count ?? "10%"}</p>
                <p class="mu">Post-training evaluation benchmark</p>
              </div>
            </div>
          </div>`
        : ""
    }

    ${
      valReport
        ? `<div class="card" style="margin-top:20px">
            <div class="row" style="justify-content:space-between">
              <h3>Backend Validation Report: ${valReport.dataset_id || "Active"}</h3>
              <button class="btn" data-a="close-val-report">Dismiss</button>
            </div>
            <p><b>Status:</b> ${valReport.is_valid ? badge("ok", "Validation Passed") : badge("bad", "Issues Found")}</p>
            <p class="mu">Schema detected: <b>${valReport.schema_type || "instruction-response"}</b> &bull; Total samples: <b>${valReport.total_samples || valReport.sample_count || 0}</b></p>
            ${valReport.errors?.length ? `<div class="note bad"><b>Validation Errors:</b><br>${valReport.errors.join("<br>")}</div>` : ""}
            ${valReport.warnings?.length ? `<div class="note"><b>Warnings:</b><br>${valReport.warnings.join("<br>")}</div>` : ""}
          </div>`
        : ""
    }

    ${
      d
        ? `<h2 style="margin-top:28px">Active Dataset: ${d.name} ${
            d.demo ? badge("warn", "Sample Data") : badge("ok", "Ready")
          } ${d.schemaType ? badge("", d.schemaType.toUpperCase()) : ""}</h2>
          <div class="grid">
            <div class="card">
              <h3>Detected Fields</h3>
              <p>${d.cols.map(c => `<code>${c}</code>`).join(", ")}</p>
              <p class="mu">${d.rows.length} total rows parsed</p>
            </div>
            <div class="card">
              <h3>Data Quality Checks</h3>
              ${d.miss.map(m => `<p style="margin:0 0 6px">${badge(m[1] ? "warn" : "ok", m[1] ? m[1] + " missing" : "OK")} ${m[0]}</p>`).join("")}
              <p style="margin:0">${badge(d.dup ? "warn" : "ok", d.dup ? d.dup + " duplicate(s)" : "No duplicates")} Duplicate rows</p>
            </div>
          </div>
          <div class="scroll card" style="margin-top:14px">
            <table><thead><tr>${d.cols.map(c => `<th>${c}</th>`).join("")}</tr></thead>
            <tbody>${d.rows
              .slice(0, 8)
              .map(r => `<tr>${d.cols.map(c => `<td>${r[c] ?? ""}</td>`).join("")}</tr>`)
              .join("")}</tbody></table>
          </div>` + next("models", "Continue to Model Hub")
        : `<div class="empty" style="margin-top:20px">No active dataset selected. Choose a local file above or pick a server dataset.</div>`
    }`
  );
};

export function useDS(name, txt, demo) {
  try {
    S.ds = parse(name, txt);
    S.ds.demo = demo;
    S.dsErr = "";
    toast("Dataset loaded into memory");

    // Asynchronously register and validate on FastAPI backend
    if (typeof api !== 'undefined') {
      api.uploadDatasetContent(name, txt)
        .then(rep => {
          if (rep && rep.dataset_id) {
            S.ds.id = rep.dataset_id;
            S.ds.schemaType = rep.schema_type;
            S.ds.backendReport = rep;
            api.getDatasets().then(list => {
              if (Array.isArray(list)) S.datasetsList = list;
              if (location.hash === "#dataset") render();
            }).catch(() => {});
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
