// Hash routing, navigation, rendering, and confirm-to-continue step flow
import { $ } from './utils.js';
import { NAV } from './data/nav.js';
import { S } from './state.js';
import { P } from './pages/shared.js';
import { ok } from './logic/compatRules.js';
import { useDS } from './pages/dataset.js';

export function go(r) {
  location.hash = r;
  if (location.hash.slice(1) === r) render();
}

export function nav(cur) {
  const side = $("#side");
  if (!side) return;
  side.innerHTML =
    `<a class="brand" href="#dashboard"><span class="mark">F</span>FineTunePilot</a>` +
    NAV.map(
      g =>
        `<div class="grp">${g[0]}</div>` +
        g[1]
          .map(
            p =>
              `<a class="nl" href="#${p[0]}" ${p[0] === cur ? 'aria-current="page"' : ""}>${p[1]}</a>`
          )
          .join("")
    ).join("");
}

export function render() {
  const r = location.hash.slice(1);
  const k = P[r] ? r : "dashboard";
  nav(k);

  const appEl = $("#app");
  if (!appEl) return;
  appEl.innerHTML = `<div class="view">${P[k]()}</div>`;

  const side = $("#side");
  if (side) side.classList.remove("open");
  const mb = $("#mb");
  if (mb) mb.setAttribute("aria-expanded", "false");

  const f = $("#f");
  if (f) {
    f.onchange = e => {
      const x = e.target.files[0];
      if (x) {
        S.dsErr = "";
        x.text().then(t => useDS(x.name, t, false));
      }
    };
  }

  const bind = (id, fn) => {
    const e = $(id);
    if (e)
      e.onchange = () => {
        fn(e.value);
        render();
      };
  };

  bind("#hw", v => {
    if (+v > 0) S.hw = String(+v);
    S.off = 1;
  });
  bind("#mt", v => {
    S.method = v;
    S.off = 1;
  });

  if (k === "compat" && S.off) {
    S.off = 0;
    if (ok()) setTimeout(() => offerNext("compat", "Everything looks compatible."), 700);
  }

  document.querySelectorAll("[data-c]").forEach(i => {
    i.onchange = () => {
      const k = i.dataset.c;
      S.cfg[k] = ["lr"].includes(k) ? i.value : Math.max(1, +i.value || 1);
      S.saved = false;
      render();
    };
  });
}

export const NX = {
  dataset: ["models", "Model Hub"],
  models: ["compat", "Compatibility & Preflight"],
  compat: ["config", "Training Configuration"],
  config: ["runs", "Training Runs & Logs"],
  runs: ["eval", "Model Evaluation"],
  eval: ["quant", "Quantization Jobs"],
  quant: ["artifacts", "Artifact Registry"],
  artifacts: ["serving", "Model Serving & vLLM"],
  serving: ["distributed", "Multi-GPU & Distributed"],
  distributed: ["settings", "Settings & Diagnostics"],
};

export function offerNext(step, msg) {
  const n = NX[step];
  const d = $("#nx");
  if (!n || !d || d.open) return;
  S.nxRoute = n[0];
  const nxt = $("#nxt");
  const nxm = $("#nxm");
  const nxg = $("#nxg");
  if (nxt) nxt.textContent = msg;
  if (nxm) nxm.textContent = "Ready for the next step. Confirm to continue, or stay to make changes.";
  if (nxg) nxg.textContent = "Continue to " + n[1];
  d.showModal();
}

if (typeof window !== 'undefined') {
  window.$ = $;
  window.go = go;
  window.nav = nav;
  window.render = render;
  window.offerNext = offerNext;
}
