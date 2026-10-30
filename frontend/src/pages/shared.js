// Page registry and shared page helpers
export const P = {};
export const next = (r, l) => `<div class="row" style="margin-top:28px"><a class="btn p" href="#${r}">${l}</a></div>`;
export const hdr = (t, sub) =>
  `<div class="crumb"><a href="#dashboard">Dashboard</a> / ${S.proj ? S.proj + " / " : ""}${t}</div><h1>${t}</h1><p class="mu">${sub}</p>`;

if (typeof window !== 'undefined') {
  window.P = P;
  window.next = next;
  window.hdr = hdr;
}
