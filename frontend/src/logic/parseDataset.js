// CSV / JSON / JSONL parsing with missing-value and duplicate checks
export function parse(name, txt) {
  let rows = [];
  const ext = name.split(".").pop().toLowerCase();
  if (ext === "jsonl") {
    rows = txt
      .split(/\n/)
      .filter(x => x.trim())
      .map(x => JSON.parse(x));
  } else if (ext === "json") {
    const j = JSON.parse(txt);
    rows = Array.isArray(j) ? j : [j];
  } else {
    const L = txt.trim().split(/\r?\n/);
    const sp = l => {
      const o = [];
      let c = "",
        q = false;
      for (const ch of l) {
        if (ch === '"') q = !q;
        else if (ch === "," && !q) {
          o.push(c);
          c = "";
        } else c += ch;
      }
      o.push(c);
      return o;
    };
    const h = sp(L[0]);
    rows = L.slice(1).map(l => {
      const v = sp(l);
      return Object.fromEntries(h.map((k, i) => [k.trim(), v[i] ?? ""]));
    });
  }
  const cols = [...new Set(rows.flatMap(r => Object.keys(r)))];
  const miss = cols.map(c => [c, rows.filter(r => r[c] == null || String(r[c]).trim() === "").length]);
  const seen = new Set();
  let dup = 0;
  rows.forEach(r => {
    const k = JSON.stringify(r);
    seen.has(k) ? dup++ : seen.add(k);
  });
  const issues = miss.filter(x => x[1]).length + (dup ? 1 : 0);
  return { name, rows, cols, miss, dup, issues };
}

if (typeof window !== 'undefined') {
  window.parse = parse;
}
