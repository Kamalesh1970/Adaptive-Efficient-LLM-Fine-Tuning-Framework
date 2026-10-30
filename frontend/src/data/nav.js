// Sidebar groups and page titles
export const NAV = [
  ["Overview", [["dashboard", "Dashboard"]]],
  ["Project setup", [["dataset", "Dataset Inspector"], ["models", "Model Hub"], ["compat", "Compatibility Checker"]]],
  ["Training", [["config", "Training Configuration"], ["runs", "Training Runs"]]],
  ["Evaluation & artifacts", [["eval", "Evaluation and Export"], ["artifacts", "Artifact Library"]]],
  ["About", [["settings", "Settings and Project Info"]]],
];

export const TITLES = {};
NAV.forEach(g => g[1].forEach(p => (TITLES[p[0]] = p[1])));

if (typeof window !== 'undefined') {
  window.NAV = NAV;
  window.TITLES = TITLES;
}
