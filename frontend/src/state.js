// Single in-memory app state connected to backend API
export const S = {
  proj: null,
  ds: null,
  model: null,
  hw: "16",
  method: "lora",
  cfg: { epochs: 3, lr: "0.0002", batch: 8, rank: 16, len: 512 },
  saved: false,
  runs: [],
  activeRun: null,
  arts: [],
  evals: [],
  log: [],
  prog: 0,
  timer: null,
  hardwareInfo: null,
  modelsList: [],
  datasetsList: [],
  k8sStatus: null,
  servingCaps: null,
  quantCaps: [],
  backendOnline: false,
  tlog: ['Type "help" for commands.'],
};

if (typeof window !== 'undefined') {
  window.S = S;
}
