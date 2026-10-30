// Plain-language explanations used by the pilot and the parameter readout
export const GL = {
  lora: "LoRA trains small add-on weights and leaves the base model frozen, so it needs much less memory.",
  qlora: "QLoRA is LoRA on a compressed (quantised) base model. Lowest memory, slightly slower.",
  epoch: "One full pass over your training data.",
  lr: "Learning rate: the size of each update step. Too high can destabilise training.",
  batch: "Batch size: examples processed per step. Lower it when memory is tight.",
  rank: "LoRA rank: adapter size. Higher can learn more but costs memory.",
  gguf: "GGUF is a file format for running a finished model locally. It is an export format, not a training method.",
  adapter: "A LoRA adapter is the small set of trained weights, kept as its own artifact.",
  overfitting: "When a model memorises training examples and does worse on new ones.",
};

export const RO = {
  epochs: "Passes over the data. More can help until the model starts memorising. Affects: run time, overfitting.",
  lr: "Size of each learning step. Too high is unstable, too low is slow. Affects: stability.",
  batch: "Examples per step. Lower values save memory. Affects: memory, speed.",
  rank: "Adapter size for LoRA and QLoRA. Affects: memory, how much can be learned.",
  len: "Longest example in tokens. Affects: memory and the model's context limit.",
};

if (typeof window !== 'undefined') {
  window.GL = GL;
  window.RO = RO;
}
