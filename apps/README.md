# Apps

Top-level apps are complete runnable domain workloads.

Use this area when an example combines several concerns, such as:

- data loading or preprocessing
- model assembly
- training or evaluation
- checkpointing or resume
- reporting, generation, or exported artifacts

Small API demonstrations should stay close to the package or runtime surface they demonstrate.

## Current Apps

- `hf-inference/` audits pretrained Hugging Face inference against independent references.
- `decoder-lm/` is the decoder language-model reference workload.
- [model-package/](model-package/README.md) experiments with a shared artifact manifest and local runtime adapters.
- [vision-serving/](vision-serving/README.md) compares an Affon V2 inference service with an ONNX Runtime baseline and provides a KServe deployment example.
