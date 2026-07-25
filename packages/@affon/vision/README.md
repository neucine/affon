# @affon/vision

Vision-domain package scaffold.

This package is reserved for image-domain APIs that are broader than one app but more specific than runtime primitives.

Expected ownership:

- image model assembly
- image classification helpers
- patch embedding and ViT-facing glue
- vision checkpoint conventions
- package-level image transforms that have not earned runtime placement

Not owned here:

- generic transformer blocks, which belong in `../transformers/`
- CNN architecture blocks, which should move to `../cnn/` if they become broadly reusable
- complete runnable workflows, which should live under `../../../apps/`
