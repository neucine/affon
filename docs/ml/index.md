# Machine Learning Docs

Machine-learning docs cover the layers that sit on top of `affon:compute`:
model construction, optimizers, losses, metrics, datasets, checkpoints, and
beginner terminology.

## Read First

- [NN Concepts](./nn/index.md) - model-building concepts and topic guides.
- [Optim Concepts](./optim.md) - optimizer steps, schedules, and training-loop control.
- [Metrics Concepts](./metrics.md) - evaluation metrics built on compute tensors.
- [Text Datasets](./text-datasets.md) - text records, tokenization, batching, and tensor export.
- [Checkpoints](./checkpoints.md) - saving and restoring training state.
- [ML Glossary](./glossary.md) - shared terms used throughout the ML docs.

## Public Boundaries

- `p.nn` owns parameterized neural-network declarations inside a Program.
- `affon:compute` owns tensors, Programs, Sessions, differentiation, built-in
  loss templates under `losses`, and Program transforms such as `optimize`.
- `affon:ops` owns tensor operations shared by formal and evaluated tensors.
- `affon:optim` owns optimizer descriptors.
- `affon:dataset` owns ingest, transforms, batching, and tensorization.
- `affon:checkpoint` owns resume-oriented persistence.
- First-party packages own reusable model-family or domain conventions.
- Apps own complete runnable workflows that combine data, training, evaluation, checkpointing, and reporting.

## Topic Index

- [ML Glossary](./glossary.md)
- [Metrics Concepts](./metrics.md)
- [Optim Concepts](./optim.md)
- [NN Concepts](./nn/index.md)
- [Checkpoints](./checkpoints.md)
- [Text Datasets](./text-datasets.md)
