# ML Glossary

This file normalizes the terms used in Affon docs, examples, tests, and discussions.

## Preferred Language

- Use one term consistently when possible.
- If code and prose need different forms, use:
  - prose-friendly English in documentation
  - exact API spelling in code
- When two terms are both valid, prefer the one listed here as the primary term.

## Project Terms

- `Affon`
  The project / runtime name in prose.

- `affon:*`
  A built-in module specifier in code, such as `affon:compute`, `affon:nn`, or `affon:dataset`.

- `module`
  The primary structural term for `affon:nn`.
  Use this for:
  - `compute.module(...)`
  - built-in `nn` layers and submodules
  - submodules
  - module traversal

- `model`
  A user-facing term for a composed module used for training or inference.
  In prose, `model` is acceptable for end-to-end examples.
  In API design discussion, prefer `module` when precision matters.

- `layer`
  A specific kind of module such as `Linear`, `RNN`, `LSTM`, `Dropout`.
  A layer is a module; not every module needs to be described as a layer.

## Tensor Layout

- `shape`
  The sizes of a tensor's axes.
  Example: `[8, 10, 3]`

- `dimension`
  One axis position in the shape.

- `axis`
  Usually interchangeable with `dimension`.

- `stride`
  The memory step used to advance by one index along a given axis.

- `reshape`
  Reinterpret tensor data with a new shape without reordering axes.
  Often zero-copy when memory layout is compatible.

- `transpose`
  Swap axes, usually in a 2D context.

- `permute`
  General axis reordering for N-dimensional tensors.
  Changing between sequence-first and batch-first is a permute, not a plain reshape.

- `zero-copy view`
  A tensor view that reuses the same underlying storage without allocating new data.
  Typical examples: reshape, transpose, slice.
  Some downstream ops may still require a contiguous materialization later.

- `contiguous`
  A tensor layout materialized into a dense buffer suitable for kernels that require contiguous memory.

## Types And Numeric Terms

- `dtype`
  The exact code/API term for tensor numeric type, such as `f32` or `f64`.

- `data type`
  Acceptable prose synonym for `dtype`.
  In API docs and examples, prefer `dtype`.

- `shape-aware`
  Preferred hyphenated prose form for compile-time shape reasoning.

- `shape aware`
  Avoid in prose; use `shape-aware`.

- `one-hot`
  Preferred hyphenated prose and API-adjacent form.

- `one hot`
  Avoid in prose; use `one-hot`.

- `logits`
  Raw, unnormalized model outputs before `softmax`.
  In classification, logits are typically turned into probabilities with `softmax`, or passed directly into cross-entropy loss.

## Training Loop

- `forward`
  One model evaluation on the given input tensor.
  In Affon, this is just the module call: `model(x)`.
  For an RNN, one model call processes the whole sequence, not just one token position.

- `training iteration`
  One full batch pass through:
  `forward -> loss -> grad(loss, params) -> optimizer.step()`

- `optimizer step`
  A parameter update performed by the optimizer after gradients are computed.
  Do not confuse this with a recurrent sequence position.

- `epoch`
  One full pass over the training dataset.

- `loss`
  The scalar objective used for gradient computation.
  In Affon docs, prefer the explicit compute-style phrasing `grad(loss, params)` over method-centric `loss.backward()` teaching.

- `autograd`
  Reverse-mode automatic differentiation over parameters and compute graphs.

- `gradient`
  The derivative stored on trainable parameters after `grad(loss, params)`.

## Parameters And Initialization

- `parameter`
  A trainable tensor created with `compute.parameter(...)`.

- `named parameter`
  A parameter paired with its serialized/module traversal name.

- `parameter collection`
  The readonly array-like value returned by `model.parameters`.

- `initializer`
  An in-place policy such as Xavier or Kaiming used to fill trainable tensors.

- `weight init`
  Acceptable shorthand in discussion for parameter initialization.

## Persistence And State

- `state`
  The in-memory structured mapping returned by `model.state()`.
  This is the primary term for the structured in-memory representation.

- `restore`
  Applying an in-memory state mapping back into a module with `model.restore(dict)`.

- `save`
  Persisting model state to disk, typically with `model.save(path)`.
  For the module-level API, prefer `checkpoint.save(state, path)` from
  `affon:checkpoint`.

- `load`
  Loading model state from disk, typically with `model.load(path)`.
  For the module-level API, prefer `checkpoint.load(path)` from
  `affon:checkpoint`.

- `state dict`
  Acceptable descriptive synonym for `state`, especially when comparing to PyTorch.
  In Affon docs, prefer `state` as the primary term.

- `SafeTensors`
  The storage format name in prose.

- `safetensors`
  The literal file format string / filename extension in code and examples.

## Batching

- `batch`
  A group of independent examples processed together for computational efficiency.
  In NLP, a batch usually contains multiple unrelated sequences.

- `batch size`
  Number of examples in one batch.

- `batched input`
  Model input that contains multiple examples in one tensor.

- `unbatched input`
  Model input for a single example only.

- `sequence-first`
  Layout where sequence length is the leading structural axis.
  For recurrent models this usually means:
  `[seq_len, batch, input_size]`

- `batch-first`
  Layout where batch is the leading axis.
  For recurrent models this usually means:
  `[batch, seq_len, input_size]`

- `batch_first`
  The exact option name in code for batch-first recurrent input/output layout.
  Use `batch-first` in prose and `batch_first` in code.

## Recurrent Models

- `sequence`
  An ordered list of inputs processed left-to-right by a recurrent model.
  In NLP, this is typically a tokenized sentence after embedding.

- `sequence position`
  One index inside a sequence.
  In NLP, this is the clearest term for what many libraries call a "time step".

- `token position`
  NLP-specific synonym for `sequence position`.

- `time step`
  Often used interchangeably with `sequence position`, especially in generic sequence modeling and time-series work.
  In Affon docs, prefer `sequence position` for NLP-oriented examples.

- `input vector`
  The feature vector consumed at one sequence position.
  In NLP, this is typically one token embedding.

- `input_size`
  Width of the input vector at one sequence position.
  In NLP, this is often the embedding size.

- `hidden_size`
  Width of the recurrent hidden state.
  This is the number of recurrent hidden units / neurons.

- `hidden unit`
  Interchangeable with `hidden neuron` in recurrent-layer discussions.

- `hidden state`
  The recurrent state carried forward across sequence positions.
  For stacked recurrent models, this is usually one hidden state per layer.

- `cell state`
  The long-lived gated memory in an LSTM.
  This term is LSTM-specific.

## Notes on Interchangeable Terms

- `sequence position`, `token position`, and `time step`
  These are often used interchangeably.
  In Affon docs:
  - prefer `sequence position` as the general term
  - use `token position` in NLP-specific examples
  - use `time step` only when the domain is naturally temporal

- `hidden_size`, `units`, `hidden units`, `hidden neurons`
  These all refer to the width of the recurrent hidden state.

- `input_size`, `embedding size`, `feature width`
  These are related but not always identical.
  In NLP examples, `input_size` is often the embedding size.

## Examples

- single-sequence `SimpleRNN`
  `[seq_len, input_size] = [4, 3]`
  means one sequence, four sequence positions, input width three.

- sequence-first `RNN`
  `[seq_len, batch, input_size] = [8, 10, 3]`
  means ten sequences in the batch, each length eight, each position width three.

- batch-first `RNN`
  `[batch, seq_len, input_size] = [10, 8, 3]`
  means the same logical batch as above with different axis order.

- `LSTM` state
  `hidden_state()` and `cell_state()`:
  `[num_layers, batch, hidden_size]`

- training nesting
  `epoch -> batch -> forward -> sequence positions`
