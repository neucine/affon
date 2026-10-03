# Recurrent Models

The canonical Program builder does not currently expose an RNN or LSTM layer.
New Programs can express supported recurrence by composing `affon:ops` and
explicit Program state, but there is not yet a stable high-level recurrent
helper in `p.nn`.

There are no public high-level `SimpleRNN`, `RNN`, or `LSTM` helpers. Recurrent
models must currently be authored from supported operations and explicit state.

Current canonical `p.nn` coverage is documented in
[NN Concepts](./index.md) and
[Linear, Embedding, and Normalization](./basic.md).
