# Recurrent Models

`affon:nn` does not currently expose an RNN or LSTM factory.
New Programs can express supported recurrence by composing `affon:ops` and
explicit Program state, but there is not yet a stable high-level recurrent
factory in `affon:nn`.

There are no public high-level `SimpleRNN`, `RNN`, or `LSTM` helpers. Recurrent
models must currently be authored from supported operations and explicit state.

Current canonical `affon:nn` coverage is documented in
[NN Concepts](./index.md) and
[Linear, Embedding, and Normalization](./basic.md).
