# Recurrent Models

The canonical Program builder does not currently expose an RNN or LSTM layer.
New Programs can express supported recurrence by composing `affon:ops` and
explicit Program state, but there is not yet a stable high-level recurrent
helper in `p.nn`.

The previous `SimpleRNN`, `RNN`, and `LSTM` callable modules remain available
from `affon:nn/legacy` for existing applications. They are not part of the new
Program API and should not be mixed into a canonical Program.

Current canonical `p.nn` coverage is documented in
[NN Concepts](./index.md) and
[Linear, Embedding, and Normalization](./basic.md).
