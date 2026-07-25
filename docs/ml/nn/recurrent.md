# NN Recurrent

Beginner-friendly mental models for `SimpleRNN`, `RNN`, and `LSTM`.

## SimpleRNN

`nn.SimpleRNN(...)` is the single-sequence version.

Input:

```txt
[seq_len, input_size]
```

Output:

```txt
[seq_len, hidden_size]
```

Mental model:
- one sequence
- one sequence position at a time
- one hidden state carried forward

At each sequence position:

$$
h_t = \mathrm{act}(x_t W_{ih} + h_{t-1} W_{hh} + b)
$$

Connectivity idea:

![Recurrent neural network and unfold view](https://commons.wikimedia.org/wiki/Special:FilePath/Recurrent_neural_network_unfold.svg)

Source:
- Wikimedia Commons, "Recurrent neural network unfold": https://commons.wikimedia.org/wiki/File:Recurrent_neural_network_unfold.svg

The hidden state at the current sequence position depends on:
- the current input vector
- the previous hidden state

## Batched RNN

`nn.RNN(...)` is the batched Elman RNN.

Default layout:

```txt
[seq_len, batch, input_size]
```

Output:

```txt
[seq_len, batch, hidden_size]
```

Example:

```txt
[8, 10, 3]
```

means:
- sequence length = 8
- batch size = 10
- input size = 3

Important:
- this is 10 sequences in one batch
- not 80 separate forwards

Inside `forward(...)`, the code loops over sequence positions:

```txt
for each sequence position:
  process the whole batch in parallel
```

So the sequential logic is:
- sequence positions

And the batched computation is:
- all batch rows at that position

At each position, the model processes the full batch in parallel.

## Sequence-first vs batch-first

These are the same logical data with different axis order.

Sequence-first:

```txt
[seq_len, batch, input_size]
```

Batch-first:

```txt
[batch, seq_len, input_size]
```

Example:

```txt
[8, 10, 3]  sequence-first
[10, 8, 3]  batch-first
```

Both mean:
- 10 sequences
- each length 8
- input width 3

Difference:
- sequence-first is more natural for recurrent implementation
- batch-first is often more natural for data pipelines

Visual comparison:

```txt
sequence-first: [seq_len, batch, input_size]
  X[0] = full batch at sequence position 0
  X[1] = full batch at sequence position 1

batch-first: [batch, seq_len, input_size]
  X[0] = one whole sequence in the batch
  X[1] = another whole sequence in the batch
```

## Hidden size vs input size

These are different.

- `input_size`
  width of the input vector
- `hidden_size`
  width of the recurrent hidden state

Example:

```txt
input_size = 3
hidden_size = 6
```

Then each sequence position gives:

```txt
x_t = [x0, x1, x2]
h_t = [h0, h1, h2, h3, h4, h5]
```

## LSTM

`nn.LSTM(...)` is like `RNN`, but with gated memory.

It keeps two states:
- hidden state
- cell state

Outputs:

```txt
output:       [seq_len, batch, hidden_size]
hidden_state: [num_layers, batch, hidden_size]
cell_state:   [num_layers, batch, hidden_size]
```

The idea:
- the hidden state is the exposed working state
- the cell state is the longer-lived memory

Connectivity idea:

![LSTM one-unit diagram](https://commons.wikimedia.org/wiki/Special:FilePath/Long_Short-Term_Memory.svg)

Source:
- Wikimedia Commons, "Long Short-Term Memory": https://commons.wikimedia.org/wiki/File:Long_Short-Term_Memory.svg
- The Wikimedia diagram is explicitly inspired by Christopher Olah's "Understanding LSTM Networks": https://research.google/pubs/understanding-lstm-networks/

So LSTM keeps two recurrent paths:
- hidden state path
- cell-state memory path

Compact math:

$$
i_t = \sigma(\cdots), \quad
f_t = \sigma(\cdots), \quad
g_t = \tanh(\cdots), \quad
o_t = \sigma(\cdots)
$$

$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

## Current built-in assumptions

- recurrent connections are fully connected
- stacked recurrent layers use one shared `hidden_size`
- the first recurrent layer maps `input_size -> hidden_size`
- later recurrent layers map `hidden_size -> hidden_size`
- `SimpleRNN` is unbatched
- `RNN` and `LSTM` are batched
- built-in recurrent modules are unidirectional
- no dropout is applied between recurrent layers
- no variable-length packing utilities are built in
- current parity coverage includes sequence-first and `batch_first` paths for single-layer `RNN` and `LSTM`

## Stacked recurrent layers

With `num_layers > 1`, the output of one recurrent layer becomes the input of the next.

Visual idea:

```txt
input sequence
  -> recurrent layer 1
  -> recurrent layer 2
  -> recurrent layer 3
  -> output sequence
```

For the current built-in layers:
- all stacked recurrent layers share the same `hidden_size`
- only the first layer sees the original `input_size`
- later layers see the previous layer's `hidden_size`
