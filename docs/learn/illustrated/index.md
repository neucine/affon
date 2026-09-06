# Illustrated Learning Coverage

This section tracks the beginner-facing illustration layer for Affon.

The goal is to cover the public compute, metrics, optimizer, and basic neural-network surfaces with a consistent teaching pattern:

```txt
intuition
math rule
shape / dim-role illustration
small numeric example
predict-before-run prompt
runnable Affon cell
interpretation of the output
common mistakes and debugging cues
common meaning in ML code
```

Illustration and math are necessary but not sufficient. The learner should repeatedly move through this loop:

```txt
see the idea -> predict a tiny result -> run it -> interpret the output -> connect it to ML code
```

## Visual Grammar

Use the same dimension colors and words everywhere:

| Role | Meaning |
| --- | --- |
| parallel dim | Selects independent work, such as batch, time, head, or position. |
| operation dim | Participates directly in the scalar/vector/matrix rule. |
| reduced dim | Is summarized away by an op such as `sum`, `mean`, `max`, or `argmax`. |
| produced dim | Is newly created by projection, stacking, one-hot encoding, or top-k selection. |

For high-rank tensors, the default explanation should be:

```txt
Find the operation dimensions.
Everything else is parallel.
The same small operation runs independently at every parallel position.
```

## Teaching Pattern

Each illustrated concept should use this order:

1. What problem it solves.
2. Intuition in plain language.
3. Math formula when the op has a meaningful formula.
4. SVG diagram showing shape roles, data movement, or curve behavior.
5. Tiny numeric example with values small enough to compute by eye.
6. Predict-before-run prompt.
7. Runnable Affon example.
8. Interpretation of the output.
9. High-rank reading rule.
10. Common beginner mistakes and debugging cues.
11. Where it appears in real ML code.

For ops without useful scalar math, replace the formula with an indexing rule or shape rule.

## Learning Blocks

Use these recurring blocks inside notebooks.

### Why This Exists

Explain the practical reason first. Example:

```txt
Softmax exists because a model often produces raw scores, but we want to compare those scores as a distribution over choices.
```

### Before You Run

Ask the learner to predict one small thing:

```txt
Before running this cell, which axis will disappear?
What shape do you expect?
Which number should be largest?
```

### Read The Output

Immediately after code, explain what changed:

```txt
The shape changed from [2, 2, 3] to [2, 2] because dim=2 was reduced.
Each [hidden=3] vector became one number.
```

### Debugging Cue

Name the likely beginner failure:

```txt
If the output shape surprises you, list every input axis and mark it as parallel, operation, reduced, or produced.
```

### ML Connection

Connect the primitive to a model-building use:

```txt
This is the same pattern used when a Linear layer projects every token embedding independently.
```

## Implemented Notebooks

| Notebook | Purpose |
| --- | --- |
| `examples/illustrated/tensor-illustrated.tsnb` | Covers tensor dim roles, construction, elementwise ops, activations, reductions, softmax, shape changes, and selection. |
| `examples/illustrated/tensor-ops-illustrated.tsnb` | Teaches tensor ops one by one, grouping only the obvious elementwise families. |
| `examples/illustrated/nn-illustrated.tsnb` | Covers basic `nn` modules, sequence blocks, masks, positions, initialization context, and diagnostics context. |
| `examples/illustrated/losses-metrics-illustrated.tsnb` | Covers regression, binary, multiclass losses, and classification/regression metrics. |
| `examples/illustrated/training-illustrated.tsnb` | Covers parameters, gradients, clearing, clipping, optimizers, schedules, `no_grad`, and `copy`. |

## Coverage Groups

### Tensor Creation And Metadata

| Surface | Illustration focus |
| --- | --- |
| `tensor` | Nested arrays become tensor axes. |
| `empty`, `zeros`, `ones`, `full` | Shape creates storage; fill rule determines values. |
| `rand`, `randn`, `seed` | Random samples, reproducibility, distribution sketch. |
| `arange`, `linspace` | Number-line construction. |
| `shape`, `rank` / `ndim`, `dtype`, `device` | Tensor identity card. |
| `cast`, `move`, `contiguous` | Same logical values, different dtype/device/layout. |

### Elementwise Arithmetic

All dimensions are parallel and shape is preserved unless broadcasting applies.

| Surface | Math / rule |
| --- | --- |
| `add` | `z = a + b` |
| `sub` | `z = a - b` |
| `mul` | `z = a * b` |
| `div` | `z = a / b` |
| `neg` | `z = -x` |
| `abs` | `z = |x|` |
| `sign` | negative -> `-1`, zero -> `0`, positive -> `1` |
| `exp` | `z = e^x` |
| `log` | `z = log(x)` |
| `sqrt` | `z = sqrt(x)` |
| `square` | `z = x^2` |
| `clamp` | values below/above limits are clipped |
| `where` | choose from `a` or `b` using a condition mask |
| `masked_fill` | replace positions selected by a mask |
| `gt_scalar` | compare each value to a threshold |

### Activations

All dimensions are parallel and shape is preserved.

| Surface | Math / visual focus |
| --- | --- |
| `relu` | hard zero floor: `max(x, 0)` |
| `sigmoid` | squeeze to `(0, 1)` |
| `tanh` | squeeze to `(-1, 1)` and center at zero |
| `gelu` | smooth ReLU-like gate |
| `silu`, `swish` | `x * sigmoid(x)` |

### Linear Algebra

| Surface | Illustration focus |
| --- | --- |
| `dot` | pairwise multiply then sum. |
| `matmul` | row-by-column multiply; leading dims are parallel. |

For high-rank `matmul`, use:

```txt
[parallel..., m, k] @ [parallel..., k, n] -> [parallel..., m, n]
```

### Reductions And Normalization

| Surface | Illustration focus |
| --- | --- |
| `sum` | reduce selected axis by adding. |
| `mean` | reduce selected axis by averaging. |
| `max`, `min` | reduce selected axis by selecting extreme value. |
| `variance`, `std` | measure spread around the mean. |
| `argmax`, `argmin` | return the position of the extreme value. |
| `softmax` | normalize one axis into a probability distribution. |

For reductions:

```txt
parallel over: all non-reduced dims
reduce over: selected dim or dims
```

For `softmax`:

```txt
parallel over: all non-softmax dims
normalize over: selected dim
shape: preserved
```

### Shape And Selection

| Surface | Illustration focus |
| --- | --- |
| `reshape` | same values, new axis grouping. |
| `transpose`, `permute` | reorder axes. |
| `squeeze`, `unsqueeze` | remove or insert size-1 axes. |
| `cat` | join tensors along an existing axis. |
| `stack` | create a new axis for a list of tensors. |
| `slice`, `at`, `range` | select positions or ranges. |
| `gather` | use an index tensor to collect values. |
| `index_select` | select rows/items along one axis. |
| `topk` | produce top values and their indices. |
| `one_hot` | turn class ids into indicator vectors. |

### Autograd And Training Control

| Surface | Illustration focus |
| --- | --- |
| `parameter` | trainable tensor state. |
| `grad` | loss sends gradients back to parameters. |
| `clear_grad` | remove stored gradients before the next batch. |
| `clip_grad_norm` | shrink gradients when the global vector is too large. |
| `no_grad` | run computation without recording a training graph. |
| `copy` | overwrite one tensor with another. |

Core training-loop picture:

```txt
data -> model -> prediction -> loss
                         |
                         v
parameters <- optimizer <- gradients
```

### Metrics

Metrics should use visual examples before formulas.

| Surface | Illustration focus |
| --- | --- |
| `accuracy` | correct predictions divided by total predictions. |
| `precision` | true positives divided by predicted positives. |
| `recall` | true positives divided by actual positives. |
| `f1` | balance between precision and recall. |
| `mse` | average squared error. |
| `mae` | average absolute error. |
| `r2` | explained variation versus predicting the mean. |

Classification metrics should share one confusion-matrix SVG:

```txt
                actual
              yes   no
pred yes      TP    FP
pred no       FN    TN
```

### Optimizers And Schedules

| Surface | Illustration focus |
| --- | --- |
| `sgd` | move opposite the gradient by learning-rate-sized steps. |
| `adam` | use running averages of gradients and squared gradients. |
| `adamw` | Adam with decoupled weight decay. |
| `scheduled` | wrap an optimizer step with a learning-rate rule. |
| `schedules.constant` | flat learning rate. |
| `schedules.linear` | linearly interpolate learning rate. |
| `schedules.cosine` | smooth decay curve. |
| `schedules.step` | drop the learning rate at fixed intervals. |
| `schedules.sequence` | stitch schedule phases together. |
| `Duration.steps`, `Duration.epochs` | measure schedule length by step or epoch. |

Optimizer illustrations should avoid advanced calculus first. Use:

```txt
gradient = slope direction
learning rate = step size
optimizer = rule for changing parameters
```

### Basic NN Building Blocks

| Surface | Illustration focus |
| --- | --- |
| `nn.Linear` | `x @ weight + bias`; leading dims are parallel. |
| `nn.Embedding` | token ids select rows from an embedding table. |
| `nn.Sequential` | pipe output of one module into the next. |
| `nn.module_list` | own repeated submodules. |
| `nn.Dropout` | randomly zero activations during training. |
| `nn.LayerNorm` | normalize over the last feature axis. |
| `nn.BatchNorm` | normalize features using batch statistics. |
| `nn.SimpleRNN`, `nn.RNN` | hidden state updated through time. |
| `nn.LSTM` | recurrent state with gates and cell state. |
| `nn.MSELoss` | regression loss from squared errors. |
| `nn.BCELoss` | binary loss from probabilities. |
| `nn.BCEWithLogitsLoss` | binary loss from logits. |
| `nn.CrossEntropyLoss` | logits plus target class produce classification loss. |
| `nn.causal_mask`, `nn.apply_causal_mask` | prevent attending to future positions. |
| `nn.sinusoidal_encoding`, `nn.position_ids` | provide position information. |
| `nn.init.*` | initialize trainable weights. |
| `nn.diagnostics.*` | check model values while running. |

## First Pass Priority

Start with the concepts that unblock the most beginner confusion:

1. Dim roles and high-rank reading rule.
2. Activations.
3. `matmul` and `nn.Linear`.
4. Reductions and `softmax`.
5. `CrossEntropyLoss` and classification metrics.
6. `parameter`, `grad`, `clear_grad`, and optimizers.
7. `Embedding`, sequence dims, and recurrent modules.
8. Shape and selection ops.
9. Remaining constructors, diagnostics, and utility helpers.

## Notebook Families

Use category notebooks rather than one notebook per function:

| Notebook | Covers |
| --- | --- |
| `tensor-illustrated.tsnb` | dim roles, construction, metadata, elementwise ops, activations, reductions, softmax, shape, selection |
| `tensor-ops-illustrated.tsnb` | op-by-op tensor teaching for constructors, math, activations, linear algebra, reductions, masks, shape, and selection |
| `nn-illustrated.tsnb` | Linear, Embedding, Sequential, normalization, Dropout, recurrent blocks, masks, positions |
| `losses-metrics-illustrated.tsnb` | losses plus classification/regression metrics |
| `training-illustrated.tsnb` | gradients, optimizer steps, clipping, schedules |

## Completion Definition

A concept is covered when it has:

1. A concise beginner explanation.
2. Formula, indexing rule, or shape rule.
3. SVG or plotted illustration.
4. High-rank dimension-role explanation when relevant.
5. A tiny runnable Affon example.
6. A common mistakes section.
