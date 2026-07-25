# NN Loss

Beginner-friendly mental models for `affon:nn` loss functions.

Loss functions answer:

```txt
how wrong is the model on this example or batch?
```

Training uses the loss to produce gradients:

```txt
prediction -> loss -> grad(loss, params) -> gradients
```

More completely, the training chain is:

```txt
parameters -> model output -> loss -> parameter gradients
```

Or mathematically:

$$
\hat{y} = f(x; \theta)
$$

$$
L = L(\hat{y}, y)
$$

Equivalently, by substitution:

$$
L = L(f(x; \theta), y)
$$

$$
\frac{\partial L}{\partial \theta}
$$

Where:
- $\theta$ denotes the model parameters
- $\hat{y}$ is the prediction
- $y$ is the target

So although a loss is written in terms of prediction and target, training ultimately cares about how the loss changes with respect to the parameters.

## Autograd graph view

The model builds a graph up to the prediction:

$$
\hat{y} = f(x; \theta)
$$

Then the loss adds more nodes on top of that graph:

$$
L = L(\hat{y}, y)
$$

So the loss does not replace the model graph.
It extends it.

### Concrete example

Take a tiny linear model:

$$
\hat{y} = xw + b
$$

The model graph is:

```txt
x ----\
      multiply ----\
w ----/             add ----> y_hat
                    /
b -----------------/
```

If we use MSE loss:

$$
L = (\hat{y} - y)^2
$$

then the full graph becomes:

```txt
x ----\
      multiply ----\
w ----/             add ----> y_hat ---- subtract target ---- square ----> loss
                    /
b -----------------/
```

So MSE adds:
- subtract
- square

If you use a different loss, the extra nodes after `y_hat` are different, and therefore the gradient signal flowing back to the parameters is different.

### Leaf nodes

A leaf node is a starting tensor in the graph, not the result of another operation.

In the example above, the leaves are:
- `x`
- `w`
- `b`
- `y`

Why `x` is a leaf:
- it is given directly as input
- it is not produced by an earlier operation in this graph

Important:
- being a leaf does not mean "trainable parameter"
- `w` and `b` are usually trainable leaf parameters
- `x` is usually just an input leaf

## 1. General idea

A loss compares:
- model output
- target / ground truth

The result is usually a scalar tensor:

```txt
[1]
```

Smaller loss means:
- prediction is closer to target

Larger loss means:
- prediction is farther from target

## 2. MSELoss

`nn.MSELoss()`

Mean squared error:

$$
L_{\mathrm{MSE}} = \mathrm{mean}((prediction - target)^2)
$$

For a batch of $N$ scalar predictions, this is:

$$
L_{\mathrm{MSE}} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2
$$

Use when:
- the target is continuous
- this is a regression problem

Expected inputs:

| Loss | Typical prediction input | Typical target input |
| --- | --- | --- |
| `MSELoss` | continuous model output | continuous target values |

Examples:
- house price
- temperature
- sales forecast

Mental model:
- wrong by a small amount -> small penalty
- wrong by a large amount -> much larger penalty

## 3. BCELoss

`nn.BCELoss()`

Binary cross-entropy:

$$
L_{\mathrm{BCE}} = -\mathrm{mean}\left(target \log(pred) + (1 - target)\log(1 - pred)\right)
$$

For a batch of $N$ scalar probabilities, this is:

$$
L_{\mathrm{BCE}} = -\frac{1}{N}\sum_{i=1}^{N}\left(y_i \log(p_i) + (1 - y_i)\log(1 - p_i)\right)
$$

Use when:
- there are two classes
- the model output is already a probability-like value in `(0, 1)`

But `BCELoss` itself only requires probability-like inputs.

So `BCELoss` expects probabilities, not raw logits.

Expected inputs:

| Loss | Typical prediction input | Typical target input |
| --- | --- | --- |
| `BCELoss` | probability-like values in `(0, 1)` | binary targets `0` or `1` |

## 4. CrossEntropyLoss

`nn.CrossEntropyLoss()`

Cross-entropy for multiclass classification.

In Affon, the current form expects one-hot targets.

Formula idea:

$$
L_{\mathrm{CE}} = -\mathrm{mean}(target \cdot \log \mathrm{softmax}(logits))
$$

For a batch of $N$ examples and $C$ classes with one-hot targets, this is:

$$
L_{\mathrm{CE}} =
-\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{ic}\log\left(\mathrm{softmax}(z_i)_c\right)
$$

Use when:
- there are multiple classes
- the model outputs one logit per class

Typical pattern:

```txt
logits -> CrossEntropyLoss
```

Important:
- this loss is designed for logits
- you usually do not apply `softmax` yourself before passing logits into cross-entropy loss

Expected inputs:

| Loss | Typical prediction input | Typical target input |
| --- | --- | --- |
| `CrossEntropyLoss` | logits or class scores | one-hot targets |

## 5. Logits vs probabilities

This is one of the most common beginner confusions.

- `logits`
  raw, unnormalized scores from the model

- `probabilities`
  normalized values after sigmoid or softmax

Examples:

```txt
logits: [2.0, 1.0, 0.1]
softmax: [0.66, 0.24, 0.10]
```

## 6. Which loss with which output?

| ML case | Typical model output | Typical loss |
| --- | --- | --- |
| Regression | continuous model output | `MSELoss` |
| Binary classification | probability-like model output | `BCELoss` |
| Multiclass classification | logits over classes | `CrossEntropyLoss` |

Common binary-classification pattern:

```txt
logits -> sigmoid -> BCELoss
```

Common multiclass pattern:

```txt
logits -> CrossEntropyLoss
```
- loss handles the softmax-style comparison internally

## 7. Shapes

### MSELoss

Prediction and target usually have the same shape.

Example:

```txt
prediction: [batch, 1]
target:     [batch, 1]
```

### BCELoss

Prediction and target usually have the same shape.

Example:

```txt
prediction: [batch, 1]
target:     [batch, 1]
```

### CrossEntropyLoss

Current Affon expectation:

```txt
logits:  [batch, classes]
targets: [batch, classes]   // one-hot
```

## 8. Binary threshold vs training loss

For binary classification:

- training uses the continuous sigmoid output
- classification can later use a threshold such as `0.5`

Example:

```txt
sigmoid output: 0.82
threshold at 0.5 -> class 1
```

The threshold is for the final decision.
The loss is for training.

## 9. Quick cheat sheet

- regression
  `MSELoss`

- binary classification
  `probability-like output -> BCELoss`

- multiclass classification
  `logits -> CrossEntropyLoss`
