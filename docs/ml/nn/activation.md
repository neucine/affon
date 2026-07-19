# NN Activation

Activation functions introduce non-linearity.

Without them, stacked linear layers would still behave like a single linear transform.

## ReLU

`relu(x)`

$$
\mathrm{relu}(x) = \max(x, 0)
$$

Use when you want a simple piecewise-linear activation.

Curve:

![Rectifier and softplus functions](https://commons.wikimedia.org/wiki/Special:FilePath/Rectifier_and_softplus_functions.svg)

Source:
- Wikimedia Commons, "Rectifier and softplus functions": https://commons.wikimedia.org/wiki/File:Rectifier_and_softplus_functions.svg

What to notice:
- all negative inputs become `0`
- positive inputs pass through linearly
- the curve is flat on the left and linear on the right

## Sigmoid

`sigmoid(x)`

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

Maps values into the continuous interval:

```txt
(0, 1)
```

Common for probability-like outputs in binary classification.

A binary decision usually comes later by applying a threshold, for example:

```txt
sigmoid(x) > 0.5 -> class 1
sigmoid(x) <= 0.5 -> class 0
```

Curve:

![Logistic sigmoid curve](https://commons.wikimedia.org/wiki/Special:FilePath/Logistic-curve.svg)

Source:
- Wikimedia Commons, "Logistic curve": https://commons.wikimedia.org/wiki/File:Logistic-curve.svg

What to notice:
- the curve is S-shaped
- very negative values go near `0`
- very positive values go near `1`
- the center is around `0.5`
- outputs stay continuous, not binary

## SiLU / Swish

`silu(x)` or `swish(x)`

$$
\mathrm{silu}(x) = x \cdot \sigma(x)
$$

SiLU keeps the smooth gating behavior of sigmoid, but unlike plain sigmoid it does not squash everything into `(0, 1)`.

What to notice:
- negative values are suppressed smoothly instead of hard-clipped
- positive values pass through with a learned-looking smooth bend
- `swish` is an alias of `silu`

## Tanh

`tanh(x)`

$$
\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}
$$

Maps values into the continuous interval:

```txt
(-1, 1)
```

Unlike sigmoid, it is centered around zero.

Curve:

![Hyperbolic tangent curve](https://commons.wikimedia.org/wiki/Special:FilePath/Hyperbolic_Tangent.svg)

Source:
- Wikimedia Commons, "Hyperbolic Tangent": https://commons.wikimedia.org/wiki/File:Hyperbolic_Tangent.svg

What to notice:
- the curve is also S-shaped
- outputs saturate near `-1` and `1`
- unlike sigmoid, it is symmetric around zero

## Softmax

`softmax(x, axis)`

$$
\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

Softmax converts values along one dimension into a probability distribution.
Each output stays continuous, and all outputs along that dimension sum to `1`.

Example:

```txt
[2.0, 1.0, 0.1] -> probabilities that sum to 1
```

Softmax is different from the scalar activations above:
- `relu`, `sigmoid`, and `tanh` act element-by-element
- `softmax` acts across a whole dimension

So it is usually better understood as:
- a normalization across logits
- not as a single 2D curve

Visual idea:

```txt
logits
  [2.0, 1.0, 0.1]

exp(logits)
  [7.39, 2.72, 1.11]

sum
  11.22

softmax
  [0.66, 0.24, 0.10]
```

So the biggest logit becomes the biggest probability, but all outputs stay positive and sum to `1`.
