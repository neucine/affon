# Metrics Concepts

Use metrics to evaluate model quality from `affon:compute`.

A metric tells you how good predictions are. It does **not** update parameters and does **not** build an autograd graph.

## Metric vs Loss

- `loss`: training objective, usually used before `grad(loss, params)`
- `metric`: evaluation number for reporting or comparison

Examples:

- train with `CrossEntropyLoss`
- report `accuracy`

or:

- train with `MSELoss`
- report `mae` and `r2`

Metrics return plain JavaScript numbers.

Use this notation in the formulas below:

- $\hat{y}$: model output or prediction
- $y$: target
- $\hat{c}$: predicted class label
- $c$: target class label
- $N$: number of examples in the batch

In Affon function signatures, this usually maps as:

- `pred` $\rightarrow$ $\hat{y}$
- `target` $\rightarrow$ $y$

For classification metrics:

- binary case: `pred` is usually a score in `(0, 1)`, and thresholding turns it into $\hat{c}$
- multiclass case: `pred` is usually logits of shape `[N, C]`, and `argmax` turns each row into $\hat{c}$

## Classification Metrics

Current classification metrics:

- `accuracy`
- `precision`
- `recall`
- `f1`

### Expected Inputs

| Metric | Typical `pred` input | Typical `target` input |
| --- | --- | --- |
| `accuracy` (binary) | probability-like scores or binary labels shaped `[N]` or `[N, 1]` | binary labels shaped `[N]` or `[N, 1]` |
| `accuracy` (multiclass) | logits or class scores of shape `[N, C]` | class labels of shape `[N]` |
| `precision` | probability-like scores or binary labels shaped `[N]` or `[N, 1]` | binary labels shaped `[N]` or `[N, 1]` |
| `recall` | probability-like scores or binary labels shaped `[N]` or `[N, 1]` | binary labels shaped `[N]` or `[N, 1]` |
| `f1` | probability-like scores or binary labels shaped `[N]` or `[N, 1]` | binary labels shaped `[N]` or `[N, 1]` |

For binary metrics:

- if `pred` is already binary, thresholding is unnecessary
- if `pred` is continuous, the threshold converts it into predicted labels
- `[N]` and `[N, 1]` are both accepted for binary paths

For multiclass `accuracy`:

- `pred` is usually logits or class scores
- the winning class is selected with `argmax`

### Accuracy

Accuracy is the fraction of correct predictions:

$$
\mathrm{accuracy} = \frac{\text{correct}}{\text{total}}
$$

Equivalent batch-wise form:

$$
\mathrm{accuracy} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\hat{c}_i = c_i]
$$

It is easy to understand, but it can be misleading on imbalanced data.

### Precision

Precision answers:

"Of the items predicted positive, how many were actually positive?"

$$
\mathrm{precision} = \frac{TP}{TP + FP}
$$

with batch-wise counts:

`TP` means true positives: predicted positive and actually positive.

$$
TP = \sum_{i=1}^{N}\mathbf{1}[\hat{c}_i = 1 \land c_i = 1]
$$

`FP` means false positives: predicted positive but actually negative.

$$
FP = \sum_{i=1}^{N}\mathbf{1}[\hat{c}_i = 1 \land c_i = 0]
$$

Use it when false positives are costly.

### Recall

Recall answers:

"Of the items that were actually positive, how many did the model find?"

$$
\mathrm{recall} = \frac{TP}{TP + FN}
$$

with:

`TP` means true positives: predicted positive and actually positive.

$$
TP = \sum_{i=1}^{N}\mathbf{1}[\hat{c}_i = 1 \land c_i = 1]
$$

`FN` means false negatives: predicted negative but actually positive.

$$
FN = \sum_{i=1}^{N}\mathbf{1}[\hat{c}_i = 0 \land c_i = 1]
$$

Use it when missing positives is costly.

### F1

F1 balances precision and recall:

$$
F1 = \frac{2 \cdot \mathrm{precision} \cdot \mathrm{recall}}{\mathrm{precision} + \mathrm{recall}}
$$

So in batch terms, `F1` is computed from the same batch-wise `TP`, `FP`, and `FN` counts through precision and recall.

Useful when you want one binary-classification score and classes are imbalanced.

## Thresholding

For binary classification, predicted scores are often continuous values in `(0, 1)`.

A threshold turns them into class decisions.

With threshold $t$:

$$
\hat{c}_i =
\begin{cases}
1 & \text{if } \hat{y}_i > t \\
0 & \text{otherwise}
\end{cases}
$$

Example with threshold `0.5`:

```txt
0.82 -> positive
0.23 -> negative
```

In the current metrics helpers, binary metrics use `0.5` by default, and you can override it:

```ts
import { precision } from 'affon:compute'
precision(pred, target, { threshold: 0.7 })
```

## Multiclass Classification

For multiclass logits shaped `[N, C]`, `accuracy` compares the winning class index against the target label.

Mental model:

- each row = one example
- each column = one class score
- predicted class = largest score in that row

$$
\hat{c}_i = \arg\max_{j} \hat{y}_{ij}
$$

## Regression Metrics

Current regression metrics:

- `mse`
- `mae`
- `r2`

### Expected Inputs

For regression metrics:

- `pred` should be continuous model output
- `target` should be continuous target values
- `pred` and `target` should have compatible shapes

### Mean Squared Error

$$
\mathrm{MSE} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2
$$

Larger mistakes count more heavily.

### Mean Absolute Error

$$
\mathrm{MAE} = \frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i - y_i|
$$

This is often easier to interpret in the original target units.

### $R^2$

$$
R^2 = 1 - \frac{\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{N}(y_i - \bar{y})^2}
$$

It measures how much of the target variation the predictions explain.

## Practical Pattern

Typical evaluation flow:

```txt
model output -> metric
```

Typical training flow:

```txt
model output -> loss -> backward -> optimizer.step()
```

So metrics help you judge model quality, while losses drive parameter updates.

## Which Metric When?

| Metric | ML case | Use when |
| --- | --- | --- |
| `accuracy` | Binary or multiclass classification | Classes are fairly balanced and you want one simple overall score. |
| `precision` | Binary classification | False positives are especially costly. |
| `recall` | Binary classification | False negatives are especially costly. |
| `f1` | Binary classification | You want one score that balances precision and recall, especially on imbalanced data. |
| `mse` | Regression | Larger regression errors should count more heavily. |
| `mae` | Regression | You want error in the original target units and less sensitivity to outliers than `mse`. |
| `r2` | Regression | You want to measure how much of the target variation the model explains. |
