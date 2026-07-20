# Python to AFFON — Migration Guide

A side-by-side reference for Python ML developers moving to AFFON.

Naming note:
- `affon:compute` currently has a mixed public naming surface.
- Examples in this guide use the exact exported names, including `clear_grad(...)`, `clip_grad_norm(...)`, and `no_grad(...)`.

---

## Setup

| Python | AFFON |
|--------|------|
| `pip install numpy torch scikit-learn` | Single binary — `zig build` |
| `python train.py` | `affon train.ts` |
| Virtual environments, pip, conda | No dependency management needed |

---

## Tensors

### Creation

| NumPy / PyTorch | AFFON |
|-----------------|------|
| `np.array([1, 2, 3])` | `tensor([1, 2, 3])` |
| `np.zeros((3, 4))` | `zeros([3, 4])` |
| `np.ones((2, 2))` | `ones([2, 2])` |
| `torch.tensor([1, 2, 3])` | `tensor([1, 2, 3])` |
| `torch.zeros(3, 4)` | `zeros([3, 4])` |
| `torch.randn(3, 4)` | `randn([3, 4])` |
| `torch.rand(3, 4)` | `rand([3, 4])` |
| `torch.tensor([1.0], requires_grad=True)` | `parameter([1]).randn()` |

```typescript
import { tensor, parameter, zeros, randn, rand } from 'affon:compute'

const a = tensor([[1, 2, 3], [4, 5, 6]])
const W = parameter([2, 2]).xavier_uniform()
const x = randn([64, 10])
const y = tensor([[1], [0], [1]])
```

### Element-wise Ops

| NumPy / PyTorch | AFFON |
|-----------------|------|
| `a + b` | `add(a, b)` |
| `a - b` | `sub(a, b)` |
| `a * b` | `mul(a, b)` |
| `a / b` | `div(a, b)` |
| `np.exp(a)` | `exp(a)` |
| `np.log(a)` | `log(a)` |
| `np.sqrt(a)` | `sqrt(a)` |
| `np.abs(a)` | `abs(a)` |

> AFFON uses function calls instead of operator overloading. `affon:compute` is the primary surface.

### Reductions

| NumPy / PyTorch | AFFON |
|-----------------|------|
| `a.sum()` | `sum(a)` |
| `a.mean()` | `mean(a)` |
| `a.sum(axis=1)` | `sum(a, 1)` |
| `a.mean(axis=0, keepdims=True)` | `mean(a, 0, true)` |
| `a.min()` / `a.max()` | `min(a)` / `max(a)` |
| `a.argmax(axis=1)` | `argmax(a, 1)` |
| `loss.item()` | `loss.item()` |

### Linear Algebra

| NumPy / PyTorch | AFFON |
|-----------------|------|
| `a @ b` / `np.matmul(a, b)` | `matmul(a, b)` |
| `np.dot(a, b)` | `dot(a, b)` |

### Shape Operations

| NumPy / PyTorch | AFFON |
|-----------------|------|
| `a.reshape(2, 3)` | `reshape(a, [2, 3])` |
| `a.T` / `a.transpose()` | `transpose(a, 0, 1)` |
| `a[0:3, :]` | `slice(a, range(0, 3), all)` |
| `a.shape` | `a.shape` |

> The compute surface uses explicit selectors: `slice(x, range(...), all)` rather than Python slice syntax.

---

## Autograd

| PyTorch | AFFON |
|---------|------|
| `loss.backward()` | `grad(loss, params)` |
| `param.grad` | `param.grad` |
| `with torch.no_grad():` | `no_grad(() => { ... })` |
| `tensor.requires_grad_(True)` | `parameter(shape)...init()` |

```python
# PyTorch
W = torch.tensor([[0.1, 0.2]], requires_grad=True)
y = (W @ x).sum()
y.backward()
print(W.grad)
```

```typescript
// AFFON
import { parameter, matmul, sum, grad } from 'affon:compute'

const W = parameter([1, 2]).randn()
const params = [W]
const y = sum(matmul(W, x))
grad(y, params)
console.log(W.grad)
```

---

## Neural Networks

### Defining Models

| PyTorch | AFFON |
|---------|------|
| `nn.Linear(in, out)` | `nn.Linear(in, out)` |
| `nn.Sequential(...)` | `nn.Sequential(...)` |
| `nn.Dropout(p)` | `nn.Dropout(p)` |
| `nn.BatchNorm1d(n)` | `nn.BatchNorm(n)` |
| `F.relu(x)` | `relu(x)` |
| `F.sigmoid(x)` | `sigmoid(x)` |
| `F.tanh(x)` | `tanh(x)` |

```python
# PyTorch
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
)
```

```typescript
// AFFON: modules are authored with compute.module(...)
import { module, relu } from 'affon:compute'
import type { Tensor } from 'affon:compute'
import nn from 'affon:nn'

const model = module({
  linear1: nn.Linear(2, 4),
  linear2: nn.Linear(4, 1),
}, (state, x: Tensor): Tensor => state.linear2(relu(state.linear1(x))))
```

Key differences:
- No subclassing or `super().__init__()` — modules are plain callable compute modules
- Models and layers are callable: `model(x)` is the module API
- Activations are free functions from `affon:compute`: `relu(x)` not `F.relu(x)`

### Quick Model (no class needed)

```typescript
// Functional — no class boilerplate
const model = nn.Sequential(
  nn.Linear(2, 4),
  relu,
  nn.Linear(4, 1)
)
```

### Loss Functions

| PyTorch | AFFON |
|---------|------|
| `nn.MSELoss()` | `nn.MSELoss()` |
| `nn.BCELoss()` | `nn.BCELoss()` |
| `nn.CrossEntropyLoss()` | `nn.CrossEntropyLoss()` |

Both return a callable criterion — usage is identical:

```typescript
const criterion = nn.MSELoss()
const loss = criterion(pred, target)
```

### Mode

| PyTorch | AFFON |
|---------|------|
| `model.train()` | `model.mode("train")` |
| `model.eval()` | `model.mode("eval")` |

---

## Optimizers

| PyTorch | AFFON |
|---------|------|
| `optim.SGD(params, lr=0.01)` | `sgd({ lr: 0.01 })` |
| `optim.SGD(params, lr=0.01, momentum=0.9)` | `sgd({ lr: 0.01, momentum: 0.9 })` |
| `optim.Adam(params, lr=0.001)` | `adam({ lr: 0.001 })` |
| `optimizer.zero_grad()` | `clear_grad(params)` |
| `optimizer.step()` | `step(params)` |

---

## Training Loop

```python
# PyTorch
model = MyModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters, lr=0.001)

for epoch in range(100):
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    print(f"epoch {epoch}, loss: {loss.item():.4f}")
```

```typescript
// AFFON
import compute, { adam, clear_grad, grad, module, relu } from 'affon:compute'
import nn from 'affon:nn'

const model = module({
  linear1: nn.Linear(2, 4),
  linear2: nn.Linear(4, 1),
}, (state, x) => state.linear2(relu(state.linear1(x))))
const criterion = nn.MSELoss()
const params = model.parameters
const step = adam({ lr: 0.001 })

for (let epoch = 0; epoch < 100; epoch++) {
  clear_grad(params)
  const pred = model(x)
  const loss = criterion(pred, target)
  grad(loss, params)
  step(params)
  console.log(`epoch ${epoch}, loss: ${loss.item().toFixed(4)}`)
}
```

---

## Dataset Loading

### Loading CSV Data

| Python (pandas + torch) | AFFON |
|--------------------------|------|
| `pd.read_csv('data.csv')` | `read('data.csv')` |
| `LabelEncoder().fit_transform(col)` | `.encode('col', 'label')` |
| `pd.get_dummies(col)` | `.encode('col', 'onehot')` |
| `df[['col1', 'col2']].values` | `.input('col1', 'col2')` |
| `df['target'].values` | `.target('target')` |
| `torch.utils.data.DataLoader(ds, batch_size=32)` | `.loader({ batchSize: 32 })` |

```python
# Python
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv('iris.csv')
X = torch.tensor(df[['sepal_length', 'sepal_width']].values, dtype=torch.float32)
y = torch.tensor(LabelEncoder().fit_transform(df['species']), dtype=torch.float32)
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

for X_batch, y_batch in loader:
    ...
```

```typescript
// AFFON
import { read, DataLoader } from 'affon:dataset'

const ds = read('iris.csv')
  .select('sepal_length', 'sepal_width', 'species')
  .encode('species', 'label')
  .shuffle()
  .input('sepal_length', 'sepal_width')
  .target('species')

for (const [X, y] of ds.loader({ batchSize: 32 })) {
  // ...
}
```

Key differences:
- `affon:dataset` is a dataset pipeline surface, not a general DataFrame API
- the current tabular CSV workflow also has an explicit home at `dataset.tabular.*`
- the first text slice lives under `dataset.text.*` for line-oriented corpora and batching
- tokenizer constructors live in `@affon/tokenizers`
- `createLookupTokenizer(...)` is the first tokenizer slice; richer tokenizer families can layer on later
- `createHFTokenizerFromJSON(...)` and `createHFTokenizerFromFile(...)` are family adapters behind the shared `TextTokenizer` interface, with support for common Hugging Face `WordLevel`, `BPE`, and `WordPiece` tokenizer JSON layouts
- `dataset.text.encode(...)` and `encodePair(...)` accept generic truncation options like `maxLength` plus `longest_first` / `only_first` / `only_second`
- pair encoders accept a small `pairTemplate` vocabulary (`bos`, `left`, `separator`, `right`, `eos`) so layouts like `[CLS] A [SEP] B [SEP]` can stay inside `dataset.text`
- for common pair-classification layouts, `pairTemplatePreset: 'bert'` expands to `bos,left,separator,right,separator,eos` and `joined` expands to `left,separator,right`
- `TextTokenizer` can now expose optional `specialTokens` / `specialTokenIds` metadata, so presets like `bert` can use tokenizer-native separator tokens without requiring `separatorText` every time
- supervised text now flows through `dataset.text.readDelimited(...)` plus field transforms
- `dataset.text.labelEncoder(...)` can turn supervised text labels into integer targets for tensor batches
- Encoding, input selection, and target selection are chained on the pipeline
- DataLoader accepts a Dataset directly, calls `toTensors()` internally
- On the tabular side, `.input(...)` is the shared cross-domain alias of `.features(...)`, and `.tensorLoader(...)` is the alias of `.loader(...)`
- `toTensor()` and `toTensors()` export `f32` by default to match `affon:compute` and `affon:nn`

### Text Datasets

```typescript
import dataset from 'affon:dataset'
import { createLookupTokenizer } from '@affon/tokenizers'

const tok = createLookupTokenizer(
  {
    '<pad>': 0,
    '<bos>': 1,
    '<eos>': 2,
    '<sep>': 3,
    '<unk>': 4,
    good: 5,
    bad: 6,
    neutral: 7,
  },
  {
    specialTokens: {
      pad: '<pad>',
      bos: '<bos>',
      eos: '<eos>',
      sep: '<sep>',
      unk: '<unk>',
    },
  },
)
```

Single-label classification:

```typescript
const labelEncoder = dataset.text.labelEncoder({ pos: 0, neg: 1, neu: 2 })

const classified = dataset.text
  .readDelimited('labeled-lines.tsv', {
    delimiter: '\t',
    columns: ['text', 'label'],
  })
  .encode('text', tok, { into: 'inputIds' })
  .encodeLabels('label', labelEncoder, { into: 'labels' })
  .input('inputIds')
  .target('labels')

for (const batch of classified.tensorLoader({ batchSize: 32, padId: 0 })) {
  // batch.inputIds: i64 tensor
  // batch.attentionMask: i64 tensor
  // batch.labels: i64 tensor
}
```

Multi-label classification:

```typescript
const tags = dataset.text.labelEncoder({ pos: 0, neg: 1, neu: 2, featured: 3 })

const multilabel = dataset.text
  .readDelimited('multilabel-lines.tsv', {
    delimiter: '\t',
    columns: ['text', 'labels'],
  })
  .splitLabels('labels')
  .encode('text', tok, { into: 'inputIds' })
  .toMultiHot('labels', tags, { into: 'labels' })
  .input('inputIds')
  .target('labels')

for (const batch of multilabel.tensorLoader({ batchSize: 32, padId: 0 })) {
  // batch.labels is a multi-hot i64 tensor of shape [batch, numClasses]
}
```

Scalar regression:

```typescript
const scored = dataset.text
  .readDelimited('scored-lines.tsv', {
    delimiter: '\t',
    columns: ['text', 'score'],
  })
  .encode('text', tok, { into: 'inputIds' })
  .cast('score', 'f32', { into: 'labels' })
  .input('inputIds')
  .target('labels')

for (const batch of scored.tensorLoader({ batchSize: 32, padId: 0 })) {
  // batch.labels: f32 tensor
}
```

Pair classification with tokenizer-native special tokens:

```typescript
const paired = dataset.text
  .readDelimited('text-pairs.tsv', {
    delimiter: '\t',
    columns: ['left', 'right'],
  })
  .encodePair({ left: 'left', right: 'right' }, tok, {
    into: 'inputIds',
    tokenTypesInto: 'tokenTypeIds',
    withTokenTypes: true,
    pairTemplatePreset: 'bert',
    maxLength: 128,
    truncation: 'longest_first',
  })
  .input('inputIds', 'tokenTypeIds')

for (const batch of paired.tensorLoader({ batchSize: 16, padId: 0 })) {
  // batch.inputIds
  // batch.attentionMask
  // batch.tokenTypeIds
}
```

---

## Metrics

| scikit-learn | AFFON |
|--------------|------|
| `accuracy_score(y_true, y_pred)` | `accuracy(pred, target)` |
| `precision_score(y_true, y_pred)` | `precision(pred, target)` |
| `recall_score(y_true, y_pred)` | `recall(pred, target)` |
| `f1_score(y_true, y_pred)` | `f1(pred, target)` |
| `mean_squared_error(y_true, y_pred)` | `mse(pred, target)` |
| `mean_absolute_error(y_true, y_pred)` | `mae(pred, target)` |
| `r2_score(y_true, y_pred)` | `r2(pred, target)` |

```python
# scikit-learn
from sklearn.metrics import accuracy_score, f1_score
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
```

```typescript
import { accuracy, f1 } from 'affon:compute'
const acc = accuracy(pred, target)
const f1Score = f1(pred, target)
```

Key differences:
- metrics live directly in `affon:compute`
- AFFON metrics take raw tensors (probabilities for classification, not thresholded labels)
- Binary classification auto-thresholds at 0.5 (configurable via `{ threshold: 0.7 }`)
- Multiclass accuracy auto-detects from 2D pred (argmax internally)
- Argument order: `pred` first, `target` second (same as PyTorch loss convention, opposite of scikit-learn)

---

## Checkpoints

Use `affon:checkpoint` for module-level persistence:

```typescript
import checkpoint from 'affon:checkpoint'
import nn from 'affon:nn'

const model = nn.Sequential(
  nn.Linear(3, 4),
  nn.Linear(4, 1),
)

checkpoint.save(model.state(), 'model.safetensors')

const restored = nn.Sequential(
  nn.Linear(3, 4),
  nn.Linear(4, 1),
)

checkpoint.restore(restored, 'model.safetensors')
```

Key differences:
- `affon:checkpoint` owns the public save/load namespace
- modules still expose `model.save(path)` and `model.load(path)` as convenience methods
- `checkpoint.load(path)` returns the raw named tensor state dictionary

---

## Visualisation

| matplotlib | AFFON |
|------------|------|
| `plt.plot(x, y)` | `plot(x, y)` |
| `plt.scatter(x, y)` | `scatter(x, y)` |
| `plt.bar(x, y)` | `bar(x, y)` |
| `plt.title('...')` | `title('...')` |
| `plt.xlabel('...')` | `xlabel('...')` |
| `plt.ylabel('...')` | `ylabel('...')` |
| `plt.savefig('file.png')` | `savefig('file.svg')` |
| `plt.show()` | `show()` |

```typescript
import { plot, title, xlabel, ylabel, savefig } from 'affon:plot'

plot(losses, { label: 'training loss' })
title('Training Loss')
xlabel('Epoch')
ylabel('Loss')
savefig('loss.svg')
```

> Output is SVG (not PNG/PDF). In Jupyter, plots render inline automatically.

---

## Jupyter

| Python | AFFON |
|--------|------|
| `pip install jupyter` | `brew install zmq` (one-time) |
| Kernel auto-installed with Python | `affon jupyter install` |
| `display(obj)` | Last expression auto-displays |
| `%matplotlib inline` | Plots are inline by default |

```bash
affon jupyter install    # one-time setup
```

In VS Code, install [Affon for VS Code](https://github.com/neucine/affon-vscode/releases/tag/v0.0.1), open a normal `.ipynb` notebook, and choose `Affon` in the kernel picker.

In JupyterLab, select `affon` as the kernel.

Rich display works automatically:
- Compute tensors render through their `repr()` display hook
- Plots render as inline SVG
- `console.log()` outputs to cell stdout

---

## FFI (calling C libraries)

| Python (ctypes) | AFFON |
|------------------|------|
| `cdll.LoadLibrary('libm.so')` | `dlopen('libm', { ... })` |
| `lib.sqrt.argtypes = [c_double]` | Declared inline: `{ args: ['f64'], returns: 'f64' }` |
| `lib.sqrt.restype = c_double` | (same declaration) |
| `lib.sqrt(16.0)` | `lib.sqrt(16.0)` |

```typescript
import { dlopen } from 'std:ffi'

const lib = dlopen('libm', {
  sqrt: { args: ['f64'], returns: 'f64' },
  pow:  { args: ['f64', 'f64'], returns: 'f64' },
})

lib.sqrt(16)   // 4
lib.pow(2, 10) // 1024
```

> Type-safe at compile time via TypeScript generics — no runtime `argtypes` setup.

---

## Error Handling

| Python / PyTorch | AFFON |
|------------------|------|
| `RuntimeError`, `ValueError` | `AffonError` (unified class) |
| `err.args[0]` (message) | `err.message` |
| No standardized error codes | `err.code` (structured union) |
| `traceback.print_exc()` | Native Zig stack traces in debug build |

```typescript
try {
  model(x)
} catch (err) {
  if (err instanceof AffonError) {
    console.log(err.code) // e.g. "shape_mismatch"
  }
}
```

> See [Error Handling](../core/errors.md) for more details.

---

## Type Safety

Something Python doesn't have — AFFON provides compile-time shape checking:

```typescript
import { tensor, matmul } from 'affon:compute'

const a = tensor([[1, 2], [3, 4]])       // Tensor<[2, 2]>
const b = tensor([[1, 2, 3], [4, 5, 6]]) // Tensor<[2, 3]>
const c = matmul(a, b)                  // Tensor<[2, 3]> — inferred

// TypeScript catches shape mismatches before you run anything
```

---

## Quick Reference: Naming Conventions

| Python convention | AFFON convention |
|-------------------|-----------------|
| `snake_case` methods | `snake_case` methods |
| `zero_grad()` | `clear_grad(params)` |
| `batch_first` | `batch_first` |
| `hidden_state()` | `hidden_state()` |
| `requires_grad` kwarg | `parameter(shape)...init()` |
| `no_grad()` context manager | `no_grad(() => { ... })` callback |
| `model(x)` (via `__call__`) | `model(x)` |
| `print(tensor)` | `console.log(tensor)` |
| `f"loss: {val:.4f}"` | `` `loss: ${val.toFixed(4)}` `` |
| `for i in range(n):` | `for (let i = 0; i < n; i++)` |
| `if __name__ == '__main__':` | Top-level code runs directly |

Notes:
- AFFON user-facing functions, methods, and option fields generally prefer `snake_case`
- `nn` class-like constructors stay PascalCase, such as `nn.Linear`, `nn.Sequential`, `nn.SimpleRNN`, `nn.RNN`, and `nn.LSTM`
