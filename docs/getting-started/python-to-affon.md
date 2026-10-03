# Python to Affon

Affon is not a line-for-line NumPy or PyTorch port. Its canonical API makes the
reusable computation explicit as a `Program`, then runs that Program in a
`Session`.

## The core translation

| Python/PyTorch idea | Affon |
| --- | --- |
| tensor shape and dtype declaration | `Tensor.f32(shape)`, `Tensor.i64(shape)` |
| model input | `p.argument(name, spec)` |
| trainable value | `p.parameter(name, spec, options)` |
| layer declaration | `p.nn.linear(...)`, `p.nn.embedding(...)` |
| tensor function | `affon:ops` |
| reusable model | `program(name, p => output)` |
| device context | `new Session({ device })` |
| model parameters and state | `session.initialize(program)` |
| compiled call | `session.compile(program).run(arguments, state)` |
| backward pass | `gradient(program, names)` |
| optimizer | `optimize(modelProgram, lossProgram, adamw(options))` |

`Tensor` intentionally names two related public concepts. In type position it
is an evaluated tensor interface; in value position it is the namespace-like
factory for `TensorSpec` values. No import alias is needed.

## From a tensor expression

PyTorch:

```py
x = torch.tensor([[1., 2.], [3., 4.]])
y = x * x
```

Affon, evaluated immediately:

```ts
import { Session } from 'affon:compute'
import { mul } from 'affon:ops'

const session = new Session({ device: 'cpu' })
const x = session.tensor([[1, 2], [3, 4]])
const y = mul(x, x)

console.log(y.to_array())

y.dispose()
x.dispose()
session.dispose()
```

The operands select the Session, so operations do not need a separate session
argument. All evaluated operands must belong to the same Session. Immediate
operations compute values but do not record gradients.

## From an `nn.Module`

PyTorch:

```py
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.GELU(),
    nn.Linear(32, 2),
)
```

Affon:

```ts
import { Tensor, program } from 'affon:compute'
import { gelu } from 'affon:ops'

const model = program('classifier', p => {
  const x = p.argument('x', Tensor.f32([1, 4]))
  const hidden = gelu(p.nn.linear(x, {
    name: 'hidden',
    out_features: 32,
  }))
  return p.nn.linear(hidden, {
    name: 'output',
    out_features: 2,
  })
})
```

`p.nn` is builder-bound because these calls declare named parameters in the
active Program. General operations live in `affon:ops`; there are no canonical
tensor methods such as `x.matmul(y)` and no global operation exports from
`affon:compute`.

## Training

```ts
import { Session, losses, optimize } from 'affon:compute'
import { adam } from 'affon:optim'

const train = optimize(model, losses.cross_entropy(), adam({ learning_rate: 1e-3 }))
const session = new Session({ device: 'cpu' })
const state = session.initialize(train, { seed: 7 })
const step = session.compile(train)

const x = session.tensor(features)
const labels = session.tensor(targets, { dtype: 'i64' })
const currentLoss = step.run({ x, labels }, state)
```

Unlike PyTorch's mutable tape, differentiation and optimization are Program
transforms. This makes argument roles, parameters, persistent state, and update
transitions inspectable before execution.

## Names and shapes

- Shapes are arrays such as `[batch, features]`.
- Public option names use `snake_case`, for example `out_features` and
  `learning_rate`.
- `Executable.run(...)` accepts a named argument record and validates names,
  shapes, dtypes, disposal state, and Session ownership before native execution.
- Classification labels for `cross_entropy` from `affon:ops` are `i64` and match the logits
  shape with the final class axis removed.

Programs, evaluated tensors, and operations are the complete public compute
surface. Use `gradient(...)` or `optimize(...)` for differentiation and
training instead of mutable tensor gradients or callable module objects.
