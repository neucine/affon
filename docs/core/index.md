# Core Numerics

Affon's public compute model separates declarations from evaluated values:

```text
Program -> Session.compile -> Executable.run -> evaluated Tensor
```

- `affon:compute` declares Programs, tensor specifications, Sessions,
  differentiation, and optimization transforms.
- `affon:ops` is the operation vocabulary shared by formal tensors inside a
  Program and evaluated tensors owned by a Session.
- `affon:optim` creates immutable optimizer descriptions consumed by
  `optimize(...)`.

Start with [Compute Programs](./compute.md), then consult the
[Compute Kernel Matrix](./kernel-matrix.md) for backend coverage and
[Error Handling](./errors.md) for failure and cleanup patterns.

## Smallest useful example

```ts
import { Session, Tensor, program } from 'affon:compute'
import { mul } from 'affon:ops'

const square = program('square', p => {
  const x = p.argument('x', Tensor.f32([3]))
  return mul(x, x)
})

const session = new Session({ device: 'cpu' })
const executable = session.compile(square)
const x = session.tensor([1, 2, 3])
const y = executable.run({ x })

console.log(y.to_array()) // [1, 4, 9]

```

For a one-off calculation, call the same `affon:ops` functions with evaluated
tensors. Immediate evaluation is computation-only: it does not create a
gradient tape. Use a Program when the computation needs differentiation,
optimization, inspection, composition, or repeated execution.

For short scripts, `Tensor.from(...)`, `Tensor.zeros(...)`,
`Tensor.ones(...)`, and `Tensor.full(...)` create evaluated tensors in a hidden
lazy default Session. Explicit disposal is optional; use it only when a workload
needs deterministic release.

The Program API is the only public compute and neural-network authoring model.
