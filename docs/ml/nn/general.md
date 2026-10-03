# Program Components

Neural-network declarations are ordinary callable factories:

```ts
const projection = linear({
  out_features: 128,
})
const y = projection({ x }, 'projection')
```

The first call fixes hyperparameters. The returned function binds formal tensors
and expands named parameters into the Program that owns those tensors.

## Reusable pieces

Use ordinary functions for reusable architecture fragments:

```ts
import type { FormalTensor } from 'affon:compute'
import { linear } from 'affon:nn'
import { gelu } from 'affon:ops'

function feedForward(width: number) {
  const up = linear({ out_features: width })
  return ({ x }: { x: FormalTensor }, name = 'feed_forward') => {
    const hidden = up({ x }, `${name}.up`)
    return linear({ out_features: x.spec.shape.at(-1)! })({ x: gelu(hidden) }, `${name}.down`)
  }
}
```

Use a child Program when the piece should have its own inspectable identity and
be composed into multiple parents. Call it with named bindings and an optional
instance name: `child({ value }, 'projection')`. The instance name namespaces the
child's parameters, state, and constants. It also automatically extends the
inspected composition path of every copied child node. Nested uses therefore
remain groupable without adding group nodes or manual path annotations.

A callable may also read `value.spec`, author or select a cached
shape-specialized child Program, and compose it during the same call. This is
useful for complete models whose batch or sequence dimensions come from their
bindings: the model remains `model({ input }, 'optional.name')`, while the outer
`program(...)` remains the explicit execution and inspection boundary.

## State roles

- `p.argument(...)` declares values supplied to every run.
- `p.parameter(...)` declares trainable state.
- `p.state(...)` declares persistent non-parameter state.
- `p.constant(...)` declares immutable Program data.

`Session.initialize(...)` materializes parameters and state into an
`ExecutionState`. `Executable.run(...)` accepts only named argument tensors;
it obtains parameters and state from that object.

There is no separate callable module tree or train/eval mode object. Programs
make parameters and state explicit and inspectable.
