# Program Components

Neural-network declarations are methods of the active `ProgramBuilder`:

```ts
const y = p.nn.linear(x, {
  name: 'projection',
  out_features: 128,
})
```

This placement is deliberate. A layer declaration creates named parameters in
the Program being authored; it is not an eager, stateful callable object.

## Reusable pieces

Use ordinary functions for reusable architecture fragments:

```ts
import type { FormalTensor, ProgramBuilder } from 'affon:compute'
import { gelu } from 'affon:ops'

function feedForward(p: ProgramBuilder, x: FormalTensor, width: number) {
  const hidden = p.nn.linear(x, { name: 'up', out_features: width })
  return p.nn.linear(gelu(hidden), {
    name: 'down',
    out_features: x.spec.shape.at(-1)!,
  })
}
```

Use a child Program when the piece should have its own inspectable identity and
be composed into multiple parents. A Program can be called positionally while
another Program is being authored, or bound explicitly with
`p.use(child, { as, ...arguments })`. The `as` alias namespaces the child's
parameters, state, and constants.

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
