# Compute Concepts

`affon:compute` is the intended math-first API surface for Affon.

There is no separate public `affon:ndarray` or `affon:tensor` module. Public
numeric data lives under `affon:compute`, and `tensor(...)` is the concrete
constructor on that surface.

It exposes computation in terms of:

- compute tensors
- parameters
- programs
- modules
- gradients
- graph compilation

`affon:compute` is Affon's public numeric substrate.

## Function names

`affon:compute` currently uses a small mixed naming surface.

- Most math/data constructors are short lowercase names:
  - `tensor`, `zeros`, `ones`, `randn`, `matmul`, `softmax`
- Some runtime/control helpers still follow JavaScript method naming:
  - `toString`
- Some promoted helpers use snake_case:
  - `clip_grad_norm`
  - `no_grad`
  - `finite_summary`
  - `gt_scalar`
  - `index_select`
  - `masked_fill`

Use the exported names exactly as declared today. They are part of the public contract even when the casing differs.

## TypeScript typing

`compute` is also the primary typed surface.

Its core declarations are generic over shape and dtype:

```ts
import type { Parameter, Tensor } from 'affon:compute'

type TokenIds = Tensor<[number, number], 'i64'>
type Hidden = Tensor<[number, number, 256], 'f32'>
type Weights = Parameter<[256, 512], 'f32'>
```

The goal is strong editor and refactor support around the main contracts without forcing every compute operation into heavy type-level shape algebra.

## Core ideas

- `tensor(...)` creates ordinary quantities.
- `parameter(shape, opts?)` creates trainable module state, then initializes fluently.
- `grad(loss, params)` differentiates a scalar loss with respect to parameters and attaches gradients onto them.
- `compile(f)` compiles a callable compute program or modeful module without changing its mathematical interface.

## Execution graph pipeline

`compile(...)` builds an execution graph and then derives optimized execution
plans from that graph.

The current internal native boundary still uses a serialized graph-build schema
under:

- `src/js/compute/captured/schema.zig`
- `src/js/compute/captured/lowering.zig`

Autograd runtime internals are also moving toward one value-centric model:

- `Value` is the runtime tensor object
- tracked/trainable behavior is internal value state
- `compute.autograd.*` is the intended non-internal surface
- inside autograd core, `State` now owns the common tracked-output lifecycle
  for ops, reducing peer-like autograd sidecar handling in forward/runtime code

This is an implementation boundary, not a second public graph API or a durable
parallel executor. The public contract is still `compile(...)`, compiled
modules/functions, the execution graph surfaced by `graph(...)`, and
native-graph planning/execution surfaced through the normal compute runtime.

## Compute Tensors

Use `tensor(...)` for concrete numeric data:

```ts
import { tensor, add, matmul } from 'affon:compute'

const x = tensor([[1, 2], [3, 4]])
const y = tensor([[5], [6]])
const z = matmul(x, y)
```

Common constructors:

- `empty(shape, opts?)`
  Creates an untracked tensor buffer. Use `parameter(...)` for trainable state.
- `zeros(shape, opts?)`
- `ones(shape, opts?)`
- `full(shape, fill, opts?)`
- `rand(shape, opts?)`
- `randn(shape, opts?)`
- `arange(...)`
- `linspace(...)`
- `seed(value)`

Intrinsic metadata are properties:

- `x.shape`
- `x.axes`
- `x.rank`
- `x.dtype`
- `x.device`

Axis names are optional semantic labels over positional tensor axes:

```ts
import { axes, tensor, relu, permute, sum } from 'affon:compute'

const x = tensor([[1, 2], [3, 4]], {
  axes: [axes.batch, axes.feature],
})

relu(x).axes              // ["batch", "feature"]
permute(x, [1, 0]).axes   // ["feature", "batch"]
sum(x, 0).axes            // ["feature"]
```

Affon executes tensor ops by numeric axes, but the public tensor object can carry
axis names so model code can retain the mathematical meaning of each dimension.
Conventional names such as `axes.batch`, `axes.token`, and `axes.feature` are
provided for reuse. They are not a closed vocabulary; user-defined names are
valid.

Ops infer output axis names when the mapping is unambiguous:

- shape-preserving ops preserve names
- permutation-style ops reorder names
- single-axis reductions remove the reduced name, unless `keep` is true
- reshaping preserves names only when the shape is unchanged

When Affon cannot infer axis meaning, the derived tensor has no axis names.
Calling a compute module with a rankful tensor that has no axis names prints a
warning. The warning only checks whether names are defined; it does not require
particular names such as `batch` or `feature`.

Axis names are included in graph/report exports as value-spec metadata, so
`exportReport(..., { format: 'json' })` and annotated text reports can be used
to inspect whether semantic axes survived compilation and lowering.

Values also expose representation helpers:

- `x.toString()`
- `x.repr(opts?)`

Example:

```ts
const x = tensor([[1, 2], [3, 4]])

console.log(x.toString())
console.log(x.repr())
console.log(x.repr({ mode: 'html' }))
```

Changes to representation stay explicit:

```ts
import { cast, move } from 'affon:compute'

const x64 = cast(z, 'f64')
const xMetal = move(x64, 'metal')
```

For index-heavy compute, `tensor(...)` and `empty(...)` also support `dtype: 'i64'`.

## Parameters

Parameters are not just ordinary values. They are trainable persistent state of a parameterized function.

Create them from shape, then initialize:

```ts
import { parameter } from 'affon:compute'

const w = parameter([2, 4]).kaiming_uniform()
const b = parameter([4]).zeros()
```

Parameters can be used directly in compute expressions:

```ts
import { add, matmul } from 'affon:compute'

const y = add(matmul(x, w), b)
```

## Modules

Modules are callable parameterized functions:

```ts
import { module, parameter, add, matmul } from 'affon:compute'

const linear = module(
  {
    w: parameter([2, 4]).kaiming_uniform(),
    b: parameter([4]).zeros(),
  },
  ({ w, b }, x) => add(matmul(x, w), b),
)

const y = linear(tensor([[1, 2]]))
```

The first argument to `module(...)` is the full persistent module state tree. The root itself may be an object, array, tensor, parameter, or plain JS data. It may contain:

- parameters
- values
- plain JS objects
- arrays
- scalars
- nested modules

Attached utilities on a callable module:

- `f.parameters`
- `f.state()`
- `f.restore(state)`
- `f.mode()`
- `f.mode("train" | "eval")`
- `f.metadata(path)`

`parameters` returns only trainable parameters, deduplicated by identity.

`state()` returns the compute-native persistent state tree needed to restore future module behavior correctly.

`metadata(path)` attaches a logical module path that native graph export can
carry through as `module_path` on captured nodes. This is intended for
observability and visualization grouping, not execution planning.
When a module owns submodules in its state tree, derived child path segments use
the actual state keys. Prefer `snake_case` state keys for new modules;
explicitly authored metadata paths are preserved.

Example:

```ts
const block = module(
  { w: parameter([4, 4]).kaiming_uniform() },
  ({ w }, x) => matmul(x, w),
).metadata('decoder.blocks.0')
```

For offline graph tooling, export the canonical report as raw JSON:

```ts
import compute, { compile, module, parameter, matmul, tensor } from 'affon:compute'

const block = module(
  { w: parameter([4, 4]).kaiming_uniform() },
  ({ w }, x) => matmul(x, w),
).metadata('decoder.blocks.0')

const compiled = compile(block)
const x = tensor([[1, 2, 3, 4]])

compiled(x)

const path = compute.exportReportFile(compiled, x, {
  format: 'json',
  boundary: 'step',
  phase: 'forward',
  runId: 'train-1',
  graphId: 'decoder',
  step: 32,
})
```

`format: 'json'` writes the raw canonical report for offline
rendering or analysis in external tools. Without `format`, report export keeps
returning the native text rendering.

For graph visualizers, the forward or derived execution graph now includes an
explicit `edges` array, so offline tools can render a pure node-edge graph
without reconstructing connectivity from value records alone. Each edge carries:

- `edge_id`
- `value_id`
- `producer_node_id`
- `consumer_node_id`

If nodes carry authored `module_path` metadata, the same export also includes a
collapsed module graph:

- `module_nodes`: unique module-path groups with their member `node_ids`
- `module_edges`: aggregated cross-module connectivity with `op_edge_count`

That lets offline tools choose between:

- exact op graph: `nodes` + `edges`
- collapsed module graph: `module_nodes` + `module_edges`

## Gradients

Gradients are attached to parameters.

Typical loop:

```ts
import compute, { clear_grad, clip_grad_norm, grad, module, parameter, tensor, mean, square, sub, sgd } from 'affon:compute'

const f = module(
  {
    w: parameter([1, 1]).randn(),
    b: parameter([1]).zeros(),
  },
  ({ w, b }, x) => add(matmul(x, w), b),
)

const params = f.parameters
const step = sgd({ lr: 0.01 })

const x = tensor([[1], [2], [3]])
const y = tensor([[2], [4], [6]])

for (let epoch = 0; epoch < 100; epoch++) {
  clear_grad(params)
  const pred = f(x)
  const loss = mean(square(sub(pred, y)))
  grad(loss, params)
  clip_grad_norm(params, 1.0)
  step(params)
}
```

`clear_grad(params)` clears attached gradients.

`grad(loss, params)` computes parameter gradients and attaches them to `param.grad`.

`clip_grad_norm(params, max_norm)` rescales gradients when their global norm exceeds `max_norm`.

Optimizers are step constructors:

```ts
const step = sgd({ lr: 0.01 })
step(params)
```

## Selection and shape

The selection language stays small:

```ts
import { at, slice, all, range } from 'affon:compute'

const x = tensor([[1, 2, 3], [4, 5, 6]])

at(x, 1, 2)
slice(x, all, range(0, 2))
```

Core shape operations:

- `reshape(x, shape)`
- `transpose(x, dim1, dim2)`
- `permute(x, dims)`
- `squeeze(x, dim?)`
- `unsqueeze(x, dim?)`
- `cat(values, dim?)`
- `stack(values, dim?)`

Wave-3 promoted indexed ops now also include:

- `one_hot(indices, numClasses)`
- `gather(input, dim, index)`
- `index_select(input, dim, index)`
- `topk(input, k, dim?)`

`topk(...)` returns a pair:

- `result.values`
- `result.indices`

Runtime/control helpers that also belong on the compute surface:

- `contiguous(x)`
- `no_grad(() => ...)`
- `copy(target, source)`
- `finite_summary(x)`
- `finite_abs_max(x)`
- `gt_scalar(x, threshold)`

`finite_summary(x)` returns a small diagnostic object:

- `ok`
- `first_bad_flat_index`
- `first_bad_value`

`finite_abs_max(x)` returns:

- a number when the tensor is fully finite
- `null` if any `NaN` or `Infinity` is present
- `null` if the runtime cannot safely complete the reduction

Reductions use `dim`:

- `sum(x, dim?, keep?)`
- `mean(x, dim?, keep?)`
- `max(x, dim?, keep?)`
- `min(x, dim?, keep?)`
- `variance(x, dim?, keep?)`
- `std(x, dim?, keep?)`
- `argmin(x, dim?, keep?)`
- `argmax(x, dim?, keep?)`

Wave-1 promoted math ops now also include:

- `abs(x)`
- `sign(x)`
- `clamp(x, min, max)`
- `dot(a, b)`

Wave-2 promoted compute ops now also include:

- `where(cond, a, b)`
- `masked_fill(input, mask, value)`
- `softmax(x, dim)`

Broadcasting is part of the compute substrate, not a legacy standalone tensor convenience.

## Compile

Graph execution is a first-class part of the compute layer:

```ts
import { compile } from 'affon:compute'

const fastLinear = compile(linear)
const y = fastLinear(x)
```

`compile(f)` accepts any callable compute function, including modules.

When `f` is a module, the compiled callable stays bound to the same module state and preserves the module utilities.

The intended inspection stack is:

- `compiled.summary(...args?)` for the quick human/debug view
- `compiled.graph(...args)` for execution-graph structure and lowering analysis
- `compiled.exportReport(...args, { format: 'json' })` for the full report with `graph`, `graph_plan`, and `derived_graph` sections

Lower-level helpers like `compiled.capturedProgram(...)` and
`compiled.captureSummary(...)` remain available for schema-oriented tooling,
but they are not the primary inspection surface.

Graph summaries may include `nodes[].outputShape` for quick inspection. Those
shapes are labeled by `summary.staticMetadata` as provisional TS capture
metadata. They are not semantic proof of operator validity; native lowering and
core inference remain the execution authority.

Compiled execution graphs run through native lowering as their execution
authority. If an execution graph shape is not yet lowerable into native
`Graph`, execution fails with a native lowering error instead of
silently falling back to a separate TS graph runtime. Use `compiled.summary()`
or `compiled.graph()?.loweringAnalysis?.()` to inspect the current lowering
decision surface when working on compiler coverage. If graph build itself
cannot be completed, `compiled.summary().graphCaptureError` reports that build
failure while execution stays on the eager-forward path. Newer inspection code
should prefer `compiled.summary().graphCaptureFailure`, which keeps the reason
and classifies the failure boundary. Today graph-build failures are classified
as `capture_adapter` unless they are clearly syntax-level; native lowerability
gaps remain visible through `graphLoweringAnalysis`. Lowering failures include
a `failure.category` so tooling can distinguish unsupported captured node kinds,
invalid adapter metadata, invalid execution metadata, and semantic rejections
from native infer/runtime lowering. For covered ops, compiled graph forwards
also preserve autograd links during training, so `grad(loss, params)` continues
to work on losses produced by native-graph compiled modules.

When you have concrete inputs and want detailed execution structure, export the
full JSON report and inspect its `graph_plan` section. If you only need quick
plan facts, `compiled.summary(...args)` includes
`planAvailable`, `planStepCount`, `planRegionCount`, and `planOutputCount`.
It also reports specialization behavior through `specializationActive`,
`activeTensorSignature`, `specializationCaptureCount`,
`specializationReuseCount`, `eagerFallbackCount`, `lastExecutionMode`,
structured `lastFallbackKind`, and human-readable `lastFallbackReason`.
`summary().nativeState` is a companion inspection projection from the native
compiled executable. It is useful when comparing TS specialization decisions
with native execution state, but it is not a separate compute model. Its
`planOutcome` field reports whether the last compiled outcome was native graph
execution or an eager-forward fallback; the eager fallback itself is still
performed by the TS wrapper in this transitional boundary.

Correctness note: compiled callables treat a successfully captured tensor
signature as exact. If a later call arrives with a different tensor shape
signature and there is no proven specialization for that shape, execution
stays on the eager-forward path instead of reusing an incompatible native
graph. In that case `compiled.graph(...args)` may return `null` for the new
shape even if an earlier shape compiled successfully. Likewise, if a compiled
module changes execution-relevant state such as train/eval mode, summary
fallback reporting distinguishes that with `lastFallbackReason === "program state changed"`.

Public training contract note: ordinary tensors stay grad-transparent at the
JS surface. Persistent gradient access is exposed through `Parameter.grad`, and
training orchestration is expected to flow through `grad(loss, params)` rather
than tensor-method backward calls. Hidden tensor-object autograd hooks are no
longer part of the runtime object contract; the remaining JS/native training
bridge now goes through explicit internal runtime helpers instead.

Internal runtime note: the compute core is now converging on one value-centric
model. `Value` is the runtime tensor object, and autograd state is treated as
internal value state (`plain`, `tracked`, or `trainable`) rather than as a
second public/runtime tensor kind. Internal autograd `State` now owns
tape/backward/provenance details as an attached sidecar, and runtime code is
pushed behind value-centric helpers instead of treating that sidecar as a peer
tensor concept at API and execution boundaries. The intended non-internal entry
surface is `compute.autograd.*`; `state.zig` should be treated as internal
substrate.

Current internal autograd boundary:
- `state.zig` owns live autograd sidecar state attached to `Value`
- `tape.zig` owns provenance-node creation and lifetime helpers
- `derived_graph.zig` owns backward IR construction and execution
- `compute.autograd` is the supported value-centric facade; it should not grow
  into a generic barrel for internal provenance/tooling layers

Current implementation note: compiled native-graph training still crosses an
internal seam between plain `Value` execution and value-attached autograd
state. Covered compiled forwards preserve grads correctly, but the grad-capable
graph executor is still an adapter layer rather than the final tensor/runtime
model. Eager and graph remain distinct execution modes internally today, even
though they are converging on the same value-centric compute core.

Regression note: the core autograd suite now primarily exercises tracked
computation through value-centric entrypoints (`autograd.makeTrainableValue`,
`autograd.backwardValue`, `autograd.gradForValue`, and value-level invoke
helpers) rather than using sidecar-level construction as the default test-time
tensor shape.
