# Interactive Program Inspection

Status: first vertical slice implemented

## Direction

The inspector should be a generic consumer of `ProgramInspection`, not a model
viewer with decoder, transformer, or neural-network conventions embedded in it.
`Program.inspect()` remains the semantic boundary. Rendering, layout, search,
selection, and local UI state belong outside `affon:compute`.

```text
Program.inspect()
      |
      v
inspection adapter/index  ---->  JSON import/export
      |
      v
generic view model
      |
      +---- hierarchy/tree
      +---- dependency graph
      +---- arguments/parameters/state/constants
      +---- transforms and provenance
```

The first implementation should work with the current inspection object. Schema
extensions should improve fidelity without making the viewer depend on live
Program objects or a Session.

## Goals

- Show the invocation contract: arguments, outputs, dtypes, shapes, and axes.
- Show model structure from composition paths, with scopes collapsed by default.
- Show parameter, state, and constant ownership without inferring meaning from
  names.
- Follow dependencies from a selected node to its operands and consumers.
- Explain transformed Programs, including gradients and optimizer transitions.
- Remain responsive for imported and authored Programs containing thousands of
  operation nodes.
- Accept serialized inspection data so viewing does not require model execution.

## Non-goals

- Do not add `Program.visualize()` or browser dependencies to `affon:compute`.
- Do not infer attention, residual blocks, embeddings, or other architecture
  concepts from operation or parameter names.
- Do not treat compiler plans or runtime telemetry as Program semantics.
- Do not expose full constant values by default.

## Package boundaries

### Inspection model

A small, DOM-free `@affon/inspector` package should normalize and index an
inspection object. It owns no rendering framework and can be tested in the Affon
runtime.

```ts
import { create_inspection_model } from '@affon/inspector'

const inspection = model.inspect()
const view = create_inspection_model(inspection)
```

The derived model should provide constant-time lookup rather than forcing every
renderer to repeatedly scan the raw node array:

```ts
interface InspectionModel {
  readonly inspection: ProgramInspection
  readonly nodes_by_id: ReadonlyMap<number, InspectionNode>
  readonly consumers_by_id: ReadonlyMap<number, readonly number[]>
  readonly scopes_by_id: ReadonlyMap<string, InspectionScope>
  readonly root_scope: InspectionScope
  readonly output_ids: ReadonlySet<number>
}

interface InspectionScope {
  readonly id: string
  readonly path: ProgramPath
  readonly parent_id: string | null
  readonly child_ids: readonly string[]
  readonly node_ids: readonly number[]
  readonly parameter_ids: readonly number[]
  readonly state_ids: readonly number[]
  readonly constant_ids: readonly number[]
}
```

All collections exposed by this layer should be immutable. Derived consumer
edges, counts, byte estimates, and scope summaries are recomputable and should
not inflate serialized inspection JSON.

### Interactive renderer

The browser viewer should initially live in Affon.ai as an inspection route or
component. It consumes only the DOM-free model and supports local JSON file
import. The same component can later be embedded by local tooling.

An Affon runtime module may later provide a convenience exporter, but exporting
plain JSON must remain sufficient:

```ts
fs.writeFileSync('decoder.affon-inspection.json', JSON.stringify(model.inspect()))
```

## Information architecture

The viewer has three synchronized areas:

1. **Structure** — a composition tree derived from `ProgramPath`. Selecting a
   scope focuses the other views. Scopes can be expanded to reveal operations.
2. **Graph** — a dependency DAG. Collapsed scopes appear as one boundary node;
   expanding a scope replaces it with its contained nodes and edges.
3. **Details** — the selected scope or node's exact role, spec, axes,
   provenance, initializer metadata, operands, consumers, path, and options.

Arguments, parameters, state, constants, outputs, and transitions should also be
available as focused tables. These are projections of the same selection model,
not separate sources of truth.

## Core interactions

- Select a tree scope, graph node, or table row and synchronize selection.
- Expand or collapse a composition scope.
- Search exact or partial names, operation names, provenance, dtype, and shape.
- Filter by role, operation, dtype, output membership, or selected scope.
- Focus on one node's immediate operands and consumers.
- Trace upstream to arguments and downstream to outputs.
- Highlight shared values by consumer count; tied parameters require no
  name-based special case.
- Copy a stable textual locator such as `node:42` or a scope path.
- Preserve only presentation state—expanded scopes, filters, selection, and
  viewport—outside the inspection object.

## Rendering strategy

The initial graph should use SVG for accessible labels and direct node
selection. Layout should run against visible nodes only:

- Start with composition scopes collapsed.
- Build a layered DAG from operand edges.
- Treat a collapsed scope as a supernode with incoming and outgoing boundary
  edges.
- Virtualize long parameter and node tables.
- Move layout to a worker before supporting very large expanded graphs.

The renderer must not assign a unique color to every operation. Color identifies
stable semantic roles—argument, parameter, state, constant, operation, and
output—while path nesting, labels, and shapes carry structural meaning.

## Inspection schema version 1

`ProgramInspection` version 1 constructs node dependencies and a scope tree:

- `nodes[].operands` defines graph edges.
- `nodes[].path` defines nesting.
- `role`, `spec`, `name`, and `provenance` define exact node meaning.
- `outputs` and `transitions` define observable results and state transitions.

It also records composed component interfaces that cannot be reconstructed
faithfully after child argument nodes have been replaced by parent bindings:

```ts
interface ProgramComponentInspection {
  readonly id: string
  readonly program: string
  readonly instance: string
  readonly path: ProgramPath
  readonly bindings: Readonly<Record<string, number>>
  readonly outputs: readonly number[]
}
```

The internal composition mechanism records these interfaces automatically;
Program authors provide no visualization annotations. Version 1 also exports
`ProgramNode`, narrows inspection kinds, supplies a compact `provenance_id`,
links optimizer transitions by node ID, and declares the constant-value policy.
`@affon/inspector` prepares persisted inspection objects with inline, summarized,
or redacted constants.

The adapter retains support for legacy unversioned inspection JSON. It derives
scopes from paths and resolves legacy transition parameter names, but exact
component interfaces remain unavailable when they were not serialized.

## Transformation and runtime overlays

Gradient and optimization Programs should use the same structural viewer.
Transitions appear as a separate overlay linked to their selected parameter
nodes. Gradient nodes link to the values named by their
`with_respect_to` metadata.

Compiler plans and runtime telemetry may later be joined by stable node ID, but
they must remain optional layers:

```text
semantic inspection  +  compile report  +  runtime samples
       required             optional             optional
```

This preserves `Program.inspect()` as a deterministic description of meaning
while still allowing the UI to explain lowering, memory, and timing.

## Delivery sequence

1. Implement and test the DOM-free inspection adapter against small authored,
   composed, multi-output, gradient, and optimized Programs.
2. Build the Affon.ai viewer with structure, tables, synchronized selection,
   search, filtering, and JSON import.
3. Add the dependency graph with collapsed scopes and neighborhood focus.
4. Extend the inspection schema with explicit component interfaces and a schema
   version, retaining the current adapter.
5. Add optional compile and runtime overlays only after stable node correlation
   is defined.

## First acceptance fixture

The decoder LM is a useful scale fixture but not a semantic special case. The
viewer should show, without decoder-specific code:

- one `token_ids` argument and one logits output;
- root-owned and block-owned parameters grouped by composition path;
- the nested `decoder / blocks.0 / attention` boundary;
- the tied token table as one parameter node with multiple consumers;
- no model state or inference transitions;
- the optimized Program's added labels argument and optimizer transition.

The same tests must also pass for a small non-neural arithmetic Program and an
imported ONNX Program.
