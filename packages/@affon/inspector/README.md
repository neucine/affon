# `@affon/inspector`

`@affon/inspector` is the DOM-free interpretation layer for core inspection
contracts. Its first adapter indexes serialized `ProgramInspection` data and
derives scope hierarchy, dependency consumers, semantic roles, output
membership, optimizer-transition links, search results, traces, and
collapsed-scope graph projections without requiring a live Program or Session.
Executable lowering, optimization, memory, and runtime inspection can add
adapters here without coupling those producers to rendering or browser APIs.

```ts
import { create_inspection_model, prepare_inspection_export } from '@affon/inspector'

const inspection = source.inspect()
const model = create_inspection_model(inspection)
const consumers = model.consumers_by_id.get(inspection.outputs[0])

const persisted = prepare_inspection_export(inspection, { constants: 'summary' })
JSON.stringify(persisted)
```

The package intentionally owns no rendering or browser APIs. See
`apps/program-inspector` for the first interactive consumer.

## Compatibility boundary

This package accepts the versioned `ProgramInspection` contract and legacy
unversioned inspection JSON. It does not change `ProgramBuilder` or the way
Programs are authored. Serialized inspection objects and live
`program.inspect()` results use the same adapter path.

Version 1 records exact component bindings and outputs automatically during
nested Program invocation. Scopes remain a navigation projection of
`nodes[].path`, while `components_by_id` is the authoritative component
interface. `component_outputs_by_node_id` distinguishes values crossing a
composition boundary from terminal `output_ids`; a component output may retain
ordinary downstream consumers in its enclosing Program. Optimizer transitions
use `parameter_ids`; the adapter falls back to legacy parameter names only for
older JSON.

`prepare_inspection_export` makes persisted constant handling explicit. It
defaults to summaries and can instead inline or redact captured values.
