# Contributing

## Public API Naming

When adding or changing exported runtime or first-party package APIs, prefer names that match the public module's role instead of local implementation style.

Rules:
- ML-facing public functions should prefer `snake_case` when they expose tensor, training, dataset, checkpoint, architecture, or model-family vocabulary
- examples include `clear_grad`, `clip_grad_norm`, `no_grad`, `index_select`, `masked_fill`, and package-level helpers with similar ML contract shape
- compute docs and declarations should use `axis` as the public term, not `dim`
- public option-object fields should use the same naming style as the function family they belong to
- preserve existing exported names unless there is an explicit compatibility or migration decision
- module/layer constructors in `affon:nn` may keep established constructor-style names such as `Linear`, `LayerNorm`, `Dropout`, and `Sequential`
- JavaScript object methods that follow existing JS conventions may keep their established style, such as `toString`
- non-public implementation helpers and ordinary TypeScript package internals may use normal TypeScript style when they are not part of the exported ML contract

Review rule:
- do not introduce a new public camelCase ML helper unless there is a documented reason and the declaration docs explain the public contract

## File Naming

When adding new files, prefer lowercase hyphenated names such as `runtime-boundary.md` and `train-decoder-lm.ts`.

Rules:
- use lowercase hyphenated names for public docs, examples, package files, scripts, and new source files when there is no stronger local convention
- keep established local conventions where they already exist, such as type declaration names like `affon-compute.d.ts` or implementation files that intentionally match exported class names
- avoid adding new mixed-case filenames unless the filename needs to mirror a public class, package convention, or external tool expectation

## Examples And Apps

Use small local examples for focused API demonstration. When an example becomes a complete domain workload, prefer a top-level `apps/` entry.

Rules:
- package-local examples should stay close to the package API they demonstrate
- top-level `examples/` should stay focused on runtime concepts and public docs
- domain workloads that combine data, model assembly, training, checkpointing, reporting, or generation should move toward `apps/`
- keep app folder and file names lowercase hyphenated

## Dataset Work

When adding dataset capabilities, keep the shared pipeline grammar in mind:

- read examples
- transform examples
- derive or select inputs
- derive or select targets
- batch
- tensorize

Runtime dataset work should focus on stable grammar and broadly reusable transforms. Ecosystem adapters, source connectors, and task-specific data recipes should start in packages or `apps/` until they prove broad reuse.

## Exported API Docs

When a change adds or modifies exported public APIs, update the matching declaration docs in `packages/@types/affon/*.d.ts`.

If public docs or declarations are mirrored into another repository, copy
`docs/` and `packages/@types/affon/` from this checkout after the same change has landed here.

Use the API declaration-doc standard from `affon-arch`:
- `sync/2026-03-23-api-declaration-doc-standard.md`

Use JSDoc style comments.

Use standard JSDoc tags where applicable:
- `@summary`
- `@param`
- `@returns`
- `@example`
- `@see`

Use Affon custom tags where applicable:
- `@category`
- `@shape`
- `@axis`
- `@formula`
- `@math`
- `@usecase`

Rules:
- use `@param` and `@returns` for input and output contracts instead of ad hoc labels
- use `@formula` for plain-text formulas
- use `@math` for richer math notation when it adds value
- omit tags that do not meaningfully apply to the API
- do not mix the old `Description:` / `Shape:` / `Formula:` prose-label style with JSDoc tags in new or updated declarations

Review rule:
- do not land exported API changes with stale declaration comments
- if runtime behavior, shape semantics, or argument contracts change, the `.d.ts` docs must change in the same PR

## Example Notebooks

When adding or updating example notebooks in `examples/`:
- prefer small focused notebooks by topic or op family
- keep one op, loss, or activation per formula/example cell pair when practical
- print inputs and outputs clearly with `repr` enabled
- use 3D examples when they better represent real ML usage
- keep formulas in markdown cells and examples in code cells
- verify notebook JSON stays valid after edits
