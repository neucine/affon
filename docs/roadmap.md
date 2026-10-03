# Roadmap: semantic computational models

Affon is an experimental runtime for making computational models legible as
semantic programs. Its direction is to keep computation, parameters, mutable
state, differentiation, transformations, and hardware execution understandable
as one end-to-end system.

This is a systems hypothesis, not a claim that Affon should replace PyTorch,
become a universal model format, or support every exported model. TypeScript is
Affon's current host interface, and local execution is an important validation
environment; neither is the project's central distinction.

The question guiding the next phase is:

> Can a useful modern computational model preserve and explain its meaning - not
> only its numeric operators - from authored or imported graph down to the
> operations that execute on hardware?

Affon should answer that question with working evidence while remaining small
enough that a contributor can understand the system across its major layers.

## Direction and principles

- **Legibility over coverage.** Prefer one model whose semantics and execution
  can be explained completely over many models that happen to produce outputs.
- **Questions before schemas.** Define the questions Affon must answer before
  designing an IR, report format, or visualizer. A field without an explanatory
  use is not automatically semantic information.
- **Explicit semantics over preserved environments.** Record computation,
  parameters, state, inputs, outputs, effects, and compatibility directly. Do
  not treat a Python repository or container as the model abstraction.
- **Every semantic claim has an owner and evidence.** Distinguish facts declared
  by an author or source artifact, facts proven by inference, recognized
  patterns, user annotations, and unknown meaning.
- **Evidence over capability claims.** A code path, passing fixture, imported
  architecture, backend, and useful application are different kinds of evidence.
  State exactly which one has been established.
- **Transformations carry contracts.** Every rewrite must report its
  preconditions, whether it is exact or approximate, and which properties such
  as differentiation and export it preserves.
- **Importers are architecture tests.** Foreign models should reveal missing
  semantic primitives. Model count is not a success metric, and model-family
  special cases should not silently enter the core.
- **Model semantics and execution planning are separate.** Stable model meaning
  should not depend on a particular device, kernel, fusion, memory plan, or
  compiler specialization.
- **Applications provide pressure, not identity.** Packaging, serving,
  preprocessing, and lightweight learning experiments remain valuable when they
  test the model and runtime boundaries. They do not by themselves define Affon.
- **CPU and Metal are the primary validation targets.** CUDA remains useful but
  has less complete testing. Evidence from one backend does not establish
  correctness or performance on another.

## Starting point, not compatibility constraint

Affon and the extracted compute core already contain useful experiments in
semantic inference, named axes, graph plans, module provenance, autograd
provenance, structural export, and offline visualization. They demonstrate that
the direction is technically grounded.

They do not define the future contract. Earlier schemas and viewers may be
reused, simplified, replaced, or retired. The requirement is to account for what
each experiment learned and preserve only concepts that help answer the new
end-to-end semantic questions. Backward compatibility with an unpublished or
inactive inspection surface is not a roadmap objective.

## Compute-core architecture prerequisite

Semantic explanation depends on clean ownership inside the extracted compute
core. The extraction from the JavaScript runtime created the correct repository
boundary, but transitional dependencies still allow logical types to reach into
storage and concrete backends, semantic inference to consume capability types,
derivation code to call back through the engine facade, and Affon's binding to
import internal pipeline modules directly.

The compute architecture should be cleaned before the explanation model becomes
a large new subsystem. This is not adequately described as fixing an import
graph. The current implementation has overlapping immediate, ordinary-graph,
reusable-graph, and autograd execution models; semantic inference already makes
backend decisions; graph plans embed eager plans; materialized tensors carry
untyped autograd state; and a nominally stateless engine relies on global runtime
policy and resource state.

The target separates three domains and an explicit provider boundary:

```text
program meaning -> transformations / autodiff
       |
       v
compilation / optimization -> executable
       |            |
       v            v
target descriptions  session-owned runtime
                           |
                           v
                    backend providers

explanation projects linked evidence from each owned artifact
Affon enters through a small program/compiler/session API
```

This is a staged, evidence-led rewrite of the compute orchestration spine, not a
repository-wide rewrite of proven numerical work. A cleaned semantic
`TensorSpec`, plus `Program`, `Executable`, and `Session`, replace the mixed
responsibilities currently spread through the existing `TensorSpec`,
`ComputeGraph`, `GraphPlan`, `EagerPlan`, `Engine`, and global state. Existing
behavior supplies numerical and performance evidence, not API or test-suite
compatibility requirements. `OpTag`/`OpOptions`, validated mathematical rules,
kernels, FFI, and proven fusion/memory algorithms are assets to reuse when their
dependencies are clean. The new core is built and tested independently, starting
with a small vertical slice and expanding across the required operator inventory
and backends. AFFON then moves to it in one intentional breaking cutover, after
which the old core and obsolete tests are deleted. The complete target, task
sequence, and acceptance gates live in the compute repository's
`docs/architecture.md`.

AFFON should become program-first. The current graph supplies useful behavior
and stable-identity lessons, but it is not the target `Program` and must not be
confused with either a captured runtime trace or a device `Executable`. The
PyTorch-inspired eager surface is not
a long-term compatibility commitment. Immediate evaluation remains useful for
debugging, notebooks, tests, and small host expressions, but it should compile a
one-instruction program and execute it through the same executable/session path
as larger programs. Eager-only rules and silent program-to-eager fallback should
not exist in the new core; the old eager planner, runner, and tape are deleted at
cutover rather than preserved behind compatibility adapters.

## What a legible model requires

The current tensor, graph, autograd, checkpoint, importer, and package systems
contain parts of the answer. The roadmap should converge them around the
following concepts without prematurely declaring a universal IR:

- **Artifact contract:** versioned entry points, named inputs and outputs,
  dtype/shape constraints, processors, provenance, and referenced files.
- **Computation:** operators, data dependencies, control flow, randomness, and
  other effects needed to reproduce behavior.
- **Semantic roles:** named axes, value roles, module or region ownership,
  reduction and contraction meaning, and the evidence supporting each label.
- **Parameters:** learned values with stable identity independent of their
  current storage representation.
- **State:** mutable values with explicit initialization, ownership, update,
  reset, snapshot, and restore behavior.
- **Differentiation:** which values and regions are differentiable, how
  derivatives are constructed, and why a boundary is not differentiable.
- **Transformation:** the change made to a model and the semantic guarantees
  retained by that change.
- **Optimization:** target-independent rewrites, target-aware compilation choices,
  and backend-local preparation, each with legality, selection, numerical, and
  before/after evidence.
- **Execution:** backend eligibility, lowering decisions, fallbacks, selected
  kernels, synchronization, and measured numerical behavior.
- **Provenance:** explicit links among authored or imported nodes, semantic
  regions, derived gradient nodes, plan steps, fused regions, and runtime events.
- **Portability report:** a scoped account of what is executable, inspectable,
  transformable, differentiable, or lossy for a specific entry point, shape,
  dtype, and backend.

These concepts need not all share one serialized representation. In particular,
a durable model representation and a specialized execution graph may evolve at
different rates. They do require stable cross-layer identities so an explanation
can follow one value or operation as it is derived, transformed, planned, and
executed.

## Explanation model

AFFON should not expose a single graph and call it the truth. It should connect
several views with explicit provenance:

```text
model contract and authored intent
              |
              v
semantic computation graph
       |              |
       v              v
derived gradient   transformation / optimization history
       \              /
        v            v
          execution plan
                |
                v
        backend and runtime evidence
```

Each layer answers a different question:

- the model contract says how the computation is invoked and what persistent
  values it owns;
- the semantic graph says what values, dimensions, operations, and regions mean;
- the gradient graph explains how a requested derivative follows from the
  forward computation;
- transformation and optimization history explains what changed, why it was
  legal and selected, and which guarantees survived;
- the execution plan explains materialization, specialization, fusion, fallback,
  and backend selection;
- runtime evidence records what actually occurred for a concrete invocation.

Reports, text explanations, notebook views, and visual graphs are projections of
this connected model. Visualization is important, but it must not become an
independent source of semantics.

## Flagship milestone: one executable semantic explanation

The next major result is not another operator inventory or architecture count.
Select one real, bounded model already expressible in Affon and make its
computational meaning inspectable across the complete pipeline. After that path
is coherent, repeat it with an imported representation to reveal which semantics
survive export and which are absent.

```text
authored or imported model
  -> declared inputs, outputs, parameters, state, and named axes
  -> semantic operations and recognized regions
  -> forward graph and derived differentiation graph
  -> execution plan, backend decisions, and fallbacks
  -> selected kernels and recorded runtime evidence
  -> linked structural report and explicit visualization
  -> one semantically described transformation and one evidenced optimization
  -> saved artifact and reproducible reload
```

The model should be small enough for independent reference checks and diverse
enough to exercise meaningful parameters, named dimensions, differentiation,
graph structure, and backend lowering. A small decoder or similarly structured
native model may provide the first coherent explanation; importer coverage
should not block establishing the semantic inspection path.

The milestone succeeds when another contributor can answer, using Affon's
artifacts and reports rather than undocumented implementation knowledge:

1. What exactly is the model's invocation contract?
2. What does each important dimension, value, operation, and region mean?
3. Is that meaning declared, preserved, derived, recognized, or unknown?
4. Where did every parameter, mutable value, and derived gradient come from?
5. How does a semantic region decompose into primitive computation?
6. Why did each operation run on its selected backend and kernel?
7. What changed during a transformation or optimization, why was it valid, and
   why was it selected?
8. Does saving and reloading preserve the claimed behavior and annotations?

## Delivery sequence

| Stage | Work | Exit evidence |
| --- | --- | --- |
| 0 - Frame questions and characterize architecture | Inventory what the runtime knows, define the semantic questions, trace the four current execution models, and choose a vertical architecture slice | Each question has an authoritative owner, required evidence, and an explicit unknown state; accidental compatibility behavior, hidden global state, fallbacks, and Affon's internal dependencies are recorded |
| 1 - Prove the architecture seam | For a small operator slice, separate value type, instruction, invocation, and runtime value; split abstract evaluation from target lowering; introduce session ownership and a real executable | Immediate and multi-instruction calls use the same executable/session path; mathematical typing is backend-independent; mutable runtime policy is not hidden globally; stable identities connect program, compilation, and runtime evidence |
| 2 - Explain a native model | Produce the connected semantic account for one model already authored in Affon | The account answers the flagship questions and distinguishes declared, inferred, recognized, user-annotated, and unknown meaning |
| 3 - Project useful views | Provide machine-readable, concise text, and explicit visual projections from the same connected account | Exact-op, module/region, differentiation, and execution views agree on identity and support drill-down without duplicating semantic inference |
| 4 - Explain differentiation, change, and optimization | Link derived gradients to forward causes; implement one semantic transformation; and evidence one program, compiler, or backend optimization | The explanation shows why each requested gradient exists or is blocked, what changed, which guarantees survived, why an optimization was legal and selected, and whether its resulting path executed |
| 5 - Trace semantics to hardware | Connect semantic nodes and regions to backend eligibility, materialization, specialization, fusion, selected kernels, optimization records, and runtime evidence | A concrete invocation can be followed from model entry point to optimized executable operations on CPU and one additional validated backend where available |
| 6 - Stress with an imported graph | Map one foreign artifact into the same account and report preserved, decomposed, inferred, unsupported, and source-export-lost semantics | Reference inputs match within stated tolerances; missing meaning remains explicit and no model-family assumption is silently invented |
| 7 - Save and reproduce | Serialize the model contract, values, state, semantic annotations, transformation history, and required compatibility metadata | A fresh process reloads the artifact and reproduces the stated structure and behavior without the source framework at execution time |
| 8 - Challenge the abstraction | Repeat the path with a structurally different model and a different semantic stressor | Shared concepts survive without architecture-named core operations; necessary revisions are documented before expanding coverage |

Stages are evidence gates, not a promise that each requires a new public API.
Work may overlap, but later claims must not be made before their earlier semantic
dependencies are understood.

## Supporting experiments

Existing application work remains useful evidence:

- Hugging Face audits test weight mapping, preprocessing, numerical behavior,
  model state, and independent reference comparison.
- The ONNX package tests graph import and execution without making ONNX Affon's
  permanent model abstraction.
- The model-package experiment tests immutable artifacts, explicit adapter
  contracts, integrity, relocation, and runtime separation.
- Decoder training tests native model construction, differentiation, optimizer
  state, checkpointing, and longer-running execution.
- Vision serving tests preprocessing, packaging, operational boundaries,
  performance measurement, and deployment integration.

Promote a concept from an experiment into the shared runtime only when at least
one end-to-end investigation needs it and its semantic ownership is clear.

## What counts as support

Report support per **artifact revision + entry point + processor + shape/dtype
regime + backend**. Use the following terms narrowly:

- **Implemented:** a code path exists.
- **Inspected:** Affon can describe the relevant structure and semantics.
- **Verified:** behavior matched an independent reference under recorded
  conditions and tolerances.
- **Transformed:** a declared rewrite completed and its postconditions were
  checked.
- **Differentiated:** gradients were independently checked for the named values
  and execution path.
- **Reproduced:** a saved Affon artifact was loaded in a fresh process and
  retained the stated contract and behavior.

Do not generalize a result from a small fixture to an architecture family, from
one backend to another, or from inference to differentiation. Approximate
behavior and semantic loss are results to report, not conditions to hide.

## Boundaries

The following are not roadmap objectives unless later evidence changes the
project's question:

- full PyTorch, JAX, NumPy, or Hugging Face API compatibility;
- maintaining a separate eager architecture for PyTorch-like behavior;
- arbitrary Python interpretation or preservation of host-language behavior;
- maximum model-count coverage;
- training foundation models at distributed scale;
- universal round-trip export to source frameworks;
- an orchestration, registry, or serving platform;
- optimizing every kernel before model semantics and measurements justify it.

Performance still matters: a transparent system that cannot execute useful work
is incomplete. Optimize measured paths after correctness and semantic boundaries
are established, and preserve the evidence connecting an optimization to its
effect.

## Decision points

Reassess the direction at four points rather than treating the roadmap as an
open-ended commitment:

1. **After explaining the native model:** Does the connected explanation make
   the runtime materially easier to understand than independent graph, trace,
   and profiler dumps?
2. **After imported-graph comparison:** Can Affon state precisely which meaning
   survived, which was reconstructed with evidence, and which was lost?
3. **After transformation and differentiation:** Is the end-to-end legibility
   useful enough to justify the complexity it introduces?
4. **After the second model:** Are the abstractions durable, or is Affon
   accumulating architecture-specific machinery?

A negative answer is still useful evidence. Affon may remain a compact compute
runtime and completed body of systems knowledge without claiming a universal
model layer.

## Documentation ownership

Current API and implementation contracts remain with their code. Accepted
cross-repository architectural decisions and curated investigations belong in
`affon-arch`. Generated measurements, temporary progress notes, and local
handoffs are not public roadmap material; retain reproducible tools and stable
reports that support a claim.
