# Roadmap: legible computational Programs

Affon is an experimental runtime for making computational models legible as
Programs. The Program-centric foundation is shipped: authored and imported
Programs are inspectable, transformed explicitly, compiled by a Session, and
executed with Session-owned state.

The next question is not how to preserve the former eager or captured-graph
APIs. It is how far one coherent Program model can carry meaning from authoring
or import through differentiation, optimization, execution, and explanation.

## Principles

- **One canonical authoring model.** New compute, model, package, and app code
  should author or import Programs. Immediate evaluated-tensor operations remain
  useful for preprocessing, inspection, and small calculations, not as a second
  training architecture.
- **Legibility over coverage.** Prefer a smaller capability whose behavior can
  be explained and tested end to end over a large compatibility claim.
- **Explicit ownership.** Sessions own tensors, executables, and execution
  state. Packages define reusable Programs; applications own runtime policy.
- **Meaning is separate from execution.** Program structure must not depend on
  a particular kernel, fusion choice, memory plan, or device.
- **Evidence scopes every claim.** A supported operator, passing fixture,
  imported architecture, backend, and production-ready application are
  different levels of evidence.
- **Importers expose gaps.** Foreign formats should map into the canonical
  Program model or report an unsupported construct explicitly, not introduce a
  parallel runtime API.

## Current foundation

The public API already provides:

- named `Program` authoring and composition
- formal arguments, parameters, persistent state, constants, and named axes
- structural inspection with stable provenance
- `gradient(...)` and `optimize(...)` as Program transformations
- Session compilation, initialization, execution, and deterministic ownership
- CPU, Metal, and CUDA execution behind the same Program contract
- SafeTensors checkpoints and first-party tokenizer, model, ONNX, and Hugging
  Face packages

This foundation is the compatibility boundary for new work. Historical graph,
module, tape, or eager authoring surfaces are not alternative public APIs.

## Near-term priorities

### Complete the Program vocabulary

Broaden operation and gradient coverage where real Programs require it. Add
multi-output representation deliberately so operations such as top-k can return
linked values without duplicated work or an eager-only exception.

### Strengthen semantic inspection

Make `Program.inspect()` answer practical questions about invocation contracts,
parameter and state identity, named axes, composition boundaries,
transformations, and provenance. Unknown meaning should remain explicit rather
than inferred from model-family conventions.

### Explain compilation and execution

Connect stable Program nodes to lowering decisions and runtime evidence without
making backend plans part of Program meaning. Reports should explain unsupported
operations and fallbacks with the exact dtype, shape, and target involved.

### Expand model evidence

Use bounded authored and imported models to validate the complete path:

```text
Program -> inspect -> transform -> compile -> initialize -> run -> checkpoint
```

Each model claim should name its fixtures, processor policy, numeric tolerance,
backend, and limitations. Architecture count alone is not a milestone.

### Improve training and serving workflows

Build reusable examples around explicit `ExecutionState`, checkpoint resume,
dataset Session ownership, metrics, telemetry, and request cleanup. Application
policy—batching, decoding, caching, scheduling, and deployment—should remain
outside model Program definitions.

## Flagship milestone

The next major result is one real model whose computation can be followed
coherently from source artifact or authored definition to evaluated output:

1. Its inputs, outputs, parameters, state, axes, and processors are explicit.
2. Forward and derived training Programs retain stable provenance.
3. Compilation reports why the requested backend can or cannot execute it.
4. Runtime telemetry links resource use to a concrete invocation.
5. Saving and reloading preserves the claimed behavior.
6. An independent reference establishes numeric agreement within stated
   tolerances.

After that path is coherent for an authored Program, repeat it with an imported
artifact and report exactly which semantics survived the source format.

## Non-goals

- recreating a PyTorch-compatible object or mutation model
- maintaining a separate eager training architecture
- treating ONNX or any foreign graph as Affon's internal model
- claiming universal model, dtype, or backend coverage
- embedding application concerns such as HTTP serving or prompt policy in the
  compute core

Affon succeeds when its examples are both useful and honest: the repository
itself should demonstrate the canonical API, make ownership visible, and state
the evidence behind every compatibility claim.
