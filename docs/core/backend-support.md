# Backend Support

Affon keeps one public compute contract across CPU, Metal, and CUDA:

```text
Program -> Session.compile -> Executable.run -> Tensor
```

Choose the target when creating a `Session`:

```ts
import { Session } from 'affon:compute'

const cpu = new Session({ device: 'cpu' })
const metal = new Session({ device: 'metal' })
const cuda = new Session({ device: 'cuda' })
```

`Tensor.from(...)` and the other top-level value constructors use the runtime's
default Session. Prefer an explicit Session when device choice, ownership, or
resource lifetime matters.

## Portable contract

The canonical operation vocabulary is exported by `affon:ops`. It includes:

- arithmetic, comparisons, selection, and common activations
- reductions and softmax
- dot products and matrix multiplication
- reshape, transpose, slicing, casting, concatenation, and stacking
- embedding, index selection, gather, and one-hot encoding
- layer normalization and the documented loss functions

The same functions accept formal tensors while a Program is authored and
evaluated tensors for immediate calculations. Shape, dtype, and axis validation
is shared across both uses.

CPU is the reference backend and the broadest compatibility target. Metal and
CUDA accelerate supported combinations, primarily `f32` model computation.
Storage support for a dtype does not imply that every operation implements that
dtype on every accelerator.

## Compatibility rules

- Supported public dtypes are `f32`, `f64`, and `i64`.
- Elementwise broadcasting is right-aligned: dimensions must be equal or one
  side must be `1`.
- Values used as indices are `i64` and are bounds-checked.
- Operations reject tensors owned by different Sessions or devices.
- Host reads such as `item()` and `to_array()` synchronize accelerator work.
- Unsupported backend, dtype, or shape combinations fail explicitly; Affon does
  not silently move a Program to another device.

Treat `not_implemented` or `unsupported_lowering` as a scoped backend gap, not
as evidence that the operation is absent from the public API. Run the same
Program in a CPU Session when you need a compatibility baseline.

## CUDA requirements

CUDA execution currently targets Linux with a working NVIDIA driver, NVRTC, and
cuBLAS. `AFFON_CUDA_DEVICE` selects the process CUDA ordinal before startup.
Use `cuda:N` in a Session device when the runtime and installation support that
ordinal. Simultaneous multi-device execution is not a documented contract.

## Verifying a workload

Backend support is best established for the exact Program, dtype, and shape you
intend to run:

1. Compile and run the Program on CPU as the reference.
2. Run it on the target accelerator with the same inputs.
3. Compare evaluated outputs using tolerances appropriate to the computation.
4. Exercise the training Program separately if you use `gradient(...)` or
   `optimize(...)`; forward support alone does not establish gradient coverage.

First-party package and app tests provide evidence for the configurations they
name. They should not be read as a blanket promise for every model or exported
graph.
