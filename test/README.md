# Tests

`test/types/` contains the Bun TypeScript contract tests and is checked with:

```sh
bun x tsc -p test/types/tsconfig.json --noEmit
```

`test/e2e/program/` follows the public Program paradigm: Program authoring and
execution, operations shared by formal and evaluated tensors, and
neural-network authoring. Dataset and checkpoint integration stay in their own
focused directories. Package and application tests live beside their owners.

The Program boundary is organized by responsibility:

- `contracts.test.ts` locks down the runtime export and value contracts.
- `authoring.test.ts` covers composition, inspection, and transforms.
- `lifecycle.test.ts` covers Session, Tensor, and Executable ownership.
- `optimization.test.ts` covers loss templates, optimizer descriptors,
  accumulation, and schedules.
- `execution.test.ts` covers initialized state, validation, and lowering.
- `tensor-construction.test.ts` covers evaluated Tensor factories and their
  default Session behavior.
- `operation-parity.test.ts` proves one operation vocabulary works for formal
  and evaluated tensors.
- `operation-coverage.test.ts` exercises the supported operation families and
  metrics.
- `operation-contracts.test.ts` isolates representation, shape, dtype, and
  authoring failures.
- `neural-network.test.ts` covers the `ProgramBuilder.nn` authoring helpers.
- `test/types/program.test.ts` locks down the compile-time public surface and
  verifies removed compatibility modules remain unavailable.

Run the canonical runtime suites with:

```sh
zig build install
./tools/test-runtime.sh
```

Run only the Program API boundary with:

```sh
./zig-out/bin/affon test test/e2e/program
```

Backend coverage is exercised through the same Program suites by selecting the
desired runtime device.
