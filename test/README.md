# Runtime tests

Tests live with the public feature they exercise. Numerical comparisons and
performance evidence live under `test/parity/`.

| Directory | Purpose |
| --- | --- |
| `test/e2e/program/authoring/` | Program construction, composition, inspection, and export contracts |
| `test/e2e/program/execution/` | Session, Executable, Tensor, state, and ownership behavior |
| `test/e2e/program/operations/` | Public operation behavior, shapes, dtypes, and invalid inputs |
| `test/e2e/program/training/` | Gradients, layers, optimizer transitions, and complete training workflows |
| `test/e2e/dataset/` | Dataset loading, transformation, and tokenization |
| `test/e2e/checkpoint/` | Checkpoint and safetensors behavior |
| `test/e2e/cli/` | Subprocess checks for commands, exits, and package resolution |
| `test/types/` | Compile-time public TypeScript contracts |
| `test/parity/` | Independent numerical comparisons, peer timings, catalogs, and coverage reporting |
| `test/support/` | Helpers shared by more than one e2e feature |

Package and application tests stay beside their owners. The e2e coverage
inventory is in [e2e/COVERAGE.md](e2e/COVERAGE.md).

## Add a test

- Place it beside the behavior's existing tests. Use a `*.test.ts` name that
  identifies the feature or workflow.
- Keep valid behavior and its related invalid inputs close together. Use a
  separate file when a case matrix becomes large enough to obscure the behavior.
- Use a table for variations of one behavior; use separate named tests for
  different failures, lifecycle stages, or state transitions.
- Assert observable values, state changes, ownership, or diagnostics. Shape and
  finite-value assertions alone rarely establish a workflow contract.
- Dispose Session-owned values and clean temporary files. Add a helper to
  `test/support/` only when at least two files need the same setup or cleanup.
- Update the coverage inventory when a public workflow or failure path changes.

## Run tests

```sh
zig build install
./tools/test-runtime.sh
./zig-out/bin/affon test test/e2e/program
bun x tsc -p test/types/tsconfig.json --noEmit
```

`tools/test-runtime.sh` runs the Program, dataset, checkpoint, package, app,
and CLI process suites. The Affon runner discovers `*.test.ts` recursively,
so adding a test inside a feature directory needs no runner edit.
