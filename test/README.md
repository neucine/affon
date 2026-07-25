# Tests

`test/types/` contains the Bun TypeScript contract tests and is checked with:

```sh
bun x tsc -p test/types/tsconfig.json --noEmit
```

`test/e2e/compute/` is the migrated Affon compute parity corpus. It uses Hao's
`std:test` and `std:util` modules directly. The corpus excludes Affon's
separate `affon:nn` suite; compute tests remain here and are the source of
truth for parity work.

Run selected migrated tests with:

```sh
zig build install
./zig-out/bin/affon test test/e2e/compute/autograd.test.ts
```
