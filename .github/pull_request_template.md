## Summary

- what changed
- why it changed

## Verification

- [ ] `zig build`
- [ ] `zig build test`
- [ ] `./tools/test-runtime.sh`
- [ ] `bun x tsc -p src/tsconfig.json --noEmit`
- [ ] `bun x tsc -p test/types/tsconfig.json --noEmit`
- [ ] `bun x tsc -p apps/decoder-lm/tsconfig.json --noEmit`
- [ ] `bun x tsc -p packages/@affon/models/tsconfig.json --noEmit`
- [ ] `bun x tsc -p packages/@affon/tokenizers/tsconfig.json --noEmit`

## Public contract

- [ ] updated `docs/` if user-facing runtime behavior changed
- [ ] updated `packages/@types/affon/*.d.ts` if exported API behavior changed
