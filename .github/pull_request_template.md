## Summary

- what changed
- why it changed

## Verification

- [ ] `zig build`
- [ ] `zig build test`
- [ ] `./zig-out/bin/affon test test/e2e/compute/autograd.test.ts`
- [ ] `./zig-out/bin/affon test packages/transformers/test`
- [ ] `./zig-out/bin/affon test packages/lm/test packages/tokenizers/test apps/decoder-lm/test`
- [ ] `bun x tsc -p test/types/tsconfig.json --noEmit`
- [ ] `bun x tsc -p apps/decoder-lm/tsconfig.json --noEmit`
- [ ] `bun x tsc -p packages/transformers/tsconfig.json --noEmit`
- [ ] `bun x tsc -p packages/lm/tsconfig.json --noEmit`
- [ ] `bun x tsc -p packages/tokenizers/tsconfig.json --noEmit`
- [ ] `bun x tsc -p packages/cnn/tsconfig.json --noEmit`
- [ ] `bun x tsc -p packages/vision/tsconfig.json --noEmit`

## Public contract

- [ ] updated `docs/` if user-facing runtime behavior changed
- [ ] updated `packages/@types/affon/*.d.ts` if exported API behavior changed
