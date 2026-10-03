# Module Loader

This document records what AFFON currently supports in module loading and external package support.

## Built-in Modules

AFFON reserves the `affon:*` namespace for runtime-provided modules.

- `affon:*` specifiers refer to built-in modules shipped by the runtime.
- Public built-ins include modules such as `affon:compute`, `affon:ops`, `affon:optim`, `affon:dataset`, and `affon:checkpoint`.
- External packages should not use the `affon:*` namespace.

## External Package Support

AFFON supports external packages as ESM modules.

- normal ESM `import` syntax
- bare package imports resolved through a `node_modules`-compatible lookup model
- relative imports and package-to-package imports within the supported subset

AFFON does not currently promise general Node compatibility.

## Package Resolution

AFFON currently resolves module paths like this:

- `affon:*` specifiers resolve to built-in runtime modules
- relative specifiers such as `./x` and `../x` resolve from the importing file
- bare package specifiers resolve through a `node_modules`-compatible lookup

For a bare package import, AFFON starts from the importing file's directory.

It then checks directories in this order:

1. `<importer dir>/node_modules/<package>`
2. `<parent dir>/node_modules/<package>`
3. the next parent directory's `node_modules/<package>`
4. continue upward until the filesystem root

The first match found in that upward walk is used.

This means nested dependencies follow the installed directory layout.

Example:

```txt
app/
  node_modules/
    a/
      index.ts
      node_modules/
        c/
    b/
      index.ts
      node_modules/
        c/
```

If `a/index.ts` imports `c`, AFFON resolves `a/node_modules/c`.

If `b/index.ts` imports `c`, AFFON resolves `b/node_modules/c`.

So different packages can resolve different installed copies of the same dependency, depending on where the import comes from.

A top-level package copy is used only when the upward walk does not find a closer nested copy first.

## `package.json` Contract

For package entry resolution, AFFON currently honors this subset:

- `exports["."].import`
- `exports["."]`
- top-level string `exports`
- `main` as fallback

CommonJS packages are not supported:

- `type: "commonjs"` is rejected

## Compatibility Matrix

| Case | Status |
| --- | --- |
| `package.json` `exports["."].import` -> `.ts` entry | supported |
| `package.json` `main` -> `.ts` entry | supported |
| nested relative imports inside a package | supported |
| package importing built-in `affon:*` modules | supported |
| package-to-package bare imports inside `node_modules` | supported |
| subpath imports like `pkg/subpath` when the package layout resolves directly | supported |
| `exports` condition object with `import` present alongside other keys | supported |
| isolated module-private bindings do not leak into importer scope | supported |
| named import aliasing (`import { x as y }`) | supported |
| default exports (`export default ...`) with default import | supported |
| namespace imports (`import * as ns from ...`) | supported |
| combined default + named import syntax (`import x, { y } from ...`) | supported |
| selective re-exports (`export { x as y } from ...`) | supported |
| star re-exports (`export * from ...`) | supported |
| direct unsupported CommonJS rejection | supported |
| `require(...)` | out of scope |
| Node built-in module compatibility | out of scope |
| Node native addon compatibility | out of scope |
| full spec-level ESM semantics | not implemented |
| live bindings | not implemented |
| full cyclic-module semantics | not implemented |
| full conditional-exports engine | not implemented |
| `exports` maps where subpath resolution must be enforced rather than falling through to direct files | not verified |
| more complex conditional `exports` selection logic | not verified |
| default + namespace import syntax (`import x, * as ns from ...`) | not implemented |
