# Affon

Affon is a package built on the Hao runtime for compute and data applications.

This repository is the next Affon package boundary. Hao owns the embedded
JavaScript runtime, module loading, async execution, and addon interface.
Affon owns its domain modules, native implementation, and TypeScript API.

## Development

This repository currently uses the sibling Hao checkout for local development:

```bash
zig build test
```

The dependency will move to a released Hao source package when the first Hao
embedding release is available.
