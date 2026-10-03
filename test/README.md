# Tests

`test/types/` contains the Bun TypeScript contract tests and is checked with:

```sh
bun x tsc -p test/types/tsconfig.json --noEmit
```

`test/e2e/compute/` is the migrated Affon compute parity corpus. It uses Hao's
`std:test` and `std:util` modules directly. Legacy neural-network parity remains
in its separate `affon:nn/legacy` suite; canonical Program API coverage lives
alongside compute tests here.

Run selected migrated tests with:

```sh
zig build install
./zig-out/bin/affon test test/e2e/compute/autograd.test.ts
```

## CUDA regressions

On a Linux host with a working NVIDIA driver, NVRTC, and cuBLAS, run:

```sh
./zig-out/bin/affon test test/cuda
```

These suites deliberately fail if CUDA is unavailable. They cover transfers,
views, selection, casts, parameter initialization over non-finite storage,
native graph execution, loss backward, clipping, and
multi-step SGD/Adam/AdamW parity against CPU. A tiny decoder additionally checks
tied embeddings, causal attention, training parity, and greedy/top-k generation. The
sibling `compute` repository's `zig build test` also contains CUDA numerical
regressions, which are skipped when its driver probe reports unavailable.

For synchronized benchmark timings, build with `zig build -Doptimize=ReleaseFast`
and run `./zig-out/bin/affon tools/bench-cuda.ts`. Uploads and first-use compilation
are outside the timed region; scalar reads synchronize each measured operation.


## Compute performance benchmarks

[test/benchmarks](benchmarks/README.md) contains the model-independent compute
benchmark cases, Affon/PyTorch workers, coverage reports and regression checks.
It runs separately from the normal correctness suite because timing requires a
ReleaseFast binary and a quiet host. The Python environment needs `torch` and
`safetensors`.

```sh
/path/to/python test/benchmarks/run.py \
  --affon /path/to/release/bin/affon \
  --compute /path/to/compute \
  --output /tmp/compute-bench-new
```

Run the benchmark reporting tests with:

```sh
python3 -m unittest discover -s test/benchmarks -p 'test_*.py'
```

Benchmark run outputs and investigation logs are local artifacts; keep reusable
runners, case registries, and correctness fixtures in this repository.
