# Numerical and timing parity

This directory is the permanent parity test facility. The layout separates
contracts, execution, references, and reports:

| Location | Purpose |
| --- | --- |
| `catalogs/*.json` | Named, rerunnable cell batches |
| `catalogs/generators/*.py` | Catalog generators; write into `catalogs/` and the master coverage policy |
| `workers/*.ts` | Affon numerical and timing workers |
| `fixtures/*.py` | Independent numerical fixture formulas |
| `reference/probe-capabilities.py` | Numerical capability probes and catalog validation |
| `reference/benchmark-cells-pytorch.py` | Matched PyTorch operator and Program correctness and timing |
| `run.py`, `promote.py`, `report.py`, `index.py` | Batch execution, reviewed promotion, batch report, and artifact index |
| `test_*.py` | Facility and retained evidence regression tests |

[The artifact index](../../benchmarks/results/PARITY-INDEX.md) lists every
catalog and retained batch, plus the current aggregate summaries and HTML
report. Regenerate it with `python3 test/parity/index.py`. Historical evidence
keeps its recorded path and hash. The old `tools/probe-capabilities.py` and
`tools/benchmark-cells-pytorch.py` commands remain compatibility entry points;
new development belongs in `reference/`.

## Run a batch

Build Affon in ReleaseFast mode on a machine with the selected device and
install PyTorch for that device. Use a fresh output directory for every run:

```sh
zig build -Doptimize=ReleaseFast
python3 test/parity/run.py \
  --catalog test/parity/catalogs/cuda-first-256.json \
  --output-dir benchmarks/results/cuda-parity-run-YYYYMMDD \
  --iterations 10 --samples 5 --rounds 3
```

The runner checks every catalog cell against the candidate matrix. It saves a
copy of the catalog, binary and catalog hashes, a numerical capability report,
raw operator and composed Program timings, an `outcomes.json` containing only
verified links, a per-cell `manifest.json`, and a readable `report.md`. Run
`python3 test/parity/report.py PATH/manifest.json --output PATH/report.md` to
regenerate a report without using the GPU.

Every case receives a numerical probe. Cases that pass it receive both matched
PyTorch comparisons, each with an independent correctness check and repeated
timings. An incorrect comparison keeps its raw timed report in the matching
`*_failures` directory. A case that fails to compile or fails the numerical
probe is marked `probe_not_correct` for both timings; it cannot produce a valid
matched benchmark until that failure is resolved. The manifest records every
status, including benchmark errors, so the batch denominator is explicit.
Passing timing ratios are diagnostic, and no batch run accepts a regression
baseline automatically.

The runner refuses to overwrite an output directory. Keep each run's raw
artifacts, then review it before importing links into the current coverage
ledger and regenerating the whole-matrix HTML report.
For a complete batch of new cells, promote the validated evidence with
`python3 test/parity/promote.py PATH/TO/BATCH`. The promotion command rejects
incomplete batches and existing ledger cells. Then regenerate both current
timing summaries, the HTML report, and `python3 test/parity/index.py`.
For a reviewed replacement of existing cells, pass `--archive PATH` to retain
the prior aggregate operator and Program reports before their links change.

## Run a small set

The 12-cell `catalogs/metal-regression-smoke.json` suite spans f32/BF16
arithmetic, matmul, reduction, normalization, indexing, scatter, loss, top-k,
and quantized matmul on Metal. Regenerate it with
`python3 test/parity/catalogs/generators/generate-metal-regression-smoke.py`.
Run the full suite or repeat `--cell CELL_ID` to select exact cells from any
catalog. The batch snapshots only selected cells and records the source and
selected catalog hashes.

```sh
python3 test/parity/run.py \
  --catalog test/parity/catalogs/metal-regression-smoke.json \
  --output-dir benchmarks/results/parity-metal-smoke-YYYYMMDD \
  --iterations 10 --samples 5 --rounds 3
```

## Compare any two runs

`regression.py` is independent of the batch runner. It compares the shared
cell IDs in any two retained manifests, reports added and removed cells, and
exits nonzero if a previously passing numerical probe, operator comparison,
or Program comparison now fails. It writes machine-readable JSON and Markdown:

```sh
python3 test/parity/regression.py \
  benchmarks/results/PREVIOUS-BATCH \
  benchmarks/results/CURRENT-BATCH \
  --output-dir benchmarks/results/REGRESSION-COMPARISON
```

The comparison accepts a batch directory or `manifest.json` on either side.
It reports timing changes only when target and measurement protocol match;
timing signals are diagnostic and do not fail the check. No timing baseline is
accepted by this comparison.

## Validate the facility

```sh
python3 -m unittest discover -s test/parity -p 'test_*.py'
```

Add new cells to `benchmarks/coverage-policy.json`, the master
`benchmarks/capability-probes.json`, and a named catalog in `catalogs/`. Extend
the Affon workers and the independent Python/PyTorch references together when
adding an operation. Each named catalog can be rerun with the same command.

Catalog contracts, past failures, and reviewed reruns are documented in
[catalog notes](catalogs/README.md). The [artifact index](../../benchmarks/results/PARITY-INDEX.md)
links every catalog, generator, retained batch, and current report.
