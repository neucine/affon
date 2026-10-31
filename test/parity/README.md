# Numerical and timing parity

A case list describes what to check. `run.py` saves numerical and timing
results; `regression.py` compares two runs. The [case and result definitions](SCHEMA.md)
explain every dimension and internal term. The [results index](../benchmark/results/PARITY-INDEX.md)
links saved runs and the report.

- `catalogs/`: named case lists and their generators.
- `workers/`: Affon execution; `reference/` and `fixtures/`: independent Python and PyTorch checks.
- `run.py`: run cases; `regression.py`: compare runs; `report.py`: render one run.
- `promote.py`: add reviewed passing results to the aggregate coverage report.

## Run a batch

Build Affon in ReleaseFast mode on a machine with the selected device and
install PyTorch for that device. Use a fresh output directory for every run:

```sh
zig build -Doptimize=ReleaseFast
python3 test/parity/run.py \
  --catalog test/parity/catalogs/cuda-first-256.json \
  --output-dir test/benchmark/results/cuda-parity-run-YYYYMMDD \
  --iterations 10 --samples 5 --rounds 3
```

The runner checks the cases named in the catalog. It saves a catalog copy,
raw numerical and timing results, a `manifest.json` summarizing every check,
and a readable `report.md`. Historical filenames such as `capability.json` and
`outcomes.json` are defined in [the schema](SCHEMA.md). Run
`python3 test/parity/report.py PATH/manifest.json --output PATH/report.md` to
regenerate a report without using the GPU.

Every case receives an independent numerical check. Cases that pass it receive both matched
PyTorch comparisons, each with an independent correctness check and repeated
timings. An incorrect comparison keeps its raw timed report in the matching
`*_failures` directory. A case that fails to compile or fails the numerical
numerical check is marked `probe_not_correct` for both timings; it cannot produce a valid
matched benchmark until that failure is resolved. The manifest records every
check status, including timing errors, so the case count is explicit.
Passing timing ratios are diagnostic, and no batch run accepts a regression
baseline automatically.

The runner refuses to overwrite an output directory. Keep each run's raw
results. Add reviewed passing checks to the aggregate coverage report when
that broader tracking is needed.
For a complete batch of new cells, add the reviewed results with
`python3 test/parity/promote.py PATH/TO/BATCH`. The promotion command rejects
incomplete batches and cells already recorded in the aggregate. Then regenerate both current
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
  --output-dir test/benchmark/results/parity-metal-smoke-YYYYMMDD \
  --iterations 10 --samples 5 --rounds 3
```

## Compare any two runs

For a kernel or runtime change, rerun every affected parity catalog with the
same target and timing settings as its retained batch. Compare each rerun with
`regression.py` and review correctness failures and timing suspects. When the
batch is complete and reviewed, replace its linked aggregate evidence with
`promote.py --archive PATH`, then regenerate both current timing summaries,
the HTML report, and the parity index. Keep the old raw evidence in the archive.

`regression.py` is independent of the batch runner. It compares the shared
cell IDs in any two retained manifests, reports added and removed cells, and
exits nonzero if a previously passing numerical probe, operator comparison,
or Program comparison now fails. It writes machine-readable JSON and Markdown:

```sh
python3 test/parity/regression.py \
  test/benchmark/results/PREVIOUS-BATCH \
  test/benchmark/results/CURRENT-BATCH \
  --output-dir test/benchmark/results/REGRESSION-COMPARISON
```

The comparison accepts a batch directory or `manifest.json` on either side.
It reports timing changes only when target and measurement protocol match;
timing signals are diagnostic and do not fail the check. No timing baseline is
accepted by this comparison.

## Validate the facility

```sh
python3 -m unittest discover -s test/parity -p 'test_*.py'
```

Add a named JSON case list in `catalogs/` to run new cases. Extend the Affon
workers and independent Python/PyTorch checks together when adding an
operation. To include those cases in the aggregate coverage report, also add
them to the planning files described in [the schema](SCHEMA.md).

Catalog contracts, past failures, and reviewed reruns are documented in
[catalog notes](catalogs/README.md). The [artifact index](../benchmark/results/PARITY-INDEX.md)
links every catalog, generator, retained batch, and current report.
