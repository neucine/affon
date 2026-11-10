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

For a checkpointed sweep, run `python3 test/parity/refresh.py PATH/TO/BATCH`
to rebuild its manifest and report from the completed per-cell files. Pending
checks remain pending. The HTML report generator also refreshes locally retained
active sweeps and displays their progress separately from reviewed aggregate
coverage. Sync remote checkpoint files before generating the report.

Every case receives an independent numerical check. Cases that pass it receive both matched
PyTorch comparisons, each with an independent correctness check and repeated
timings. An incorrect comparison keeps its raw timed report in the matching
`*_failures` directory. A case that fails to compile or fails the numerical
numerical check is marked `probe_not_correct` for both timings; it cannot produce a valid
matched benchmark until that failure is resolved. The manifest records every
check status, including timing errors, so the case count is explicit.
Passing timing ratios are diagnostic, and no batch run accepts a regression
baseline automatically.

The timing summaries classify each independent round using its median Affon time
divided by its median PyTorch time. `candidate_gap` means every round exceeds
1.25; `mixed_rounds` means only some rounds exceed 1.25; `better_than_peer`
means every round is below 1.0. Other results are `within_diagnostic_target`.
These labels describe the measured target and do not replace numerical parity.

Top-k peer timings use `reference/stable_topk.py`: PyTorch performs a full
stable sort along the selected axis, then narrows to `k`. This preserves the
lower-index tie rule checked by the numerical oracle. The reported peer cost
therefore includes sorting values that are not returned, and must not be read
as the cost of native `torch.topk`. State this contract when interpreting
top-k ratios or comparing a top-k optimization with the peer.

The runner refuses to overwrite an output directory. Keep each run's raw
results. Add reviewed passing checks to the aggregate coverage report when
that broader tracking is needed.
For a complete batch of new cells, add the reviewed results with
`python3 test/parity/promote.py PATH/TO/BATCH`. The promotion command rejects
incomplete batches and cells already recorded in the aggregate. Then regenerate both current
timing summaries, the HTML report, and `python3 test/parity/index.py`.
For a reviewed replacement, including a batch mixing existing and new cells,
pass `--archive PATH`. This retains prior ledger entries in `ledger-before.json`
and copies every existing aggregate operator and Program report before its link
changes. `reports-before.json` records which reports existed, including absent
reports for previously unsupported cells. The archive path must be new.

Before promotion, audit the complete batch against the intended target and
actual tested binary, with the peer version used by that run:

```sh
python3 test/parity/audit_complete_batch.py PATH/TO/BATCH \
  --expected-catalog PATH/TO/TARGET.json --binary PATH/TO/AFFON \
  --peer-version VERSION --output PATH/TO/AUDIT.json
```

This checks exact scope and contracts, binary identity, raw numerical results,
and every operator and Program round and sample. A passing subset does not
establish the full target. The audit accepts the explicitly selected peer
version instead of requiring a fixed historical PyTorch release.

When a reviewed verifier correction requires a separate rerun, preserve both
source batches. `reconcile.py BASE REPLACEMENT OUTPUT --expected-catalog TARGET
--reason TEXT --binary AFFON --peer-version VERSION` creates a fresh derived
collection, replacing every peer report in the replacement catalog's scope.
It requires a finished base, a complete passing replacement, identical binary,
target and sampling protocol, and unchanged workload and numerical contracts.
The final audit still requires every target cell. `reconciliation.json` records
source identities and hashes, old reports (including failures), and each
replacement report. This never resumes or overwrites either source sweep.

## Run a small set

The CUDA expansion target is `catalogs/cuda-metal-parity.json`. It mirrors
every currently supported Metal contract from the master catalog and reviewed
coverage ledger, changing only the backend in each ID. This is a target list,
not evidence of CUDA support. Regenerate it and the 142-cell
`catalogs/cuda-metal-smoke.json` subset with:

```sh
python3 test/parity/catalogs/generators/generate-cuda-metal-parity.py
```

Pass `--update-plan` to register mirrored cases and policy variants in the
master planning files. It rejects conflicting existing contracts and preserves
existing definitions. Registration does not change the support ledger.

The same generator writes `catalogs/cuda-metal-gap-regression.json`: the 1,350
mirrored contracts affected by the initial sine/cosine, quantized-matmul,
reduction-to-shape, and normalization changes. This broader list is useful for
checking layout and shape coverage after the smoke suite passes.

Start with the smoke subset on the CUDA machine, resolve numerical failures,
then expand to the full target. The subset includes every dtype/operation pair
and representative strided, offset, transposed, batched, and outer-axis cases.
Keep missing operations in the target so coverage gaps remain visible. Record
complete operator and Program checks before promoting a cell. Add newly
mirrored IDs and variants to the planning inputs before aggregate promotion
using `--update-plan`; generating the target alone does not alter those files.

Optimization follows correctness: retain a baseline, rank repeated timing
gaps, change a measured bottleneck, and compare the affected cells using
`regression.py`. Retain target identity and competing GPU activity with the
results; memory pressure or concurrent computation can invalidate a timing
conclusion.

For a stable timing gap, capture one exact cell with the selected Affon path,
schema 3 runtime evidence, and the same PyTorch operation used by parity:

```sh
python3 test/parity/profile-cell.py \
  --catalog test/parity/catalogs/metal-regression-smoke.json \
  --cell metal/f32/add/small_dense --kind operator_peer \
  --output /tmp/affon-add-operator-profile.json
```

Run again with `--kind program_peer` to examine composition and fusion.
The command requires PyTorch for the selected device and supports cells handled
by `workers/affon-benchmark.ts`; specialized workers report an explicit error.
It reuses the parity fixtures and Program, warms the executable, selects one
run after resetting the evidence buffer, and checks sampled output against
PyTorch. The output contains implementation identity, physical-step attribution,
kernel-count semantics, timing and memory evidence, and the PyTorch parity
timing samples. The instrumented Affon elapsed time includes observation
overhead; use an ordinary parity rerun for performance ratios and the complete
correctness audit. CUDA profiler markers permit a separate Nsight capture of
the same command; this command does not collect or correlate external activity.

CUDA optimizations must use the operation's general runtime dimensions,
strides, dtypes, and documented semantics. Catalog IDs and fixture values must
not influence backend dispatch. The seeded general-case checker exercises
independent shapes and values, checks every output against PyTorch after input
quantization, and records public-API rank rejections separately from numerical
passes:

```sh
python3 test/parity/reference/check-cuda-general.py \
  --binary zig-out/bin/affon --seed 314159 \
  --output-dir test/benchmark/results/parity-cuda-general-FRESH-NAME
```

`reference/check-cuda-quantized-general.py` separately checks every output of
packed-weight matmul with all f32/BF16 activation/metadata combinations, 2/4/8
bits, random shapes and values, odd groups crossing packed-word boundaries,
and independent views of all four inputs. It uses the same fixed f32
peak-scaled tolerance as the catalog oracle and requires a native CUDA path.
Both standalone checkers take `--binary`, `--seed`, and `--output-dir`.

Use multiple seeds and the complete affected contract catalogs for correctness.
For dense-loss tuning, `cuda-metal-loss-regression.json` contains 784 mirrored
contracts. `cuda-loss-general-performance-24.json` compares tiny, wide,
multi-row, offset, strided, and batched cases, including shapes outside the
Metal catalog. Keep the before/after evidence and review slower regimes as
well as improvements. Passing the smoke suite alone does not establish general
correctness or a universal performance gain.

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

## Resume a full sweep

`sweep.py` uses the same numerical oracle and matched peer runner as `run.py`,
with an atomic evidence file for every completed check. First check the full
numerical target, then resume for operator and Program evidence:

```sh
python3 test/parity/sweep.py \
  --catalog test/parity/catalogs/cuda-metal-parity.json \
  --binary zig-out/bin/affon \
  --output-dir test/benchmark/results/parity-cuda-metal-full-FRESH-NAME \
  --phase probe
python3 test/parity/sweep.py \
  --catalog test/parity/catalogs/cuda-metal-parity.json \
  --binary zig-out/bin/affon \
  --output-dir test/benchmark/results/parity-cuda-metal-full-FRESH-NAME \
  --phase peers --resume
```

Resume with the same catalog, binary, verifier/runner sources, machine target,
and timing settings. Changed identities require a fresh output directory.
Completed failures remain failed evidence on resume. A process lock prevents
concurrent sweeps from writing the same directory; `progress.json` records
current counts, while `run.json` binds provenance. Numerical completion alone
does not set `all_checks_passed`: every case also needs passing operator and
Program checks. The master catalog, support ledger, and accepted timing
baselines are never changed by a sweep.

Add a named JSON case list in `catalogs/` to run new cases. Extend the Affon
workers and independent Python/PyTorch checks together when adding an
operation. To include those cases in the aggregate coverage report, also add
them to the planning files described in [the schema](SCHEMA.md).

Catalog contracts, past failures, and reviewed reruns are documented in
[catalog notes](catalogs/README.md). The [artifact index](../benchmark/results/PARITY-INDEX.md)
links every catalog, generator, retained batch, and current report.
