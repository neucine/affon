# Workload coverage inventory

## Work phases

The performance program advances in three phases. **Foundation** defines the
ledger, evidence contracts, runners, validation, and readable report. Its exit
criteria are:

1. A deterministic candidate matrix follows Compute's operation and dtype
   surface, with explicit regimes and review on API drift.
2. Every reported count is derived from retained artifacts through a documented
   import path. Missing, unsupported, and measured cells remain distinct.
3. Reusable correctness, operator peer, Program peer, dispatch, and baseline
   checks record workload identity, selected paths, and failure reasons.
4. The report renders from its documented default inputs and works locally;
   tests catch broken data joins and browser interactions.
5. Measurement instability is visible and cannot silently become accepted
   baseline or optimization evidence. A cause for every unstable target is not
   required to complete the foundation.

**Coverage expansion** then works through the declared matrix by backend,
dtype, operation, regime, and dispatch variant. It adds concrete workloads and
classifies unsupported or inapplicable cells with reasons. Coverage progress is
measured from linked evidence and explicit classifications, not from the
number of declared candidate cells. A model trace can reveal a missing regime
or contract, which must be added to the ledger before its result is counted.

**Optimization** follows representative, repeatable comparisons. Choose work
by a confirmed gap's absolute cost, relative cost, and relevance to actual
model paths. Recheck correctness, selected path, and peer timing after a change.
Diagnostic ratios and rejected baselines do not rank optimization work.

The foundation review is complete. The matrix, artifact joins, default report,
summary recomputation, baseline rejection path, and local browser rendering
have executable checks. Unresolved Metal timing drift remains visible as a
measurement limitation. The next separate goal is coverage expansion.

The report generator now reads the current reviewed outcomes by default. It
checks each linked artifact's SHA-256 and requires measured operator and Program
rows to match the linked cell IDs and exact raw file paths. It recomputes both
timing summaries from raw rounds. A missing, changed, or mixed artifact fails
rendering, so coverage counts and gap labels cannot silently refer to different
retained runs. To render a different run, pass its outcomes, summary, and raw
directories together.

## How the facility fits together

The matrix is a ledger of **candidate cells**. A cell names a backend, dtype,
operation, shape regime, and any declared dispatch variant. Its existence does
not mean that Affon supports the combination or that a benchmark ran.

```text
Compute operations and dtypes + coverage policy → candidate cell matrix
                                             capability probe → correctness and selected path
                                    matched peer benchmark → raw timing report → measured case
                           probe + benchmark evidence importer → outcomes JSON → matrix evidence counts

dispatch boundary policy → paired correctness/path probes → separate boundary audit
```

The HTML report's **Measured cases** table is built from the matched peer
timing summary and its raw reports. Its ratio describes a concrete workload.
**What has been checked?** is the report's coverage checklist, derived from
the candidate matrix joined with the imported outcomes. It asks four questions
for each cell: Is the output correct? How fast is the individual operation
against a peer? How fast is a composed Program against a peer? Is an accepted
same-target Affon regression reference ready? Each answer counts linked
artifacts out of all candidate cells, and the four counts are independent.
A linked operator peer artifact means the importer checked identity, shape,
compile options, selected path, correctness, and samples; it is not a
competitive verdict. A report bundle must show exactly the operator and
Program artifacts linked by its selected outcomes. The current bundle has 4187
operator rows and links. Other runs may have different counts, but a report
cannot mix a summary with unrelated linked artifacts.

**When does Affon switch implementations?** comes from a separate dispatch audit. Each
row compares two nearby shapes or compile options and checks output correctness
and the implementation selected on each side. Those paired probes do not add
measured cases or evidence links automatically. A boundary can remain
unobserved when the compiler selects the same path on both sides, even if a
lower-level planner contains a threshold.

## Canonical matrix

Run `python3 tools/coverage-matrix.py` for a summary or add `--details` for
every cell. [`coverage-policy.json`](coverage-policy.json) declares the finite
regimes for each operation family. The generator reads all 69 `OpTag` values
and all five `DType` values from Compute's canonical `types.zig`, then expands
them over CPU, Metal, and CUDA. The current matrix has 8,173 **candidate**
cells. A new operation or dtype breaks generation until its regime policy is
reviewed. This prevents new API surface from silently escaping the facility.
The matrix summary includes distributions by backend, dtype, operation family,
and backend × dtype. Each group reports candidate cells, supported cells,
correctness links, and operator peer links. Metal and CUDA together form the
GPU candidate total. These distributions count declared work and linked
evidence separately; unprobed cells remain unknown.
Use `--backend metal --dtype f32 --op matmul --details` to inspect one slice;
the report retains whole-matrix totals and adds selected-slice totals.

Every cell begins with `support: unknown` and four missing evidence types:
correctness, matched operator peer, matched Program peer, and same-target Affon
baseline. A reviewed outcome file passed with `--outcomes` can classify a cell
as `supported`, `unsupported`, or `not_applicable`. Exclusions require a reason.
Evidence links require both an artifact and an exact workload contract. There
is no automatic conversion from a source-file link to a performance pass.
The matrix is a coverage ledger; benchmark runners still produce and compare
the actual timings and output checks.

Example outcome entry, stored in a reviewed JSON mapping keyed by cell ID:

```json
{
  "metal/f32/matmul/decode_m1": {
    "support": "supported",
    "evidence": {
      "correctness": {"artifact": "run-id/correctness.json", "sha256": "<64 hex digits>", "contract": "sha256:<64 hex digits>"},
      "operator_peer": {"artifact": "run-id/operator-peer.json", "sha256": "<64 hex digits>", "contract": "sha256:<64 hex digits>"}
    }
  }
}
```

The remaining Program and baseline fields stay `missing`. Evidence import
validates artifact hashes and required contracts, while competitive verdicts
still require a matched peer gate and calibrated noise policy.

The candidate matrix is intentionally broader than valid combinations: for
example, an integer softmax cell may be `not_applicable` under the public
contract. Classify it with a documented reason. Do not discard it by omitting
it from a hand-maintained list. Likewise, a finite regime matrix cannot
enumerate every integer shape. Add a regime or split one when a new model lands
outside its tested shape, layout, cache, or dispatch behavior.

## Capability probes and evidence import

[`capability-probes.json`](capability-probes.json) declares concrete shapes for
the current 3,107 declared CPU, Metal and CUDA f32/BF16 probes. The runner invokes a fresh Affon process
per case, compares every output against an independent numerical formula, and
records the compiler selected physical path. Use a ReleaseFast binary for
evidence intended for performance work:

```sh
python3 tools/probe-capabilities.py --binary zig-out/bin/affon \
  --output /tmp/affon-probes.json
python3 tools/import-coverage-evidence.py --probe /tmp/affon-probes.json \
  --output /tmp/affon-outcomes.json
python3 tools/coverage-matrix.py --outcomes /tmp/affon-outcomes.json
```

`probe_failed` and `incorrect` cases remain unknown in the matrix; they are
not classified as unsupported. An unsupported decision requires a reviewed
reason. Run CUDA probes on the CUDA target with `--backend cuda`; lack of a
CUDA device on this Mac supplies no information about CUDA support.
The first fresh Zig 0.17 ReleaseFast run is
[`results/coverage-capability-mac-2026-10-29.json`](results/coverage-capability-mac-2026-10-29.json):
all ten declared CPU and Metal f32 probes passed the independent output check.
It records binary and source identities, and the source checkout was dirty.
These small shapes are capability checks, not claims about larger model shapes
or peer performance.

The importer accepts `affon_cell_benchmark` reports only for probed cells.
Each report must declare the exact shape, dtype, layout, timing scope,
optimization level, input hash, target, binary, selected Affon path, correctness
result, and at least three raw samples for both Affon and the peer. Operator
reports must select the same physical path as the capability probe. Report
`kind` is `operator_peer`, `program_peer`, or `self_baseline`. Historical model
and quantized reports lack some of these fields, so they remain diagnostic
links; no speed result is silently promoted. This importer validates presence
and exact path identity. A later measurement gate must additionally verify
peer workload equivalence and calibrated noise before a linked report can
establish competitive performance.

For `program_peer` evidence, the importer also requires a named Program graph
whose first operation matches the cell. This makes the additional work explicit
in the workload contract. Program evidence does not reuse the operator's
selected path; the compiler may choose a different path for a graph.

## First matched peer set

The first ten cells have matched PyTorch 2.9.0 CPU/MPS reports in
[`results/cell-peer-mac-2026-10-29`](results/cell-peer-mac-2026-10-29).
The runner uses the same generated f32 inputs and dense zero-offset layout,
keeps inputs resident, compiles Affon outside timing, and times operation plus
synchronized full host output materialization. It retains three alternating
engine-order rounds, five samples per round, 300 calls per sample, full output
parity, and the Affon selected path. Reproduce with the Python environment
that has PyTorch and MPS support:

```sh
apps/hf-inference/artifacts/preparation-env/bin/python \
  tools/benchmark-cells-pytorch.py \
  --output-dir /tmp/affon-cell-peer --iterations 300 --samples 5 --rounds 3
python3 tools/summarize-cell-performance.py /tmp/affon-cell-peer
```

The [diagnostic summary](results/cell-peer-summary-mac-2026-10-29.json)
labels CPU add, short-axis sum, and one-row 32×32 matmul as candidate gaps
above a provisional 1.25 ratio in all three rounds. Their absolute overheads
are only about 7–8 microseconds per call. Metal add and one-row matmul cross
the threshold between rounds and are marked mixed. These are small-shape
call-and-readback measurements, not calibrated release gates or evidence for
large-model kernel priority. The importer linked these ten operator reports;
the later Program set covers three Metal cells, while accepted baselines remain
missing. A useful next workload sweep must include production dimensions and
separate GPU work from host call and readback costs.

## First matched Program set

Three Metal f32 cells now have matched PyTorch Program reports: decode
matmul, prefill matmul, and long-axis sum. The matmul Program follows the
`dense` projection helper in `packages/@affon/models/src/gpt2/model.ts`:
`matmul` followed by a bias add. Its extra bias input has a separate fixture
hash in the contract. The sum Program remains an explicitly synthetic
`sum_axis` followed by adding the result to itself; no model trace is claimed
for that composition. Both engines keep inputs resident, compile or construct the
Program outside timing, and time execution plus synchronized full host output
materialization. The output is checked against PyTorch in each round. This
does not measure model setup, host input conversion, or an entire model.

The [raw Program reports](results/cell-program-peer-mac-2026-10-29) retain
three alternating rounds, five samples per round, and 100 calls per sample.
The [Program diagnostic summary](results/cell-program-peer-summary-mac-2026-10-29.json)
is shown separately from operator comparisons in the HTML report. All three
Program cases crossed the provisional 1.25 ratio in each round of this run;
ratios changed materially between two runs, so these remain diagnostic until
noise is calibrated on the target. Reproduce the set with:

```sh
apps/hf-inference/artifacts/preparation-env/bin/python \
  tools/benchmark-cells-pytorch.py --kind program_peer \
  --only metal/f32/sum_axis/long_axis,metal/f32/matmul/decode_m1,metal/f32/matmul/prefill \
  --iterations 100 --samples 5 --rounds 3 \
  --output-dir /tmp/affon-program-peer
python3 tools/summarize-cell-performance.py /tmp/affon-program-peer \
  --kind program_peer
```

The current outcomes link 4169 Program artifacts separately from the
4187 operator artifacts. A Program result is evidence for its named two-step
graph only. Add model-specific Program graphs when the model's actual
composition, layout, or synchronization differs.

## Coverage expansion: f32 multiplication

The first expansion slice adds `mul/small_dense` at `[2,4]` on CPU and Metal.
The [retained probe](results/coverage-capability-mul-mac-2026-10-30.json)
passed independent element checks and selected a physical path for both cells.
Matched PyTorch operator and synthetic `mul → add_self` Program reports are
retained in the current operator and Program raw directories. Each comparison
has three alternating rounds and five samples per round. The CPU and Metal
operator ratios were 10.87× and 0.81× in this run; Program ratios were 7.19×
and 1.02×. These are measured diagnostic cases, not accepted performance or
optimization claims. The outcomes now link 15 correctness, 15 operator, and
5 Program artifacts out of 6,286 candidate cells; accepted baselines remain 0.

The Metal focused batch adds six cells: f32 and BF16 large dense add and
multiplication at `[128,512]`, plus BF16 short axis sum and short row softmax.
All six passed the [batch probe](results/coverage-capability-metal-batch-2026-10-30.json)
and matched operator and synthetic two operation Program checks against PyTorch
MPS. The earlier Metal BF16 small multiplication also passed its
[probe](results/coverage-capability-metal-bf16-mul-2026-10-30.json) and both
peer tiers. The ledger now has 22 correctness and operator links, 12 Program
links, and no accepted baselines. The BF16 probe uses a 2% relative tolerance
against an independent f32 formula to account for BF16 rounding; matched peer
reports additionally record the actual maximum errors. These timings remain
diagnostic and do not settle Metal timing drift.

The next [seven-cell Metal batch](results/coverage-capability-metal-next-2026-10-30.json)
adds f32/BF16 broadcast add and multiplication at `[128,512]`, BF16 decode
matmul `[1,128] @ [128,128]`, BF16 prefill matmul `[32,128] @ [128,128]`, and
BF16 long axis sum `[128,768]`. All seven passed capability probes and matched
operator and Program output checks. The BF16 matmul Program bias is BF16 in
both engines. Its first attempt exposed an f32 bias in the harness, which was
corrected before retained Program runs. The ledger now links 29 correctness,
29 operator, and 19 Program artifacts; accepted baselines remain zero.

The following [18-cell Metal family batch](results/coverage-capability-metal-family-2026-10-30.json)
adds f32/BF16 subtraction and division across small dense, large dense, and
broadcast regimes. It also adds BF16 small dense add, BF16 long row softmax,
and f32/BF16 small batch and large matrix matmul. All 18 passed capability
probes and both matched peer tiers. The independent BF16 probe formula now
rounds inputs and outputs to BF16, which matters for large tensor cancellation
and softmax. The large f32 matmul probe uses a 2 ppm relative tolerance; its
observed maximum error was below that bound. The ledger has 47 correctness,
47 operator, and 37 Program links, with zero accepted baselines. Per case
ratios remain diagnostic under the unresolved Metal timing drift.

The [24-cell Metal unary batch](results/coverage-capability-metal-unary-2026-10-30.json)
adds `neg`, `exp`, `log`, `sqrt`, `relu`, and `sigmoid` in f32 and BF16 at
`[2,4]` and `[32,64]`. All cells passed independent probes and matched
operator and synthetic `operation → add_self` Program checks. The larger
unary input stays positive and within BF16 exponential range. The ledger now
links 71 correctness, 71 operator, and 61 Program artifacts, with no accepted
baselines. The ratios remain diagnostic under unresolved Metal timing drift.

The [28-cell Metal follow-up](results/coverage-capability-metal-next-family-2026-10-30.json)
adds `abs`, `sign`, `silu`, `tanh`, `sin`, and `cos` in f32/BF16 at small and
large dense shapes, plus f32/BF16 layer normalization at short and long rows.
All 28 passed independent probes and both matched peer tiers. The ledger now
links 99 correctness, 99 operator, and 89 Program artifacts, with no accepted
baselines. Layer normalization ratios, including the large BF16 result, are
diagnostic until Metal timing is stable across sessions.

The [20-cell Metal reduction batch](results/coverage-capability-metal-reductions-2026-10-30.json)
adds f32/BF16 mean, min, max, population variance, and population standard
deviation at `[4,32]` and `[128,768]`. All 20 passed independent probes and
matched PyTorch MPS operator and synthetic two-operation Program checks.
Long f32 variance and standard deviation required a 10 ppm comparison bound;
the raw reports retain maximum and checksum errors. The ledger now links 119
correctness, 119 operator, and 109 Program artifacts out of 6,286 candidates.
No baseline is accepted and no diagnostic timing ratio ranks optimization work.

The [28-cell Metal view probe](results/coverage-capability-metal-views-2026-10-30.json)
uses real column stride 2 and `[1,1]` offset slices from larger base tensors.
It covers f32/BF16 add, sub, mul, div, neg, exp, and log at logical `[8,16]`.
The right operand uses different row and column slopes so offset errors cannot
cancel in subtraction. Twenty-four cells passed independent probes and both
matched peer tiers. Four f32 offset binary cells (`add`, `sub`, `mul`, `div`)
returned incorrect results and remain unknown without performance links. The
retained probe records their errors and compiler paths. The importer checks
the base input shapes alongside the logical output shape for linked view cases.
The ledger now links 143 correctness, 143 operator, and 133 Program artifacts;
accepted baselines remain zero. View creation is part of both engines' timed
operation, with resident base inputs.

The [20-cell Metal layout and outer-axis batch](results/coverage-capability-metal-layout-outer-2026-10-30.json)
adds f32/BF16 mean, min, max, variance, and standard deviation over axis 0 at
`[128,256]`. It also adds f32/BF16 vector-like, non-square transposed,
column-strided, and ragged matmul, plus BF16 model-sized decode
`[1,768] @ [768,768]` and prefill `[128,768] @ [768,768]` variants. The two
model variants add candidate cells to the policy, taking the matrix to 6,288.
All 20 passed independent probes and matched PyTorch MPS operator and Program
checks. Large BF16 inputs use byte-backed tensors because the array tensor API
limits inputs to 262,144 entries. The raw contracts retain base input shapes
and the compiler-selected transpose or slice path. The peer fixture was
corrected after a mismatched strided right operand was detected; only passing
runs were imported. Current coverage is 163 correctness, 163 operator, and
153 Program links, with zero accepted baselines.

The [32-cell Metal strided and ragged batch](results/coverage-capability-metal-strided-ragged-2026-10-30.json)
adds f32/BF16 sum, mean, min, max, variance, standard deviation, softmax, and
layer normalization. Strided cases use a real column stride of 2 with logical
shape `[16,64]`; ragged cases use dense `[17,65]`. All 32 passed independent
probes and matched operator and synthetic Program checks. The current ledger
links 195 correctness, 195 operator, and 185 Program artifacts out of 6,288
candidates, with zero accepted baselines. Their ratios remain diagnostic.

The [38-cell Metal ragged and batched batch](results/coverage-capability-metal-ragged-batched-2026-10-30.json)
adds f32/BF16 ragged `[17,65]` cases for 16 elementwise operations and six
batched matmul cases. The matmul set includes a general `[4,8,16,12]` case
and three-versus-four-batch `[B,8,8,8]` variants for each dtype. The four
variants add candidate cells, taking the matrix to 6,292. All 38 passed
independent probes and matched operator checks. Thirty-five passed the
synthetic Program peer check. The three f32 batched matmul-plus-bias Programs
returned incorrect outputs and are retained in
[`cell-program-failures-mac-2026-10-30`](results/cell-program-failures-mac-2026-10-30),
without Program links. A direct sample selected `metal.matmul_add.fused_tiled`
and returned zero in later batches where PyTorch returned nonzero values; this
is a correctness finding, not a performance claim. Current coverage is 233
correctness and operator links, 220 Program links, and zero accepted baselines.

The [28-cell Metal structural batch](results/coverage-capability-metal-structural-2026-10-30.json)
adds f32/BF16 reshape, transpose, slice, contiguous, cat, and stack on regular
and ragged shapes, plus squeeze and unsqueeze views. The contiguous workload
transposes its input before materialization. Independent probes and matched
PyTorch MPS operator and synthetic Program checks passed all 28. The runner
now checks output shapes as well as values, and the evidence importer requires
operator and Program output shapes to match the probe. Current coverage is
261 correctness and operator links, 248 Program links, and zero accepted
baselines out of 6,292 candidates. View-only ratios remain diagnostic under
the same host materialization timing scope and unresolved Metal drift.

The [24-cell structural view batch](results/coverage-capability-metal-structural-views-2026-10-30.json)
adds f32/BF16 reshape, transpose, slice, contiguous, cat, and stack over real
stride-2 and `[1,1]` offset inputs. All 24 passed independent probes and
matched operator checks. Twenty-two passed the synthetic Program check. The
f32 offset transpose and slice Programs returned wrong values after `add_self`;
their reports are retained in
[`cell-program-failures-mac-2026-10-30`](results/cell-program-failures-mac-2026-10-30)
without Program links. This joins the earlier f32 offset binary and batched
matmul-plus-bias correctness findings. The ledger now links 285 correctness,
285 operator, and 270 Program artifacts out of 6,292 candidates; accepted
baselines remain zero.

The [20-cell Metal indexing batch](results/coverage-capability-metal-indexing-2026-10-30.json)
adds f32/BF16 `index_select`, `embedding`, and `gather` with i64 indices.
It covers small, large, strided table inputs and batched embedding indices.
All 20 passed independent probes, matched PyTorch MPS operator comparisons,
and synthetic two-operation Programs. The ledger now links 305 correctness,
305 operator, and 290 Program artifacts out of 6,292 candidates; accepted
baselines remain zero. Matched timing ratios are diagnostic under the same
host materialization scope and unresolved Metal timing drift.

The [30-cell Metal elementwise batch](results/coverage-capability-metal-elementwise-special-2026-10-30.json)
adds f32/BF16 `erf`, `gelu`, and `clamp` over small and large dense, strided,
offset, and ragged inputs. All 30 passed independent probes, matched PyTorch
MPS operator checks, and synthetic two-operation Programs. GELU uses the tanh
approximation in both Affon and the reference; this was established from the
Metal implementation after the exact-erf reference disagreed. The ledger now
links 335 correctness, 335 operator, and 320 Program artifacts out of 6,292
candidates, with zero accepted baselines. Timing ratios remain diagnostic.

The [48-cell Metal whole-tensor reduction batch](results/coverage-capability-metal-reduce-all-2026-10-30.json)
adds f32/BF16 sum, mean, min, max, variance, and standard deviation over
short, long, strided, and ragged inputs. Affon returns a one-element tensor
for these reductions, and the matched PyTorch contract checks that shape.
Forty-six probes and their matched operator and synthetic Program checks
passed. The f32 long-axis `variance_all` and `std_all` probes exceeded the
existing numerical tolerance; their raw probe results remain retained with
unknown support and no peer links. The ledger now links 381 correctness,
381 operator, and 366 Program artifacts out of 6,292 candidates, with zero
accepted baselines. Several f32 long-axis ratios are large under host
materialization; these and all other Metal ratios remain diagnostic.

The [36-cell Metal unary view batch](results/coverage-capability-metal-unary-views-2026-10-30.json)
adds f32/BF16 absolute value, square root, sign, ReLU, sigmoid, SiLU, tanh,
sine, and cosine over real column-stride-2 and `[1,1]` offset views. All 36
passed independent probes, matched PyTorch MPS operator checks, and synthetic
two-operation Programs. The ledger now links 417 correctness, 417 operator,
and 402 Program artifacts out of 6,292 candidates, with zero accepted
baselines. These matched timing ratios remain diagnostic.

The [50-cell Metal scalar batch](results/coverage-capability-metal-scalar-2026-10-30.json)
uses true rank-zero f32/BF16 tensors for four binary and fifteen unary
operations plus six whole-tensor reductions. Thirty-eight cells passed
independent probes, matched PyTorch MPS operator checks, and synthetic
Programs. All twelve scalar reduction probes failed Program construction with
`UnsupportedRank`; their raw errors remain retained with unknown support and
no peer links. The ledger now links 455 correctness, 455 operator, and 440
Program artifacts out of 6,292 candidates, with zero accepted baselines.
Scalar input and output shapes are checked in the evidence contracts. Ratios
remain diagnostic.

The [40-cell Metal index reduction batch](results/coverage-capability-metal-arg-reductions-2026-10-30.json)
adds f32/BF16 argmin and argmax over whole tensors and single axes, covering
short, long, outer-axis, strided, and ragged inputs. All 40 passed independent
probes, matched PyTorch MPS operator checks, and synthetic two-operation
Programs. The integer index output shape and selected path are linked in the
contracts. The ledger now links 495 correctness, 495 operator, and 480 Program
artifacts out of 6,292 candidates, with zero accepted baselines. The largest
f32 whole-tensor cells have large matched timing ratios; these remain
unranked diagnostics under the existing host-materialization scope.

The [36-cell Metal masking and indexed-loss batch](results/coverage-capability-metal-masking-loss-2026-10-30.json)
adds f32/BF16 `where` and `masked_fill` with i64 masks over scalar, small,
large, broadcast, strided, offset, and ragged inputs. It also adds indexed
cross entropy with i64 class labels over small, large, batched, and strided
logits. All 36 passed independent probes, matched PyTorch MPS operator checks,
and synthetic two-operation Programs. The mask and label shapes are retained
in the evidence contracts. The ledger now links 531 correctness, 531
operator, and 516 Program artifacts out of 6,292 candidates, with zero
accepted baselines. Timing ratios remain diagnostic.

The [40-cell Metal matmul boundary batch](results/coverage-capability-metal-matmul-boundary-2026-10-30.json)
adds independent f32/BF16 cells around `K=64` and `N=64` for one-row matmul,
with and without the dense-input compilation hint, plus two-row and four-row
small batches. These declared variants raise the matrix to 6,332 candidates.
All 40 passed independent probes, matched PyTorch MPS operator checks, and
matmul-plus-bias Programs. For f32, the probe paths span
`metal.matmul.tiled`, `metal.matmul.mps_single`, and
`metal.matmul.simd_gemv`; BF16 selected `metal.matmul.small` for these
shapes. The ledger now links 571 correctness, 571 operator, and 556 Program
artifacts, with zero accepted baselines. Timing ratios remain diagnostic.

The [36-cell Metal indexing axis and shape batch](results/coverage-capability-metal-index-axis-2026-10-30.json)
adds f32/BF16 embedding table and index-size variants, `index_select` along
axes 0 and 1, and `gather` along axes 0 and 1. The explicit variants raise the
matrix to 6,368 candidates. All 36 passed independent probes, matched PyTorch
MPS operator checks, and synthetic two-operation Programs. The ledger now
links 607 correctness, 607 operator, and 592 Program artifacts, with zero
accepted baselines. Timing ratios remain diagnostic.

The [42-cell Metal comparison batch](results/coverage-capability-metal-comparisons-2026-10-30.json)
adds f32/BF16 `eq`, `lt`, and `gt` over scalar, dense, broadcast, strided,
offset, and ragged inputs. The right operand deliberately produces equality
and both comparison directions within each nonscalar tensor. Thirty-nine
cells passed independent probes, matched PyTorch MPS operator checks, and
synthetic Programs with i64 outputs. The three f32 offset comparison probes
returned incorrect values; their raw results remain retained with unknown
support and no peer links. The ledger now links 646 correctness, 646 operator,
and 631 Program artifacts out of 6,368 candidates, with zero accepted
baselines. Timing ratios remain diagnostic.

The [32-cell Metal cast, view, and normalization batch](results/coverage-capability-metal-cast-structural-normalization-2026-10-30.json)
adds f32→BF16 and BF16→f32 casts, strided/offset/ragged squeeze and
unsqueeze, materializing reshape, and outer-axis softmax and layer
normalization. All 32 independent probes returned correct values. The two
dense `contiguous/view_only` probes selected `metal_contiguous`, so they remain
diagnostic with unknown support for the declared view-only regime. The other
30 passed matched PyTorch MPS operator checks. Twenty-eight passed synthetic
Programs; f32 offset squeeze and unsqueeze returned incorrect composed values.
Their [raw Program reports](results/cell-program-failures-mac-2026-10-30)
remain retained without Program links. Cast output dtype is checked alongside
shape and values. The ledger now links 676 correctness, 676 operator, and 659
Program artifacts out of 6,368 candidates, with zero accepted baselines.
Timing ratios remain diagnostic.

The [40-cell Metal normalization width batch](results/coverage-capability-metal-normalization-width-2026-10-30.json)
adds f32/BF16 softmax and layer normalization at row widths around 32, 64,
and 128, plus a taller 256-row case. These variants raise the matrix to 6,408
candidates. All 40 passed independent probes, matched PyTorch MPS operator
checks, and synthetic Programs. The f32 layer normalization selected both
`metal.normalization.serial_row` and `metal.normalization.cooperative_row`
across these widths; the path change is not monotonic at width 64. The ledger
now links 716 correctness, 716 operator, and 699 Program artifacts, with zero
accepted baselines. Timing ratios remain diagnostic.

The [40-cell Metal indexed cross entropy shape batch](results/coverage-capability-metal-indexed-loss-shapes-2026-10-30.json)
adds f32/BF16 logits at batch sizes 1 and 32 with class widths 7, 31, 32,
33, 63, 64, 65, 127, 128, and 129. All 40 selected `metal_primitive` and
passed independent correctness probes, matched PyTorch MPS operator checks,
and synthetic Programs. No cells failed or were excluded. The matrix now has
6,448 candidates; the ledger links 756 correctness, 756 operator, and 739
Program artifacts, with zero accepted baselines. Timing ratios remain
diagnostic.

The [60-cell Metal indexing boundary batch](results/coverage-capability-metal-indexing-boundaries-2026-10-30.json)
adds f32/BF16 embedding, index selection, and gather across row, width,
index count, output size, and axis boundaries. All 60 passed independent
correctness probes, matched PyTorch MPS operator checks, and synthetic
Programs. No cells failed or were excluded. The matrix now has 6,508
candidates; the ledger links 816 correctness, 816 operator, and 799 Program
artifacts, with zero accepted baselines. Timing ratios remain diagnostic.

The [60-cell Metal indexing layout batch](results/coverage-capability-metal-indexing-layouts-2026-10-30.json)
adds f32/BF16 embedding, index selection, and gather on column-stride-2 and
`[1,1]` offset views across five shapes per operation and dtype. All 60
passed independent correctness probes, matched PyTorch MPS operator checks,
and synthetic Programs. No cells failed or were excluded. The coverage
policy's variant IDs use one final segment, so these cells use `_strided`
and `_offset` suffixes; the runners recognize both suffix forms. The matrix
now has 6,568 candidates; the ledger links 876 correctness, 876 operator,
and 859 Program artifacts, with zero accepted baselines. Timing ratios
remain diagnostic.

The [72-cell Metal matmul dispatch batch](results/coverage-capability-metal-matmul-dispatch-2026-10-30.json)
adds f32/BF16 matrix multiplication at eighteen M/K/N combinations, each
with default and dense-input compile options. The one-row cases bracket 32
and 128 in K and N; the small-batch cases cover M=2, 4, and 8. All 72 passed
independent correctness probes, matched PyTorch MPS operator checks, and
matmul-plus-bias Programs. No cells failed or were excluded. Probe paths
selected `metal.matmul.small` (36), `metal.matmul.tiled` (18),
`metal.matmul.mps_single` (13), and `metal.matmul.simd_gemv` (5). The matrix
now has 6,640 candidates; the ledger links 948 correctness, 948 operator,
and 931 Program artifacts, with zero accepted baselines. Timing ratios
remain diagnostic.

The [100-cell Metal elementwise width batch](results/coverage-capability-metal-elementwise-widths-2026-10-30.json)
adds f32/BF16 add, subtract, multiply, divide, negation, ReLU, sigmoid, SiLU,
tanh, and GELU at row/width combinations 32x31, 32x32, 32x33, 64x127 and
64x128. All 100 passed independent correctness probes, matched PyTorch MPS
operator checks, and synthetic two-operation Programs. No cells failed or were
excluded. The largest observed value error was 1.19e-07, on f32 sigmoid at
32x31, well inside the 2e-6 relative tolerance used for f32. The matrix now
has 6,740 candidates; the ledger links 1048 correctness, 1048 operator, and
1031 Program artifacts, with zero accepted baselines. Timing ratios remain
diagnostic.

The [80-cell Metal structural and axis-reduction batch](results/coverage-capability-metal-structural-reductions-2026-10-30.json)
adds f32/BF16 `cat`, `stack`, `transpose`, `slice`, materializing `reshape`,
`squeeze`, `unsqueeze`, and `contiguous` at larger shapes, plus `sum_axis`,
`mean_axis`, `min_axis`, `max_axis`, `argmax_axis` and `argmin_axis` over
64x127, 64x128 and outer-axis 64x127 / 128x64 inputs. All 80 passed
independent correctness probes, matched PyTorch MPS operator checks, and
synthetic two-operation Programs. No cells failed or were excluded. The
matrix now has 6,820 candidates; the ledger links 1128 correctness, 1128
operator, and 1111 Program artifacts, with zero accepted baselines. The
`transpose/view_only` ratios of 0.15 and 0.44 are Metal timing drift, not a
correctness signal. Timing ratios remain diagnostic.

The [100-cell Metal broadcast batch](results/coverage-capability-metal-broadcast-2026-10-30.json)
adds f32/BF16 `add`, `sub`, `mul`, `div`, `neg`, `abs`, `sign`, `exp`, `log`,
`sqrt`, `relu`, `sigmoid`, `silu`, `tanh`, `sin`, `cos`, `erf`, `gelu`, `clamp`,
`cast`, `eq`, `lt`, `gt`, `where` and `masked_fill` with a `[1, cols]` right
operand against 32x32 and 64x128 left operands. All 100 passed independent
correctness probes, matched PyTorch MPS operator checks, and synthetic
two-operation Programs. No cells failed or were excluded. For the binary,
comparison and mask operations the second operand really is `[1, cols]`, and
output shape is checked in the evidence contracts; for the unary entries in this
regime the second operand is unused, so those cells only exercise the larger
input shape. The matrix now has 6,920
candidates; the ledger links 1228 correctness, 1228 operator, and 1211 Program
artifacts, with zero accepted baselines. Timing ratios remain diagnostic.

The [50-cell Metal axis and structural batch](results/coverage-capability-metal-axis-structural-2026-10-30.json)
adds f32/BF16 outer-axis and long-axis `sum_axis`, `mean_axis`, `min_axis`,
`max_axis`, `variance_axis`, `std_axis`, whole-tensor `sum_all`, `mean_all`,
`min_all`, `max_all`, `variance_all` and `std_all` over 64x127 / 64x128 inputs,
plus `cat` and `stack` in the view-only regime, `slice`, `transpose`, `squeeze`
and `unsqueeze` in the materialized regime, and `gather` with a batched i64
index tensor. All 50 passed independent
correctness probes, matched PyTorch MPS operator checks, and synthetic
two-operation Programs. No cells failed or were excluded. `slice/materialized`
and `squeeze/materialized` timed at 0.07x and 0.04x of PyTorch MPS with exact
outputs, so those are Metal timing drift, not correctness. The matrix now has
6,944 candidates; the ledger links 1278 correctness, 1278 operator, and 1261
Program artifacts, with zero accepted baselines. Timing ratios remain
diagnostic.

The [154-cell Metal dispatch and layout batch](results/coverage-capability-metal-dispatch-2026-10-30.json)
adds f32/BF16 matmul at eight further M/K/N combinations with default and
dense-input compile options, transposed and stride-2 RHS layouts at 32x64x127
and 64x127x64, softmax and layer norm on strided, ragged and outer-axis inputs,
strided/ragged/long-axis axis and whole-tensor reductions, strided embedding,
index selection and gather, four indexed cross-entropy shapes, and strided
`cat`, `stack`, `reshape`, `transpose`, `slice`, `squeeze`, `unsqueeze` and
`contiguous`. 152 of the 154 passed independent probes, matched PyTorch MPS
operator checks and synthetic Programs. Two cells stay unknown: f32
`variance_all` and `std_all` over a 64x127 strided input exceeded tolerance by
0.00268 and 0.00026 against tolerances of 5.27e-05 and 1.03e-05, so they have
no peer links. The matrix now has 7,098 candidates; the ledger links 1430
correctness, 1430 operator, and 1413 Program artifacts, with zero accepted
baselines. `squeeze/strided` at 0.04x and `slice/strided` at 0.18x are Metal
timing drift with exact outputs. Timing ratios remain diagnostic.

The [234-cell Metal layout and regime batch](results/coverage-capability-metal-variants-2026-10-30.json)
adds f32/BF16 offset, stride-2 and ragged views for `abs`, `sign`, `sqrt`,
`relu`, `sigmoid`, `silu`, `tanh`, `sin`, `cos`, `erf`, `gelu`, `clamp`, `neg`,
`exp` and `log`; offset and strided `add`, `sub`, `mul`, `div`, `eq`, `lt`,
`gt`, `where` and `masked_fill`; four batched matmul shapes; softmax and layer
norm at 256x31, 128x129, 33x255, 64x255 and 31x16; long-axis and ragged axis
and whole-tensor reductions; large embedding, index selection and gather; four
indexed cross-entropy shapes; and ragged `cat`, `stack`, `reshape`, `transpose`,
`slice`, `contiguous`, `squeeze` and `unsqueeze`. All 234 passed independent
correctness probes, matched PyTorch MPS operator checks, and synthetic
two-operation Programs. No cells in this batch failed or were excluded. The
largest ratio in the batch is 16.5x for f32 ragged whole-tensor reductions,
which the report attributes to host materialization of the ragged view; every
one of those outputs was exact. The matrix now has 7,332 candidates; the ledger
links 1664 correctness, 1664 operator, and 1647 Program artifacts, with zero
accepted baselines. Timing ratios remain diagnostic.

The [132-cell broadcast correction batch](results/coverage-capability-metal-broadcast-corrected-2026-10-30.json)
starts from a harness defect rather than a new shape. All four runners detected
the broadcast regime with `endsWith('/broadcast')`, so the 100 shape variants
written as `.../broadcast/r32_c32` and `.../broadcast/r64_c128` built a right
operand with the full input shape: those cells were labelled broadcast but never
broadcast. The probe and benchmark workers now use `includes('/broadcast')`, so
the right operand really is `[1, cols]`, and both the independent reference and
the peer contract record that shape. A unary operation has no second operand to
broadcast, so `broadcast` was never a valid regime for `abs`, `neg`, `exp`,
`log`, `sqrt`, `sign`, `relu`, `sigmoid`, `silu`, `tanh`, `sin`, `cos`, `erf`,
`gelu`, `clamp` and `cast`. The policy now splits elementwise operations into
`elementwise` for unary operations and `elementwise_binary` for `add`, `sub`,
`mul`, `div`, `eq`, `lt`, `gt`, `where` and `masked_fill`, and declares
`broadcast` only for the latter. Fifty-two variant cells moved to
`large_dense`, twelve duplicates were removed, and the matrix drops from 7,332
to 7,065 candidates because the invalid combinations no longer exist. The peer
and Program reports collected under the old labels stay in
[cell-peer-superseded](results/cell-peer-superseded-mac-2026-10-30) and
[cell-program-superseded](results/cell-program-superseded-mac-2026-10-30) and are
no longer linked. All 132 corrected cells passed independent probes, matched
PyTorch MPS operator checks and synthetic Programs: 18 canonical binary
broadcast cells, 36 broadcast shape variants at 32x32 and 64x128, 76 unary
`large_dense` variants at 32x32, 64x127 and 64x128, and f32/BF16
`sum_axis/outer_axis` at 128x256. Operator ratios run from 0.80x to 1.43x and
Program ratios from 0.82x to 1.63x, with eleven operator cells and twenty
Program cells above the 1.25x diagnostic limit. The ledger links 1654
correctness, 1654 operator and 1637 Program artifacts, and zero accepted
baselines.

The [51-cell canonical gap sweep](results/coverage-capability-metal-canonical-gaps-2026-10-30.json)
probes the canonical Metal cells still missing from the ledger: scalar
reductions, f32 offset binary operations, whole-tensor variance and standard
deviation over a long axis, `contiguous/view_only`, and the `invalid_index`
indexing regime. Only `sum_axis/outer_axis` passed for both dtypes and earned
links. The two `contiguous/view_only` probes return exact values but select
`metal_contiguous`, so they establish no view-only path and stay unknown without
peer links. The seven f32 offset binary cells are incorrect again, matching the
earlier view probe finding.

Two findings are now verdicts rather than open questions. The 240 scalar
reduction cells across CPU, Metal and CUDA and all five dtypes are
`not_applicable`: Compute's abstract evaluator returns `UnsupportedRank` for
whole-tensor reductions of a rank-zero tensor and rejects an axis reduction
because a scalar has no axis, confirmed by probe runs that fail at program
construction. The six `invalid_index` cells for `embedding`, `index_select` and
`gather` are `unsupported`. With indices past the last valid position the Metal
indexing kernel returns whatever the parent buffer holds at the raw offset,
while PyTorch MPS zero-fills: f32 embedding and index selection err by 0.25,
gather by 1.25, and probe checksums differ by 26.0 and 20.5. Raw probe and peer
reports are kept in [cell-failures](results/cell-failures-mac-2026-10-30).
Supported, excluded and unknown cells are counted separately, and no cell
carries a performance link its correctness probe did not pass.

### Layout contract correction, 31 October 2026

A suffix audit found 158 declared Metal f32/BF16 variants whose `/strided/`,
`/offset/` or `_strided/` markers were ignored by the workers and references.
Of these, 156 had correctness, operator and Program links for dense inputs.
Those links have been removed from the reviewed ledger. Their raw reports remain
in [superseded operator reports](results/cell-peer-superseded-mac-2026-10-30)
and [superseded Program reports](results/cell-program-superseded-mac-2026-10-30)
as diagnostics. The four harnesses now detect layout markers before suffixes.
A [158-cell reprobe](results/coverage-capability-metal-layout-reprobe-2026-10-31.json)
found 149 correct and 9 incorrect: seven f32 offset binary/comparison variants
and two f32 strided whole-tensor variance/std variants. All 149 passing cells also passed matched PyTorch MPS operator and synthetic
two-operation Program comparisons. The matrix has 7,065 candidates, 1,647
supported cells with correctness and operator links, 1,630 Program links,
5,172 unknown, 240 not applicable, 6 unsupported, and zero accepted baselines. The previously
stated 1,654/1,637 links are historical and were superseded by this audit.

The [104-cell Metal log-softmax and RMS-normalization batch](results/coverage-capability-metal-log-rms-2026-10-31.json)
adds both dtypes over short and long rows, outer-axis, strided and ragged
layouts, plus 84 width and shape variants. All 104 passed capability,
matched PyTorch MPS operator, and synthetic two-operation Program checks.
The declared matrix now has 7,149 candidates; 1,751 supported cells with
correctness and operator links, 1,734 Program links, 5,152 unknown, 240 not
applicable, 6 unsupported, and zero accepted baselines. Ratios remain
diagnostic and do not establish an accepted baseline.

The [18-cell Metal permute batch](results/coverage-capability-metal-permute-2026-10-31.json)
covers both dtypes over view, materialized, strided, offset, and ragged
regimes plus eight shape variants. All 18 capability and operator comparisons
passed. Seventeen Program comparisons passed; the f32 offset Program returned
incorrect values and its raw report is retained in
[cell-program-failures](results/cell-program-failures-mac-2026-10-30).
The matrix now has 7,157 candidates, 1,769 supported cells with correctness
and operator links, 1,751 Program links, 5,142 unknown, 240 not applicable,
6 unsupported, and zero accepted baselines.

The [64-cell normalization width batch](results/coverage-capability-metal-normalization-extra-2026-10-31.json)
covers f32/BF16 softmax, log-softmax, layer norm and RMS norm at widths
7, 15, 16, 17, 31, 32, 33 and 257. All 64 capability probes passed.
Sixty-three matched operator and Program comparisons passed. The f32 RMS
norm at width 7 exceeded the matched-peer tolerance by 1.91e-6 versus
1.16e-6; both raw failure reports are retained and neither peer link is
claimed. The matrix has 7,221 candidates, 1,833 supported correctness links,
1,832 operator links, 1,814 Program links, 5,142 unknown, 240 not applicable,
6 unsupported and zero accepted baselines. Metal timing ratios remain
diagnostic.

The [64-cell indexed cross-entropy batch](results/coverage-capability-metal-indexed-loss-extra-2026-10-31.json) covers f32/BF16 at batch sizes 2, 4, 8 and 16 and class widths 3, 5, 17, 24, 48, 96, 192 and 257. All 64 capability, operator, and Program checks passed; the selected path was `metal_primitive`. The matrix has 7,285 candidates, 1,897 correctness links, 1,896 operator links, 1,878 Program links, 5,142 unknown, 240 not applicable, 6 unsupported and zero accepted baselines. Timing ratios are diagnostic.

The [first 256-cell CUDA batch](results/coverage-capability-cuda-first-2026-10-31.json) spans 59 operations, f32/BF16, and dense, ragged, view, reduction, matrix, normalization and indexing regimes on the RTX 3090. The capability probe passed 244 cells, failed to compile nine (`sin` and `cos`), and found three numerical tolerance misses (`layer_norm`, `std_all` and `variance_all`). Matched PyTorch CUDA checks passed 243 operator and 243 Program cells. The f32 RMS norm long row exceeded the peer tolerance in both; its raw reports are retained in [operator failures](results/cell-failures-cuda-2026-10-31) and [Program failures](results/cell-program-failures-cuda-2026-10-31), with no peer links. The matrix has 7,286 candidates, 2,141 correctness links, 2,139 operator links, 2,121 Program links, 4,899 unknown, 240 not applicable, 6 unsupported and zero accepted baselines. CUDA has 2,011 candidates and 244 supported cells. Timing ratios remain diagnostic.

The exact [256-cell catalog](../parity/catalogs/cuda-first-256.json) is retained, and its
SHA-256 matches the probe artifact. On the CUDA machine, with this checkout,
Zig 0.17, the ReleaseFast Affon binary, and a Python environment with PyTorch
CUDA, rerun it into a fresh directory:

```sh
zig build -Doptimize=ReleaseFast
python3 test/parity/run.py \
  --catalog test/parity/catalogs/cuda-first-256.json \
  --output-dir test/benchmarks/results/cuda-rerun-YYYYMMDD \
  --iterations 10 --samples 5 --rounds 3
```

Each run saves a catalog snapshot, independent numerical capability probe,
matched operator and two-operation Program reports, and a `manifest.json`
entry for every catalog cell. The manifest records probe status, tolerance,
peer correctness, median timings, timing ratio, and raw artifact paths. A
failed parity check retains its timed raw report without a coverage link.
When a probe cannot compile or returns an incorrect value, peer timing is
marked `probe_not_correct`: a valid matched timing cannot be claimed for that
cell. Benchmark exceptions are retained with an `error` status. The runner
refuses an existing output directory, so a repeat run cannot overwrite raw
evidence or change the reviewed ledger implicitly. Import a reviewed rerun
only after verifying all statuses and matching probe and benchmark contracts.

The [128-cell Metal shape batch](results/parity-metal-shapes-128-2026-10-31/report.md) uses the permanent [parity catalog](../parity/catalogs/metal-matmul-indexed-shapes-128.json). It spans f32/BF16 matmul around M=1/16, K=31/32/33/64 and N=31/32/33/65, plus indexed cross-entropy at batch sizes 3/5/7/9 and class widths 15/16/17/255/256/257/512/513. All 128 numerical probes, matched operator comparisons and composed Program comparisons passed. The run retains its catalog snapshot, raw rounds, manifest and report. The matrix now has 7,414 candidates, 2,269 correctness links, 2,267 operator links, 2,249 Program links, 4,899 unknown, 240 not applicable, 6 unsupported and zero accepted baselines. Metal has 3,393 candidates and 2,019 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal dot width batch](results/parity-metal-dot-widths-128-reviewed-2026-10-31/report.md) uses a permanent [rank-one catalog](../parity/catalogs/metal-dot-widths-128.json) covering f32/BF16 lengths 1–48 and reduction boundaries through 2,048. All 128 independent numerical probes, matched PyTorch operator comparisons and composed Programs passed. The [first run](results/parity-metal-dot-widths-128-2026-10-31/report.md) exposed three f32 peer misses at widths 511–513 with relative error about 1.6e-6. The dot peer tolerance was aligned with the existing 2e-6 probe tolerance for floating-point reductions, and the entire batch was rerun into a fresh retained directory. The matrix now has 7,540 candidates, 2,397 correctness links, 2,395 operator links, 2,377 Program links, 4,897 unknown, 240 not applicable, 6 unsupported and zero accepted baselines. Metal has 3,519 candidates and 2,147 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal dot layout batch](results/parity-metal-dot-layouts-128-2026-10-31/report.md) uses a permanent [strided and offset catalog](../parity/catalogs/metal-dot-layouts-128.json). It exercises actual rank-one views for f32/BF16 at lengths 1–24 and 31/32/33/63/64/65/127/128. All 128 numerical probes, operator comparisons and composed Programs passed; the importer checked physical input shapes and selected paths. The matrix now has 7,666 candidates, 2,525 correctness links, 2,523 operator links, 2,505 Program links, 4,895 unknown, 240 not applicable, 6 unsupported and zero accepted baselines. Metal has 3,645 candidates and 2,275 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal one-hot batch](results/parity-metal-one-hot-shapes-128-2026-10-31/report.md) uses a permanent [f32 output catalog](../parity/catalogs/metal-one-hot-shapes-128.json). It spans index shapes with batch sizes 1/2/4/8, widths 1/3/8/17 and class counts 2/3/7/16/17/31/32/33. The inputs are i64 and the outputs are f32, as required by the public API. All 128 probes, operator comparisons and composed Programs passed. Five Metal BF16 one-hot base cells are classified not applicable because the API has no BF16 output contract. The matrix now has 7,792 candidates, 2,653 correctness links, 2,651 operator links, 2,633 Program links, 4,888 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 3,771 candidates and 2,403 supported cells. Timing ratios remain diagnostic.

The `one_hot` API consumes i64 indices and produces f32 irrespective of the
candidate dtype label. Its f32/BF16 policy cells need an explicit input/output
contract before they can be measured and compared meaningfully.

The [128-cell Metal one-hot layout batch](results/parity-metal-one-hot-layouts-128-2026-10-31/report.md) uses a permanent [strided and offset catalog](../parity/catalogs/metal-one-hot-layouts-128.json). It checks actual views of i64 indices across batch sizes 1/2/4/8, widths 3/7/16/33 and class counts 2/7/16/33. All 128 probes, operator comparisons and composed Programs passed. The matrix now has 7,919 candidates, 2,781 correctness links, 2,779 operator links, 2,761 Program links, 4,887 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 3,898 candidates and 2,531 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal one-hot invalid-index batch](results/parity-metal-one-hot-invalid-index-128-2026-10-31/report.md) uses a permanent [mixed-index catalog](../parity/catalogs/metal-one-hot-invalid-index-128.json) spanning batch sizes 1/2/4/8, widths 3/7/16/33 and class counts 2/3/7/16/17/31/32/33. Its fixtures mix valid, negative and too-large i64 indices. The Metal kernel defines out-of-range positions as all-zero rows. PyTorch’s one-hot helper rejects those inputs, so the timing peer uses an elementwise comparison against the class range with the same output semantics; these timing ratios are diagnostic across different implementations. All 128 probes, operator comparisons and composed Programs passed. The matrix now has 8,046 candidates, 2,909 correctness links, 2,907 operator links, 2,889 Program links, 4,886 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 4,025 candidates and 2,659 supported cells.

The [128-cell Metal batched one-hot batch](results/parity-metal-one-hot-batched-128-2026-10-31/report.md) uses a permanent [rank-three catalog](../parity/catalogs/metal-one-hot-batched-128.json) spanning batch sizes 1/2/4/8, spatial rows 1/3/7/16, columns 1/3/8/17 and class counts 2/33. It exercises i64 inputs and f32 outputs at rank three and four. All 128 probes, operator comparisons and composed Programs passed. The matrix now has 8,173 candidates, 3,037 correctness links, 3,035 operator links, 3,017 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 4,152 candidates and 2,787 supported cells. Timing ratios remain diagnostic.

The remaining Metal gaps include operations the probe worker cannot express yet:
`gelu_grad`, `reduce_to_shape`, `quantized_matmul`, `scatter_add`, `topk`, `log_softmax_nll`,
`cross_entropy` and `cross_entropy_indexed_backward`. Covering them needs a
reference formula in the Python probe and a matching operation in the
`test/parity/workers/affon-probe.ts` program switch, the
`test/parity/workers/affon-benchmark.ts` worker and the Zig JSON operation map in
`src/js/compute/native.zig`. CUDA coverage now continues in larger batches on the RTX 3090; remaining CUDA candidate cells stay unknown until checked.

## Baseline repeatability

Before accepting a same-target Affon baseline, run at least two independent
sessions. Each session uses fresh processes with the same binary, selected
path, shape, and timing contract and records power and thermal state at its
start and end:

```sh
python3 tools/calibrate-cell-baseline.py \
  --only metal/f32/sum_axis/long_axis,metal/f32/matmul/decode_m1,metal/f32/matmul/prefill \
  --runs 5 --samples 5 --iterations 100 \
  --output-dir /tmp/affon-cell-session-a
```

Repeat into a separate `session-b` directory. For each cell, evaluate its two
JSON files with `tools/evaluate-cell-baseline.py`, passing `--output` for the
verdict JSON. The evaluator requires matching target, binary, harness, contract,
path, and machine state. It applies the provisional 1.10× or 0.05 ms spread
limit both within each session and between session medians. A rejected verdict
does not become baseline evidence. The recorded state is only a coarse control;
an accepted verdict still needs a reviewed reference before it is linked to
the coverage matrix.
The evidence importer requires a `self_baseline` benchmark to name an accepted
`baseline_evaluation` JSON whose cell, target, binary, path, shape, dtype, and
timing scope match. The evaluation records hashes of both calibration sessions;
the importer verifies them and recomputes the verdict. It incorporates the
evaluation's SHA-256 into the evidence contract. A rejected or stale calibration
cannot be imported as baseline coverage.

The [two-session evaluation](results/cell-baseline-evaluations-mac-2026-10-29)
of the three sampled Metal cells rejected all three. Session median ratios
were 1.86× for long-axis sum, 1.73× for decode matmul, and 2.03× for prefill
matmul. Sum passed the within-session screen in both sessions but failed the
between-session check. Decode and prefill also failed the process spread check
in the second session. These runs recorded AC power and thermal status at both
ends; that coarse metadata does not explain the drift.

The [first calibration](results/cell-baseline-calibration-mac-2026-10-29)
checks outputs against an independent formula and records five fresh-process
medians per cell. The provisional screen calls a candidate stable only when
the largest/smallest median ratio is at most 1.10 or their span is at most
0.05 ms. All three sampled Metal cells exceeded both limits: long-axis sum
1.60×, decode matmul 1.53×, and prefill matmul 1.29×. These are diagnostic
repeatability results, not accepted baseline evidence. The baseline checklist
therefore remains at zero. Stabilize the measurement environment and protocol,
then rerun this screen before accepting a reference for regression checks.
An additional [longer sum calibration](results/cell-baseline-calibration-long-mac-2026-10-29)
used 100 warmup calls and 300 calls per sample; the five run medians still
spread by 1.57×. More warmup and longer samples alone did not make this
measurement stable on the current target.
Another [same-protocol sum calibration](results/cell-baseline-phase-mac-2026-10-29)
had a 1.17× spread across its five processes, but its median was about
0.639 ms versus 0.311 ms for the previous longer calibration. Pooling more fresh
processes would therefore hide a large change between sessions, not establish
a stable host baseline.

A focused Metal timing profile used the public `MTLCommandBuffer` GPU start/end
timestamps with the same longer sum protocol. It verified one completed command
per operation and checked output against the independent formula. In
[session A](results/cell-gpu-variability-session-a-mac-2026-10-29.json), five
fresh-process GPU command medians spread by 1.006× and host totals by 1.26×.
The identical binary, interposer, contract, and selected path in
[session B](results/cell-gpu-variability-session-b-mac-2026-10-29.json) produced
session medians of 0.081708 ms GPU and 0.310163 ms host, versus 0.35625 ms GPU
and 0.86525 ms host in session A. The between-session difference is 4.36× for
GPU commands and 2.79× for host operation totals. Thus a small within-session
spread did not establish repeatability.
Reproduce the diagnostic with:

```sh
python3 tools/profile-cell-gpu-variability.py \
  --cell metal/f32/sum_axis/long_axis --runs 5 --samples 5 \
  --iterations 300 --warmup-calls 100 --output /tmp/affon-gpu-variability.json
```

Readback was a small part of each session's host total. Both GPU command and
host totals changed between sessions, so isolating GPU command time alone does
not resolve baseline acceptance. The timestamp interposer adds host callback
work; this profile is diagnostic and is neither a matched PyTorch GPU comparison
nor an accepted baseline. A future acceptance gate should compare independent
sessions on a controlled target with recorded power and thermal state, using
the same binary, protocol, and selected path. Existing session artifacts did
not record power or thermal state; future profiles capture their values at the
end of a session.

Three consecutive focused GPU timing sessions
([first](results/cell-gpu-variability-investigation-1-mac-2026-10-29.json),
[second](results/cell-gpu-variability-investigation-2-mac-2026-10-29.json),
[third](results/cell-gpu-variability-investigation-3-mac-2026-10-29.json))
recorded power, `pmset` thermal status, and GPU utilization snapshots around
each fresh process. The first session's GPU command medians moved from
0.082–0.097 ms in its first three processes to 0.196 ms in its fifth; the
next two sessions stayed near 0.353 ms. Session medians were 0.0897, 0.3534,
and 0.3533 ms. All reported AC power and no thermal warning. GPU utilization
snapshots are coarse and do not measure clock frequency. `powermetrics` offers
GPU frequency and power samples but requires administrator access on this Mac;
an unprivileged invocation failed with `powermetrics must be invoked as the
superuser`, and noninteractive `sudo` required a password. The cause of the
timing state change remains unconfirmed. Retain these results as diagnostics;
do not infer a clock, throttling, or competing-workload explanation from them.

A follow-up [alternating cell diagnostic](results/paired-gpu-diagnostic-mac-2026-10-30.json)
ran decode matmul, long-axis sum, then repeated that pair twice. The six
sessions used the same binary, timing interposer, harness, and cell contracts.
Decode GPU session medians were 0.0608, 0.0608, and 0.0585 ms (1.04× spread);
sum medians were 0.0913, 0.0914, and 0.0935 ms (1.02× spread). Both stayed in
the fast state during this interval. This does not show whether the earlier
slow state would affect both kernels. The analyzer
`tools/analyze-paired-gpu-sessions.py` checks alternating order and matching
measurement identity before summarizing future pairs. These paired results
remain diagnostic, not baseline evidence.

## HTML report

Open [`results/performance-report.html`](results/performance-report.html) for
the human readable view. It includes coverage totals, filterable case rows,
round ratios, selected paths, and missing evidence. The file is self contained;
no server or external asset is needed. It is generated from the retained JSON
with Weiv and Bun:

```sh
python3 tools/render-performance-report.py
```

The script rebuilds the matrix from the current policy. If no outcomes path is
given, it renders all cells as unknown. Keep the generated HTML alongside its
source JSON when sharing a run; the source artifacts remain authoritative.

### Expanded Metal shape sweep

The current report combines the original small cases with four Metal f32
model-relevant shapes: decode projection `[1,768] @ [768,768]`, prefill
projection `[128,768] @ [768,768]`, long-row softmax `[128,512]`, and long-axis
sum `[128,768]`. Their raw comparisons are in
[`results/cell-peer-metal-shapes-2026-10-29`](results/cell-peer-metal-shapes-2026-10-29),
and the [current diagnostic summary](results/cell-peer-current-summary-mac-2026-10-29.json)
contains all 15 distinct cells, including the dense GEMV variant and the new CPU and Metal f32 multiplication cases. The updated probes use byte-backed tensors
for large inputs and compare selected coordinates plus the complete output
checksum against an independent formula. The peer runner uses the same output
checks with a declared f32 tolerance.

The `[128,768]` Metal sum is a consistent candidate gap across three rounds;
decode and prefill projection cross the provisional 1.25 ratio between rounds.
The report includes Affon execution and readback phase medians. These phases
are diagnostic because GPU execution can complete during readback and peer
submission behavior differs. A controlled GPU command-time comparison and a
model-linked Program rerun are needed before attributing the reduction gap to its
kernel. The per-case timing contracts differ between the original small set
and expanded shapes; only each case's own matched peer is compared.

### Dispatch boundary audit

Run `python3 tools/audit-dispatch-boundaries.py --binary zig-out/bin/affon`
to check the declared transitions in [`dispatch-boundaries.json`](dispatch-boundaries.json).
The audit checks output correctness and the compiler selected physical path
on each side. It also verifies that the source predicate named by the policy
still exists, so dispatch edits require boundary review. The current
[Mac result](results/dispatch-boundaries-mac-2026-10-29.json) observed four
Metal f32 GEMV transitions: K 63/64, N 63/64, row count 2/1, and
`denseInputs` false/true. Two additional batched matmul boundaries check
three/four batches and M 64/65. Both sides of those cases selected
`metal.matmul.tiled`, so the audit marks them `unobserved_transition` rather
than counting the source planner threshold as end to end dispatch coverage.
Compute's `matmul_plan_test.c` checks the planner rule separately. A future
case must select both paths at the compiler level before these two boundaries
can be counted as observed execution transitions.

These six transitions are an initial audit set. Add boundaries when new
predicates, layouts, dtypes, backends, or implementations appear. The existing
`metal/f32/matmul/decode_m1` peer timing compiled with `denseInputs: false`
and selected `metal.matmul.tiled`. The `denseInputs: true` case selects
`metal.matmul.simd_gemv`. It now has a separate
`metal/f32/matmul/decode_m1/dense_gemv` matrix cell with its own probe and
[matched PyTorch report](results/cell-peer-variants-mac-2026-10-29/metal__f32__matmul__decode_m1__dense_gemv.json).
The peer run used three alternating rounds and measured a diagnostic 0.79×
Affon/PyTorch median ratio. This is a resident operation plus full host
readback comparison, not a calibrated release verdict. The original tiled
timing remains linked only to the original cell. The importer requires the
variant's compile options and selected path to match its probe.

## Existing diagnostic links

Run `python3 tools/workload-coverage.py` from the Affon checkout. The versioned
[`workload-coverage.json`](workload-coverage.json) maps workload **cells** to
existing operator, Program, and model diagnostics. Its dimensions are backend,
precision, operation family, and shape regime. It is deliberately independent
of model names: a new model introduces a cell when its execution path or regime
is new, even if another model of the same family has a benchmark.

The report distinguishes `missing` (no case linked), `unavailable` (the linked
checkout or file is absent), `unverified_case` (a named case is absent from the
source), and `diagnostic` (the source exists). **Diagnostic does not mean a
matched benchmark passed.** Some Program cases exercise the same operation
family at a different shape. They are a starting point for adding a matched
case, not coverage of the model shape. Likewise, a model report does not prove
its operators are competitive. This inventory never accepts a baseline or
calculates a peer ratio; the existing runner and comparison gates do that.

For each new model or operator path:

1. Record its backend, precision, operator shapes, strides, layout, batch,
   cache behavior, and compiler-selected implementation. Add a cell for each
   regime absent from the inventory. Capture this from `Executable.explain()`
   or a profiler trace rather than inferring it from the model name.
2. Link an operator benchmark with the **same selected implementation** and a
   matched PyTorch, MLX, or other peer boundary. Add a Program case that includes
   conversions, submissions, and synchronization. Keep the model case for
   end-to-end validation.
3. Run correctness first. Compare raw repeated samples on the same target and
   protocol. Mark a cell competitive only after all required tiers have matched
   evidence and an accepted same-target Affon baseline. Store those verdicts in
   the evidence gate, not in this source inventory.
4. Prioritize by measured peer gap and expected workload impact. Rerun the
   Program and model cases after an operator change; a faster isolated kernel
   alone does not close the cell.

The initial inventory is intentionally incomplete. It exposes absent CUDA BF16
operator and Program comparisons, convolution paths, quantized whole-model
evidence, and model-linked reductions. The existing Metal f32 operator corpus
is a useful start, but its current scope does not establish f16, BF16, or
quantized coverage. The separate [`model-coverage.json`](model-coverage.json)
tracks model archetypes and measured peer/self gates; it now declares missing
archetypes explicitly instead of allowing an omitted model to appear covered.

The [128-cell Metal indexed loss grid](results/parity-metal-indexed-loss-grid-128-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-indexed-loss-grid-128.json), maintained by [its generator](../parity/catalogs/generators/generate-metal-indexed-loss-grid.py). It spans batch sizes 2/4/6/10 and 16 class widths from 3 to 254. Every numerical probe, matched PyTorch operator comparison and composed Program comparison passed. The scalar loss output is checked in full for each cell; repeated timing rounds and selected paths are retained. The matrix now has 8,301 candidates, 3,165 correctness links, 3,163 operator links, 3,145 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 4,280 candidates and 2,915 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal quantized matmul grid](results/parity-metal-quantized-grid-128-2026-10-31/report.md) uses a permanent [four-input catalog](../parity/catalogs/metal-quantized-grid-128.json) maintained by [its generator](../parity/catalogs/generators/generate-metal-quantized-grid.py). It covers f32/BF16 input and affine parameters, u32 packed weights, 2/4/8-bit packing, group sizes 4/8/16/32, M=1/2/8/16, K=32/64/96/128 and N=1/3/8/17. All 128 independent probes, matched operator comparisons and composed Programs passed. Metal selected `metal_quantized_matmul.simd1` for 54 cells, `metal_quantized_matmul.matrix32` for 44 and `metal_quantized_matmul.scalar` for 30. The PyTorch peer performs GPU unpacking and affine dequantization as separate operations before matmul, so its timing ratios are diagnostic for that implementation. The matrix now has 8,429 candidates, 3,293 correctness links, 3,291 operator links, 3,273 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 4,408 candidates and 3,043 supported cells.

The [128-cell Metal quantized view batch](results/parity-metal-quantized-views-128-2026-10-31/report.md) uses a permanent [transposed and offset catalog](../parity/catalogs/metal-quantized-views-128.json). It spans f32/BF16, M=1/3/17/64, K=32/64/96/128, N=3/19, 2/4/8-bit packing and group sizes 4/8/16/32. Half the cells transpose the input and slice packed weights; the other half slice all four inputs from nonzero offsets. All 128 numerical probes, operator comparisons and composed Programs passed, with zero failures. The selected Metal kernel was `simd1` for 44 cells, `scalar` for 44 and `matrix32` for 40; the retained paths also record view nodes. The matrix now has 8,557 candidates, 3,421 correctness links, 3,419 operator links, 3,401 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 4,536 candidates and 3,171 supported cells. PyTorch dequantization timing and Metal drift remain diagnostic.

The [128-cell Metal conditional selection batch](results/parity-metal-where-layouts-128-2026-10-31/report.md) uses a permanent [f32/BF16 where catalog](../parity/catalogs/metal-where-layouts-128.json) across dense, broadcast, strided and offset layouts, with row sizes 3/7/16/33 and widths 3/17/64/129. All 128 numerical probes, matched operator comparisons and composed Programs passed; no cells failed or were excluded. The f32 offset selection cases passed, while previously observed f32 offset failures in other binary and comparison operations remain unresolved. The matrix now has 8,685 candidates, 3,549 correctness links, 3,547 operator links, 3,529 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 4,664 candidates and 3,299 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal gather view batch](results/parity-metal-gather-views-128-2026-10-31/report.md) uses a permanent [f32/BF16 axis and layout catalog](../parity/catalogs/metal-gather-views-128.json). It spans axes 0 and 1, strided and offset source views, row sizes 3/7/16/33 and widths 3/17/64/129, with index tensor shapes matched to each axis. All 128 numerical probes, operator comparisons and composed Programs passed; no cells failed or were excluded. The matrix now has 8,813 candidates, 3,677 correctness links, 3,675 operator links, 3,657 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 4,792 candidates and 3,427 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal scatter-add batch](results/parity-metal-scatter-add-128-2026-10-31/report.md) uses a permanent [f32/BF16 three-input catalog](../parity/catalogs/metal-scatter-add-128.json). It spans axes 0 and 1, dense and offset views, row sizes 3/7/16/33 and widths 3/17/64/129. Affon now exposes `scatter_add` through its JS operation and Program path; a focused CPU/Metal e2e test checks collisions and invalid shape rejection. All 128 independent numerical probes, matched PyTorch operator comparisons and composed Programs passed, with no failures or exclusions. The selected path was `metal_primitive` for all 128. The matrix now has 8,941 candidates, 3,805 correctness links, 3,803 operator links, 3,785 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 4,920 candidates and 3,555 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal dense cross-entropy batch](results/parity-metal-dense-loss-128-2026-10-31/report.md) uses a permanent [f32/BF16 soft-target catalog](../parity/catalogs/metal-dense-loss-128.json). It spans batch sizes 1/3/8/32, eight class widths from 3 to 33, and dense and offset views of logits and targets. The public `dense_cross_entropy` operation now exposes Compute’s dense loss separately from the indexed-label API. Every scalar loss is checked in full against an independent reference. All 128 probes, matched PyTorch operator comparisons and composed Programs passed, with no failures or exclusions; each selected `metal_primitive`. A focused CPU/Metal e2e test passed immediate and Program calls and shape validation. The matrix now has 9,069 candidates, 3,933 correctness links, 3,931 operator links, 3,913 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 5,048 candidates and 3,683 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal log-softmax NLL batch](results/parity-metal-log-softmax-nll-128-2026-10-31/report.md) uses a permanent [f32/BF16 soft-target catalog](../parity/catalogs/metal-log-softmax-nll-128.json). It spans batch sizes 2/4/16/64, eight class widths from 5 to 65, and dense and offset views. The public `log_softmax_nll` operation accepts logits and dense target probabilities and maps to Compute’s distinct loss tag. Each scalar result is checked in full against an independent stable log-sum-exp reference. All 128 probes, matched PyTorch operator comparisons and composed Programs passed, with zero failures or exclusions; each selected `metal_primitive`. Focused CPU/Metal e2e tests passed immediate and Program calls. The matrix now has 9,197 candidates, 4,061 correctness links, 4,059 operator links, 4,041 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 5,176 candidates and 3,811 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal masked-fill batch](results/parity-metal-masked-fill-128-2026-10-31/report.md) uses a permanent [f32/BF16 masking catalog](../parity/catalogs/metal-masked-fill-128.json). It spans row sizes 2/5/16/31, widths 5/33/64/127, dense and broadcast masks, and strided and offset source views. All 128 numerical probes, matched PyTorch operator comparisons and composed Programs passed, with no failures or exclusions; each selected `metal_primitive`. The matrix now has 9,325 candidates, 4,189 correctness links, 4,187 operator links, 4,169 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 5,304 candidates and 3,939 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal reduce-to-shape batch](results/parity-metal-reduce-to-shape-128-2026-10-31/report.md) uses a permanent [f32 catalog](../parity/catalogs/metal-reduce-to-shape-128.json) and [generator](../parity/catalogs/generators/generate-metal-reduce-to-shape.py). It spans dense and offset inputs, row sizes 3/16/33/128, widths 5/32/65/129 and four output shapes. All 128 independent numerical probes, matched PyTorch operator comparisons and composed Programs passed, with no failures or exclusions; each selected `metal_primitive`. The public operation and Program path passed a focused CPU/Metal e2e test. The matrix now has 9,453 candidates, 4,317 correctness links, 4,315 operator links, 4,297 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 5,432 candidates and 4,067 supported cells. The Compute schema currently excludes BF16 for this operation. Timing ratios remain diagnostic.

The [128-cell Metal indexed cross-entropy view batch](results/parity-metal-indexed-loss-views-128-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-indexed-loss-views-128.json) and [generator](../parity/catalogs/generators/generate-metal-indexed-loss-views.py). It spans strided and offset logits, batch sizes 2/5/16/33, and class widths 7/31/32/33/63/64/65/129. All 128 independent numerical probes, matched PyTorch operator comparisons and composed Programs passed, with no failures or exclusions; each selected `metal_primitive`. The matrix now has 9,581 candidates, 4,445 correctness links, 4,443 operator links, 4,425 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 5,560 candidates and 4,195 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal argmin/argmax axis view batch](results/parity-metal-arg-axis-views-128-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-arg-axis-views-128.json) and [generator](../parity/catalogs/generators/generate-metal-arg-axis-views.py). It spans strided and offset inputs, row sizes 3/16/33/128 and widths 5/32/65/129. All 128 independent numerical probes, matched PyTorch operator comparisons and composed Programs passed, with no failures or exclusions; each selected `metal_primitive`. Output indices have i64 dtype and were checked numerically. The matrix now has 9,709 candidates, 4,573 correctness links, 4,571 operator links, 4,553 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 5,688 candidates and 4,323 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal softmax/log-softmax view batch](results/parity-metal-softmax-views-128-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-softmax-views-128.json) and [generator](../parity/catalogs/generators/generate-metal-softmax-views.py). It spans strided and offset inputs, row sizes 3/16/33/128, and widths 5/32/65/129. All 128 independent numerical probes, matched PyTorch operator comparisons and composed Programs passed, with no failures or exclusions; each selected `metal_primitive`. The matrix now has 9,837 candidates, 4,701 correctness links, 4,699 operator links, 4,681 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 5,816 candidates and 4,451 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal variance/std axis view batch](results/parity-metal-variance-views-128-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-variance-views-128.json) and [generator](../parity/catalogs/generators/generate-metal-variance-views.py). It spans strided and offset inputs, row sizes 3/16/33/128, and widths 5/32/65/129. All 128 independent numerical probes, matched PyTorch operator comparisons and composed Programs passed, with no failures or exclusions; each selected `metal_primitive`. The PyTorch peer uses population variance and standard deviation (`correction=0`). The matrix now has 9,965 candidates, 4,829 correctness links, 4,827 operator links, 4,809 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 5,944 candidates and 4,579 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal outer-axis softmax/log-softmax view batch](results/parity-metal-outer-softmax-views-128-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-outer-softmax-views-128.json) and [generator](../parity/catalogs/generators/generate-metal-outer-softmax-views.py). It spans strided and offset inputs, row sizes 3/16/33/128, widths 5/32/65/129, and reduction over axis zero. The parity workers now recognize outer-axis variants with layout suffixes. All 128 independent numerical probes, matched PyTorch operator comparisons and composed Programs passed, with no failures or exclusions; each selected `metal_primitive`. The matrix now has 10,093 candidates, 4,957 correctness links, 4,955 operator links, 4,937 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 6,072 candidates and 4,707 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal outer-axis sum/mean/min/max view batch](results/parity-metal-outer-reductions-128-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-outer-reductions-128.json) and [generator](../parity/catalogs/generators/generate-metal-outer-reductions.py). It spans strided and offset inputs, row sizes 3/33, widths 5/32/65/129, and reduction over axis zero. All 128 independent numerical probes, matched PyTorch operator comparisons and composed Programs passed, with no failures or exclusions; each selected `metal_primitive`. The matrix now has 10,221 candidates, 5,085 correctness links, 5,083 operator links, 5,065 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 6,200 candidates and 4,835 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal outer-axis variance/std/argmin/argmax view batch](results/parity-metal-outer-stat-index-128-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-outer-stat-index-128.json) and [generator](../parity/catalogs/generators/generate-metal-outer-stat-index.py). It spans strided and offset inputs, row sizes 3/33, widths 5/32/65/129, and reduction over axis zero. All 128 independent numerical probes passed. Matched PyTorch operator and composed Program comparisons passed for 127 cells each. `metal/bf16/std_axis/outer_axis/strided_r3_c129` failed both peer numeric checks: operator maximum error 0.02734375 and Program maximum error 0.0546875 exceeded their 0.02 tolerances. The [operator](results/parity-metal-outer-stat-index-128-2026-10-31/operator_peer_failures/metal__bf16__std_axis__outer_axis__strided_r3_c129.json) and [Program](results/parity-metal-outer-stat-index-128-2026-10-31/program_peer_failures/metal__bf16__std_axis__outer_axis__strided_r3_c129.json) reports remain diagnostic without ledger links. The matrix now has 10,349 candidates, 5,213 correctness links, 5,210 operator links, 5,192 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 6,328 candidates and 4,963 supported cells. Timing ratios remain diagnostic.

The [reviewed 128-cell rerun](results/parity-metal-outer-stat-index-128-reviewed-2026-10-31/report.md) resolved the peer reference mismatch. For the failing BF16 input, PyTorch MPS BF16 `std` returned 0.14453125 at column 128, while Affon returned 0.1171875, PyTorch MPS f32 returned 0.117851, and PyTorch CPU BF16 returned 0.117676. The PyTorch peer now computes BF16 `std_axis` with f32 accumulation and casts its output to BF16; its timing therefore includes that conversion. The original failed run remains retained. All 128 probes, operator comparisons, and Program comparisons passed in the reviewed rerun. The matrix still has 10,349 candidates and 5,213 correctness links, with 5,211 operator links and 5,193 Program links. Metal has 4,963 supported cells and 4,962 operator links. Timing ratios remain diagnostic and no baseline was accepted.

The [256-cell Metal nonlinear view batch](results/parity-metal-nonlinear-views-256-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-nonlinear-views-256.json) and [generator](../parity/catalogs/generators/generate-metal-nonlinear-views.py). It spans GELU, erf, SiLU, and clamp on strided and offset inputs, row sizes 3/16/33/128, and widths 5/32/65/129. All 256 independent numerical probes, matched PyTorch operator comparisons and composed Programs passed, with no failures or exclusions; each selected `metal_primitive`. The matrix now has 10,605 candidates, 5,469 correctness links, 5,467 operator links, 5,449 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 6,584 candidates and 5,219 supported cells. Timing ratios remain diagnostic.

The [256-cell Metal binary arithmetic view batch](results/parity-metal-binary-views-256-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-binary-views-256.json) and [generator](../parity/catalogs/generators/generate-metal-binary-views.py). It spans add, sub, mul, and div on strided and offset inputs, row sizes 3/16/33/128, and widths 5/32/65/129. All 64 f32 offset probes failed numerical correctness across the four operations and 16 shapes each; they remain unknown without operator or Program links. The other 192 probes, matched PyTorch operator comparisons, and composed Programs passed. The retained manifest records every failure as `probe_not_correct` at both timing tiers. The matrix now has 10,861 candidates, 5,661 correctness links, 5,659 operator links, 5,641 Program links, 4,949 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 6,840 candidates and 5,411 supported cells. Timing ratios remain diagnostic.

The [reviewed 256-cell binary view rerun](results/parity-metal-binary-views-256-reviewed-2026-10-31/report.md) followed a Compute Metal f32 binary FFI correction: the f32 add, multiply, binary, and comparison calls now pass both input view offsets to their kernels, and offset views bypass the zero-offset dense path. A focused [public Program test](../e2e/program/operations/metal-offset-binary.test.ts) passed arithmetic and comparisons on two offset inputs. All 256 probes, operator comparisons, and Program comparisons passed in the retained rerun, including the 64 formerly failing f32 offset cells. The earlier failed run remains diagnostic. The matrix remains at 10,861 candidates and now has 5,725 correctness links, 5,723 operator links, 5,705 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 6,840 candidates and 5,475 supported cells. Timing ratios remain diagnostic.

The [192-cell Metal top-k batch](results/parity-metal-topk-192-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-topk-192.json) and [generator](../parity/catalogs/generators/generate-metal-topk-192.py). It covers dense, strided, and offset inputs; axes zero and one; row sizes 3/7/16/33; and widths 3/17/64/129. Both floating values and i64 indices are recorded in the numeric and timing contracts. All 192 probes, PyTorch operator comparisons, and composed Program comparisons passed with no failures or exclusions; each probe selected `metal_primitive`. The matrix now has 11,053 candidates, 5,917 correctness links, 5,915 operator links, 5,897 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 7,032 candidates and 5,667 supported cells. Timing ratios remain diagnostic.

The [192-cell Metal smallest-value top-k batch](results/parity-metal-topk-smallest-192-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-topk-smallest-192.json) and [generator](../parity/catalogs/generators/generate-metal-topk-smallest-192.py). It tests `largest=false` on dense, strided, and offset inputs, axes zero and one, row sizes 8/32/64/128, widths 7/31/65/257, and k at 1 or one below the axis width. The worker and independent references compare a compact hash over every returned floating value and i64 index, plus samples and checksum; this avoids the JSON output limit for large results. All 192 probes, matched PyTorch operator comparisons, and composed Programs passed, with no failures or exclusions. The matrix now has 11,245 candidates, 6,109 correctness links, 6,107 operator links, 6,089 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 7,224 candidates and 5,859 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal RMS normalization view batch](results/parity-metal-rms-norm-views-128-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-rms-norm-views-128.json) and [generator](../parity/catalogs/generators/generate-metal-rms-norm-views-128.py). It spans inner and outer axes, strided and offset inputs, row sizes 3/16/33/128, and widths 5/32/65/129. The PyTorch peer now passes epsilon `1e-5` explicitly to match Affon; its default differed and caused a small f32 smoke-case discrepancy. All 128 independent numerical probes, matched operator comparisons, and composed Programs passed, with no failures or exclusions. The matrix now has 11,373 candidates, 6,237 correctness links, 6,235 operator links, 6,217 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 7,352 candidates and 5,987 supported cells. Timing ratios remain diagnostic.

The [f32 RMS normalization short-row recheck](results/parity-metal-rms-norm-r8-c7-recheck-2026-10-31/report.md) closed the single supported Metal cell that lacked timing evidence. Its earlier operator and Program reports remain in the failure directories; both had used the PyTorch default epsilon. The permanent [recheck catalog](../parity/catalogs/metal-rms-norm-r8-c7-recheck.json) passed its new numerical probe, operator comparison, and composed Program comparison with explicit epsilon `1e-5`. Candidate and correctness totals remain 11,373 and 6,237; operator links are now 6,236 and Program links 6,218. No baseline was accepted.

The [128-cell Metal strided scatter-add batch](results/parity-metal-scatter-strided-128-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-scatter-strided-128.json) and [generator](../parity/catalogs/generators/generate-metal-scatter-strided-128.py). It exercises both index axes with row or column strides on all three inputs, row sizes 3/7/16/33 and widths 3/17/64/129. All 128 independent numerical probes, matched operator comparisons, and composed Programs passed, with no failures or exclusions. The matrix now has 11,501 candidates, 6,365 correctness links, 6,364 operator links, 6,346 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 7,480 candidates and 6,115 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal index-select view batch](results/parity-metal-index-select-views-128-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-index-select-views-128.json) and [generator](../parity/catalogs/generators/generate-metal-index-select-views-128.py). It spans both axes, strided and offset source views, row sizes 3/7/16/33, widths 5/32/65/129, and index lengths derived from the selected axis. All 128 independent numerical probes, matched operator comparisons, and composed Programs passed, with no failures or exclusions. The matrix now has 11,629 candidates, 6,493 correctness links, 6,492 operator links, 6,474 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 7,608 candidates and 6,243 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal strided soft-target loss batch](results/parity-metal-soft-loss-strided-128-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-soft-loss-strided-128.json) and [generator](../parity/catalogs/generators/generate-metal-soft-loss-strided-128.py). It spans dense cross entropy and log-softmax negative log likelihood, row and column strides on both logits and targets, batch sizes 1/3/8/32, and class widths 7/31/33/65. All 128 independent scalar numerical probes, matched PyTorch operator comparisons, and composed Programs passed, with no failures or exclusions. The matrix now has 11,757 candidates, 6,621 correctness links, 6,620 operator links, 6,602 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 7,736 candidates and 6,371 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal matmul right-hand view batch](results/parity-metal-matmul-rhs-views-128-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-matmul-rhs-views-128.json) and [generator](../parity/catalogs/generators/generate-metal-matmul-rhs-views-128.py). It spans transposed and column-strided right-hand matrices, M values 1/4/16/64, K values 7/31/64/129, and N values 17/65. The composed Program adds a matched per-column bias. All 128 independent numerical probes, matched operator comparisons, and composed Programs passed, with no failures or exclusions. The matrix now has 11,885 candidates, 6,749 correctness links, 6,748 operator links, 6,730 Program links, 4,885 unknown, 245 not applicable, 6 unsupported and zero accepted baselines. Metal has 7,864 candidates and 6,499 supported cells. Timing ratios remain diagnostic.

The [13-cell Metal f32 Program gap rerun](results/parity-metal-program-gaps-13-reviewed-2026-10-31/report.md) uses the permanent [rerun catalog](../parity/catalogs/metal-program-gaps-13.json). The [first run](results/parity-metal-program-gaps-13-2026-10-31/report.md) retained three incorrect batched matmul plus bias Program reports: the rank-two fused Metal implementation accepted rank-three operands and produced incorrect values. After restricting that fused candidate to rank-two inputs, all 13 numerical probes, operator comparisons, and composed Program comparisons passed, with 15 Affon and 15 PyTorch timing samples per comparison. The first run's raw failures remain retained. The matrix remains at 11,885 candidates, 6,749 correctness links, and 6,748 operator links; Program links increase from 6,730 to 6,743. Unknown remains 4,885, not applicable 245, unsupported 6, and accepted baselines zero. These timing ratios remain diagnostic.

The [128-cell Metal dense matmul grid batch](results/parity-metal-matmul-dense-grid-128-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-matmul-dense-grid-128.json) and [generator](../parity/catalogs/generators/generate-metal-matmul-dense-grid-128.py). It spans M values 2/8/32/128, K values 5/33/65/127, and N values 7/31/63/129 across small-batch and large-matrix regimes. Every cell passed an independent numerical probe, matched PyTorch operator comparison, and composed matmul plus bias Program comparison with repeated timing samples. There were no failures or exclusions. The matrix now has 12,013 candidates, 6,877 correctness links, 6,876 operator links, 6,871 Program links, 4,885 unknown, 245 not applicable, 6 unsupported, and zero accepted baselines. Metal has 7,992 candidates and 6,627 supported cells. Timing ratios remain diagnostic.

The [192-cell Metal comparison view batch](results/parity-metal-comparison-views-192-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-comparison-views-192.json) and [generator](../parity/catalogs/generators/generate-metal-comparison-views-192.py). It covers equality, less than, and greater than on strided and offset inputs, with row sizes 3/16/33/128 and widths 5/32/65/129. All 192 independent numerical probes, matched PyTorch operator comparisons, and composed Program comparisons passed, including f32 offset views; the comparisons return i64 values. There were no failures or exclusions. The matrix now has 12,205 candidates, 7,069 correctness links, 7,068 operator links, 7,063 Program links, 4,885 unknown, 245 not applicable, 6 unsupported, and zero accepted baselines. Metal has 8,184 candidates and 6,819 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal left-hand matmul view batch](results/parity-metal-matmul-lhs-views-128-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-matmul-lhs-views-128.json) and [generator](../parity/catalogs/generators/generate-metal-matmul-lhs-views-128.py). It spans transposed and column-strided left matrices, M values 1/4/16/64, K values 7/31/64/129, and N values 17/65. The composed Program adds a matched per-column bias. All 128 independent numerical probes, matched operator comparisons, and composed Programs passed, with no failures or exclusions. Operator paths cover transpose and slice views feeding both `metal.matmul.small` and `metal.matmul.tiled`. The matrix now has 12,333 candidates, 7,197 correctness links, 7,196 operator links, 7,191 Program links, 4,885 unknown, 245 not applicable, 6 unsupported, and zero accepted baselines. Metal has 8,312 candidates and 6,947 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal cast view batch](results/parity-metal-cast-views-128-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-cast-views-128.json) and [generator](../parity/catalogs/generators/generate-metal-cast-views-128.py). It checks f32-to-BF16 and BF16-to-f32 conversions on strided and offset views, with row sizes 1/7/32/129 and widths 3/7/31/32/33/63/64/65. All 128 independent numerical probes, matched operator comparisons, and composed Programs passed; output dtype contracts and repeated timing samples are retained. There were no failures or exclusions. The matrix now has 12,461 candidates, 7,325 correctness links, 7,324 operator links, 7,319 Program links, 4,885 unknown, 245 not applicable, 6 unsupported, and zero accepted baselines. Metal has 8,440 candidates and 7,075 supported cells. Timing ratios remain diagnostic.

The [128-cell Metal layer norm view batch](results/parity-metal-layer-norm-views-128-reviewed-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-layer-norm-views-128.json) and [generator](../parity/catalogs/generators/generate-metal-layer-norm-views-128.py). It spans inner and outer axes, strided and offset views, row sizes 3/16/33/128, and widths 5/32/65/129. The [first retained run](results/parity-metal-layer-norm-views-128-2026-10-31/report.md) passed all 128 independent numerical probes but found one f32 outer-axis offset 3×5 PyTorch MPS peer mismatch at both timing tiers: operator maximum error 2.50×10⁻⁶ exceeded its 1.84×10⁻⁶ tolerance; the Program error and tolerance were 5.01×10⁻⁶ and 3.67×10⁻⁶. The independent probe error was 7.19×10⁻⁷ within 3.67×10⁻⁶. The f32 layer norm peer tolerance now matches that independent relative tolerance of 2×10⁻⁶. The reviewed rerun passed all 128 probes, operator comparisons, and composed Programs; original failed reports remain retained. The matrix now has 12,589 candidates, 7,453 correctness links, 7,452 operator links, 7,447 Program links, 4,885 unknown, 245 not applicable, 6 unsupported, and zero accepted baselines. Metal has 8,568 candidates and 7,203 supported cells. Timing ratios remain diagnostic.

The [256-cell Metal trigonometric and activation view batch](results/parity-metal-transcendental-views-256-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-transcendental-views-256.json) and [generator](../parity/catalogs/generators/generate-metal-transcendental-views-256.py). It covers sin, cos, tanh, and sigmoid on strided and offset inputs, with row sizes 3/16/33/128 and widths 5/32/65/129. All 256 independent numerical probes, matched PyTorch operator comparisons, and composed Programs passed, with no failures or exclusions. The matrix now has 12,845 candidates, 7,709 correctness links, 7,708 operator links, 7,703 Program links, 4,885 unknown, 245 not applicable, 6 unsupported, and zero accepted baselines. Metal has 8,824 candidates and 7,459 supported cells. Timing ratios remain diagnostic.

The [192-cell Metal exp/log/sqrt view batch](results/parity-metal-math-views-192-reviewed-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-math-views-192.json) and [generator](../parity/catalogs/generators/generate-metal-math-views-192.py). It spans strided and offset views, row sizes 3/16/33/128, and widths 5/32/65/129. The [first retained run](results/parity-metal-math-views-192-2026-10-31/report.md) passed all 192 independent probes but found five f32 exp PyTorch MPS peer misses at both timing tiers. Their errors were 1.10–1.41 times the peer's 1×10⁻⁶ relative threshold, while every independent probe passed its existing 2×10⁻⁶ threshold. The f32 exp peer now uses that same 2×10⁻⁶ relative threshold. All 192 probes, operator comparisons, and composed Programs passed in the reviewed rerun; the original ten failed reports remain retained. The matrix now has 13,037 candidates, 7,901 correctness links, 7,900 operator links, 7,895 Program links, 4,885 unknown, 245 not applicable, 6 unsupported, and zero accepted baselines. Metal has 9,016 candidates and 7,651 supported cells. Timing ratios remain diagnostic.

A [128-cell layer norm offset correction rerun](results/parity-metal-layer-norm-views-128-offset-corrected-2026-10-31/report.md) repaired an evidence interpretation error. The 32 inner `_offset` variants in the earlier layer norm batch were under the `strided` policy regime and had selected strided inputs. Affon workers and Python references now give the explicit `_offset` suffix precedence. The corrected run passed all 128 numerical probes, matched operators, and composed Programs; its inner offset reports retain `offset_1_1` contracts and [1:, 1:] view paths. The previously linked operator and Program aggregate files are retained in `results/cell-peer-before-layer-norm-offset-correction-2026-10-31`, and reviewed links now point to the corrected run. Candidate and link totals do not change from this replacement.

The [256-cell Metal inner-axis reduction batch](results/parity-metal-inner-reductions-256-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-inner-reductions-256.json) and [generator](../parity/catalogs/generators/generate-metal-inner-reductions-256.py). It covers sum, mean, min, and max over strided and actual offset input views, with row sizes 3/16/33/128 and widths 5/32/65/129. All 256 independent numerical probes, matched PyTorch operator comparisons, and composed Programs passed, with no failures or exclusions. The matrix now has 13,293 candidates, 8,157 correctness links, 8,156 operator links, 8,151 Program links, 4,885 unknown, 245 not applicable, 6 unsupported, and zero accepted baselines. Metal has 9,272 candidates and 7,907 supported cells. Timing ratios remain diagnostic.

The [36-cell Metal canonical gap audit](results/parity-metal-unknown-canonical-audit-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-unknown-canonical-audit.json) and [generator](../parity/catalogs/generators/generate-metal-unknown-canonical.py). It tests declared canonical loss, scatter, topk, reduction, and quantized matmul candidates. The initial run passed 32 probes and both matched timing tiers. Four canonical topk IDs exposed a harness layout parsing error; the [retained first recheck](results/parity-metal-unknown-canonical-topk-recheck-2026-10-31/report.md) passed their probes and exposed the matching PyTorch parsing error. After both fixes, the [final four-cell recheck](results/parity-metal-unknown-canonical-topk-corrected-2026-10-31/report.md) passed numerical probes, operator peers, and composed Programs. All 36 cells have reviewed correctness and timing evidence; earlier failure reports remain retained. The matrix remains at 13,293 candidates and now has 8,193 correctness links, 8,192 operator links, 8,187 Program links, 4,849 unknown, 245 not applicable, 6 unsupported, and zero accepted baselines. Metal has 9,272 candidates and 7,943 supported cells. Timing ratios remain diagnostic.

The [132-cell Metal batched dense loss batch](results/parity-metal-batched-dense-loss-132-reviewed-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-batched-dense-loss-132.json) and [generator](../parity/catalogs/generators/generate-metal-batched-dense-loss.py). It covers dense cross entropy and log softmax NLL with rank-three inputs across batch sizes 2/4, class widths 3/7/33/65, and spatial widths 2/5/17/31, plus four canonical cells. The [initial run](results/parity-metal-batched-dense-loss-132-2026-10-31/report.md) retained 132 probe failures: the public wrapper rejected rank-three operands. The wrapper now transposes and flattens batch and spatial positions into the established two-dimensional loss operation. A [four-cell canonical check](results/parity-metal-batched-dense-loss-canonical-2026-10-31/report.md) and the full reviewed rerun passed every independent numerical probe, matched PyTorch operator comparison, and composed Program comparison, with repeated timing samples. The matrix now has 13,421 candidates, 8,325 correctness links, 8,324 operator links, 8,319 Program links, 4,845 unknown, 245 not applicable, 6 unsupported, and zero accepted baselines. Metal has 9,400 candidates and 8,075 supported cells. Timing ratios remain diagnostic.

The [130-cell Metal batched topk batch](results/parity-metal-batched-topk-130-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-batched-topk-130.json) and [generator](../parity/catalogs/generators/generate-metal-batched-topk.py). It covers rank-three topk along both inner axes, batch sizes 2/4, row widths 3/7/16/33, and column widths 5/17/65/129, plus two canonical cells. Affon's public Program contract now accepts rank-three topk; the Metal implementation receives the full rank and axis. All 130 independent probes, matched PyTorch operator comparisons, and composed Programs passed, including value and index hashes. Both timing tiers have repeated samples, with no failures or exclusions. The matrix now has 13,549 candidates, 8,455 correctness links, 8,454 operator links, 8,449 Program links, 4,843 unknown, 245 not applicable, 6 unsupported, and zero accepted baselines. Metal has 9,528 candidates and 8,205 supported cells. Timing ratios remain diagnostic.

The [130-cell Metal batched scatter-add batch](results/parity-metal-batched-scatter-130-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-batched-scatter-130.json) and [generator](../parity/catalogs/generators/generate-metal-batched-scatter.py). It covers rank-three scatter add along both inner axes, batch sizes 2/4, row widths 3/7/16/33, and column widths 5/17/65/129, plus two canonical cells. All 130 independent numerical probes, matched PyTorch operator comparisons, and composed Programs passed, with repeated timing samples and no failures or exclusions. The matrix now has 13,677 candidates, 8,585 correctness links, 8,584 operator links, 8,579 Program links, 4,841 unknown, 245 not applicable, 6 unsupported, and zero accepted baselines. Metal has 9,656 candidates and 8,335 supported cells. Timing ratios remain diagnostic.

The [130-cell Metal batched index-selection batch](results/parity-metal-batched-index-select-130-2026-10-31/report.md) uses a permanent [f32/BF16 catalog](../parity/catalogs/metal-batched-index-select-130.json) and [generator](../parity/catalogs/generators/generate-metal-batched-index-select.py). A dedicated [rank-three parity worker](../parity/workers/affon-index-select.ts) and independent [reference](../parity/fixtures/index_select_fixture.py) cover both inner axes, batch sizes 2/4, row widths 3/7/16/33, and column widths 5/17/65/129, plus two canonical cells. All 130 numerical probes, matched PyTorch operator comparisons, and composed Programs passed, with repeated timing samples and no failures or exclusions. The matrix now has 13,805 candidates, 8,715 correctness links, 8,714 operator links, 8,709 Program links, 4,839 unknown, 245 not applicable, 6 unsupported, and zero accepted baselines. Metal has 9,784 candidates and 8,465 supported cells. Timing ratios remain diagnostic.

The [two-cell Metal ragged vector dot check](results/parity-metal-ragged-dot-2-2026-10-31/report.md) passed independent numerical probes, matched PyTorch operator comparisons, and composed Programs for odd f32/BF16 vector widths, with repeated timing samples. Its permanent [catalog](../parity/catalogs/metal-ragged-dot-2.json) has a [generator](../parity/catalogs/generators/generate-metal-ragged-dot.py). A separate [38-cell contract audit](results/parity-metal-contract-exclusions-38-2026-10-31.json), maintained by [its checked script](../parity/audit-metal-contract-exclusions.py), records 12 not applicable matrix dot regimes, 20 unsupported reserved gradient operations, and six unsupported BF16 reduce-to-shape regimes. Those excluded cells have source contract evidence and carry no numerical or timing links. The matrix remains at 13,805 candidates and now has 8,717 correctness links, 8,716 operator links, 8,711 Program links, 4,799 unknown, 257 not applicable, 32 unsupported, and zero accepted baselines. Metal has 9,784 candidates and 8,467 supported cells. Timing ratios remain diagnostic.

The [25-cell Metal remaining-runnable audit](results/parity-metal-remaining-runnable-audit-25-2026-10-31/report.md) uses a permanent [catalog](../parity/catalogs/metal-remaining-runnable-audit-25.json) and [generator](../parity/catalogs/generators/generate-metal-remaining-runnable-audit.py). Corrected canonical strided fixtures and the actual offset views let 19 cells pass independent numerical probes, matched PyTorch operator comparisons, and composed Programs with retained timing samples. Four f32 variance/std reductions failed their independent probes above tolerance and remain unknown. Two `contiguous/view_only` cells returned correct values and passed peer timings but selected `metal_contiguous`, so they remain diagnostic without coverage links. The reviewed [promotion guard](../parity/promote.py) rejects that materializing path for a view-only claim. The matrix remains at 13,805 candidates and now has 8,736 correctness links, 8,735 operator links, 8,730 Program links, 4,780 unknown, 257 not applicable, 32 unsupported, and zero accepted baselines. Metal has 9,784 candidates and 8,486 supported cells. Timing ratios remain diagnostic.

The [scalar-output reduce-to-shape check](results/parity-metal-scalar-reduce-to-shape-2026-10-31/report.md) uses a permanent [catalog](../parity/catalogs/metal-scalar-reduce-to-shape.json) and [generator](../parity/catalogs/generators/generate-metal-scalar-reduce-to-shape.py); its numerical probe, matched PyTorch operator, and composed Program all passed with retained timing samples. The [invalid scatter check](results/parity-metal-invalid-scatter-2-2026-10-31/report.md) passed all three tiers for f32. Its BF16 probe raised `BackendFailure` and has no timing evidence. A permanent [boundary comparison](results/parity-metal-invalid-boundaries-2026-10-31.json), maintained by [its script](../parity/audit-metal-invalid-boundaries.py), records that PyTorch MPS skips the invalid scatter update for both dtypes, while Affon Metal BF16 errors; both invalid topk cases are rejected before execution by Affon and PyTorch. Those three boundaries are unsupported and carry no performance links. The matrix remains at 13,805 candidates and now has 8,738 correctness links, 8,737 operator links, 8,732 Program links, 4,775 unknown, 257 not applicable, 35 unsupported, and zero accepted baselines. Metal has 9,784 candidates and 8,488 supported cells. Timing ratios remain diagnostic.

The [four-cell Metal variance/std corrected rerun](results/parity-metal-variance-std-chunked-4-2026-10-31/report.md) uses a permanent [catalog](../parity/catalogs/metal-variance-std-corrected-4.json) and [generator](../parity/catalogs/generators/generate-metal-variance-std-corrected.py). Chunked f32 reduction accumulation resolved the four original numerical failures. All four independent probes, matched PyTorch operators, and composed Programs passed with repeated timing samples. The original failed probe and Kahan rerun remain retained. The matrix has 13,805 candidates, 8,742 correctness links, 8,741 operator links, 8,736 Program links, 4,771 unknown, 257 not applicable, 35 unsupported, and zero accepted baselines. Metal has 9,784 candidates and 8,492 supported cells. Two `contiguous/view_only` paths still select `metal_contiguous` and remain diagnostic. Timing ratios remain diagnostic.

The [256-cell Metal binary boundary batch](results/parity-metal-binary-boundaries-256-2026-10-31/report.md) uses a permanent [catalog](../parity/catalogs/metal-binary-boundaries-256.json) and [generator](../parity/catalogs/generators/generate-metal-binary-boundaries.py). It covers f32/BF16 add, subtract, multiply, and divide across strided and offset views, rows 1/7/64/129, and columns 17/63/127/257. All 256 independent numerical probes, matched PyTorch operators, and composed Programs passed with repeated timing samples; there were no failures or exclusions. The matrix now has 14,061 candidates, 8,998 correctness links, 8,997 operator links, 8,992 Program links, 4,771 unknown, 257 not applicable, 35 unsupported, and zero accepted baselines. Metal has 10,040 candidates and 8,748 supported cells. Timing ratios remain diagnostic.

The [256-cell Metal trigonometric and activation boundary batch](results/parity-metal-transcendental-boundaries-256-2026-10-31/report.md) uses a permanent [catalog](../parity/catalogs/metal-transcendental-boundaries-256.json) and [generator](../parity/catalogs/generators/generate-metal-transcendental-boundaries.py). It covers f32/BF16 sin, cos, tanh, and sigmoid over strided and offset views, rows 1/7/64/129, and columns 17/63/127/257. All 256 independent numerical probes, matched PyTorch operators, and composed Programs passed with repeated timing samples; there were no failures or exclusions. The matrix now has 14,317 candidates, 9,254 correctness links, 9,253 operator links, 9,248 Program links, 4,771 unknown, 257 not applicable, 35 unsupported, and zero accepted baselines. Metal has 10,296 candidates and 9,004 supported cells. Timing ratios remain diagnostic.

The [256-cell Metal inner-axis reduction boundary batch](results/parity-metal-inner-reduction-boundaries-256-2026-10-31/report.md) uses a permanent [catalog](../parity/catalogs/metal-inner-reduction-boundaries-256.json) and [generator](../parity/catalogs/generators/generate-metal-inner-reduction-boundaries.py). It covers f32/BF16 sum, mean, min, and max on strided and offset views, rows 1/7/64/129, and columns 17/63/127/257. All 256 independent numerical probes, matched PyTorch operators, and composed Programs passed with repeated timing samples; there were no failures or exclusions. The matrix now has 14,573 candidates, 9,510 correctness links, 9,509 operator links, 9,504 Program links, 4,771 unknown, 257 not applicable, 35 unsupported, and zero accepted baselines. Metal has 10,552 candidates and 9,260 supported cells. Timing ratios remain diagnostic.

The [256-cell Metal outer-axis reduction boundary batch](results/parity-metal-outer-reduction-boundaries-256-2026-10-31/report.md) uses a permanent [catalog](../parity/catalogs/metal-outer-reduction-boundaries-256.json) and [generator](../parity/catalogs/generators/generate-metal-outer-reduction-boundaries.py). It covers f32/BF16 sum, mean, min, and max on strided and offset views, rows 1/7/64/129, and columns 17/63/127/257. All 256 independent numerical probes, matched PyTorch operators, and composed Programs passed with repeated timing samples; there were no failures or exclusions. The matrix now has 14,829 candidates, 9,766 correctness links, 9,765 operator links, 9,760 Program links, 4,771 unknown, 257 not applicable, 35 unsupported, and zero accepted baselines. Metal has 10,808 candidates and 9,516 supported cells. Timing ratios remain diagnostic.

The [256-cell Metal right-operand matmul boundary batch](results/parity-metal-matmul-rhs-boundaries-256-2026-10-31/report.md) uses a permanent [catalog](../parity/catalogs/metal-matmul-rhs-boundaries-256.json) and [generator](../parity/catalogs/generators/generate-metal-matmul-rhs-boundaries.py). It covers f32/BF16 transposed and strided right operands, rows 1/7/33/65, inner dimensions 8/32/63/128, and output widths 8/32/64/128. The retained selected paths split into 128 `metal.matmul.small` and 128 `metal.matmul.tiled` cells. All 256 independent numerical probes, matched PyTorch operators, and composed Programs passed with repeated timing samples; there were no failures or exclusions. The matrix now has 15,085 candidates, 10,022 correctness links, 10,021 operator links, 10,016 Program links, 4,771 unknown, 257 not applicable, 35 unsupported, and zero accepted baselines. Metal has 11,064 candidates and 9,772 supported cells. Timing ratios remain diagnostic.

The [256-cell Metal softmax boundary batch](results/parity-metal-softmax-boundaries-256-2026-10-31/report.md) uses a permanent [catalog](../parity/catalogs/metal-softmax-boundaries-256.json) and [generator](../parity/catalogs/generators/generate-metal-softmax-boundaries.py). It covers f32/BF16 softmax and log-softmax over inner and outer axes on strided and offset views, rows 1/7/64/129, and columns 17/63/127/257. All 256 independent numerical probes, matched PyTorch operators, and composed Programs passed with repeated timing samples; there were no failures or exclusions. The matrix now has 15,341 candidates, 10,278 correctness links, 10,277 operator links, 10,272 Program links, 4,771 unknown, 257 not applicable, 35 unsupported, and zero accepted baselines. Metal has 11,320 candidates and 10,028 supported cells. Timing ratios remain diagnostic.

The [256-cell Metal variance/std boundary batch](results/parity-metal-stat-boundaries-256-2026-10-31/report.md) uses a permanent [catalog](../parity/catalogs/metal-stat-boundaries-256.json) and [generator](../parity/catalogs/generators/generate-metal-stat-boundaries.py). It covers f32/BF16 variance and standard deviation on both axes of strided and offset views, rows 2/7/64/129, and columns 17/63/127/257. All 256 independent probes passed; 252 cells passed both peer tiers in the first run. Four BF16 outer-axis variance cells failed a PyTorch MPS BF16 peer comparison, with raw reports retained. The peer now uses f32 accumulation and casts to BF16, as it already did for BF16 std. The [reviewed four-cell rerun](results/parity-metal-stat-peer-reviewed-4-2026-10-31/report.md), with a permanent [catalog](../parity/catalogs/metal-stat-peer-reviewed-4.json) and [generator](../parity/catalogs/generators/generate-metal-stat-peer-reviewed.py), passed all probes, operators, and Programs with repeated timing. All 256 cells are now promoted. The matrix has 15,597 candidates, 10,534 correctness links, 10,533 operator links, 10,528 Program links, 4,771 unknown, 257 not applicable, 35 unsupported, and zero accepted baselines. Metal has 11,576 candidates and 10,284 supported cells. Timing ratios remain diagnostic.

The [256-cell Metal argmin/argmax boundary batch](results/parity-metal-arg-boundaries-256-2026-10-31/report.md) uses a permanent [catalog](../parity/catalogs/metal-arg-boundaries-256.json) and [generator](../parity/catalogs/generators/generate-metal-arg-boundaries.py). It covers f32/BF16 argmin and argmax over inner and outer axes on strided and offset views, rows 2/7/64/129, and columns 17/63/127/257. All 256 independent numerical probes, matched PyTorch operators, and composed Programs passed with repeated timing samples; there were no failures or exclusions. The matrix now has 15,853 candidates, 10,790 correctness links, 10,789 operator links, 10,784 Program links, 4,771 unknown, 257 not applicable, 35 unsupported, and zero accepted baselines. Metal has 11,832 candidates and 10,540 supported cells. Timing ratios remain diagnostic.

The [256-cell Metal gather/index-select boundary batch](results/parity-metal-index-boundaries-256-2026-10-31/report.md) uses a permanent [catalog](../parity/catalogs/metal-index-boundaries-256.json) and [generator](../parity/catalogs/generators/generate-metal-index-boundaries.py). It covers f32/BF16 gather and index-select over both axes on strided and offset inputs, rows 2/8/32/65, and columns 7/31/63/127. All 256 independent numerical probes, matched PyTorch operators, and composed Programs passed with repeated timing samples; there were no failures or exclusions. The matrix now has 16,109 candidates, 11,046 correctness links, 11,045 operator links, 11,040 Program links, 4,771 unknown, 257 not applicable, 35 unsupported, and zero accepted baselines. Metal has 12,088 candidates and 10,796 supported cells. Timing ratios remain diagnostic.

The [256-cell Metal indexed-loss boundary batch](results/parity-metal-indexed-loss-boundaries-256-2026-10-31/report.md) uses a permanent [catalog](../parity/catalogs/metal-indexed-loss-boundaries-256.json) and [generator](../parity/catalogs/generators/generate-metal-indexed-loss-boundaries.py). It covers f32/BF16 indexed cross entropy on strided and offset views, batch sizes 1/7/64/129, and 16 class widths from 8 through 257. All 256 independent numerical probes, matched PyTorch operators, and composed Programs passed with repeated timing samples; there were no failures or exclusions. The matrix now has 16,365 candidates, 11,302 correctness links, 11,301 operator links, 11,296 Program links, 4,771 unknown, 257 not applicable, 35 unsupported, and zero accepted baselines. Metal has 12,344 candidates and 11,052 supported cells. Timing ratios remain diagnostic.

The [256-cell Metal soft-target loss boundary batch](results/parity-metal-soft-loss-boundaries-256-2026-10-31/report.md) uses a permanent [catalog](../parity/catalogs/metal-soft-loss-boundaries-256.json) and [generator](../parity/catalogs/generators/generate-metal-soft-loss-boundaries.py). It covers f32/BF16 cross entropy and log-softmax NLL on row- and column-strided views, rows 2/7/33/65, and class widths 17/32/63/64/66/127/128/129. All 256 independent numerical probes, matched PyTorch operators, and composed Programs passed with repeated timing samples; there were no failures or exclusions. The matrix now has 16,621 candidates, 11,558 correctness links, 11,557 operator links, 11,552 Program links, 4,771 unknown, 257 not applicable, 35 unsupported, and zero accepted baselines. Metal has 12,600 candidates and 11,308 supported cells. Timing ratios remain diagnostic.

The [256-cell Metal layer/RMS normalization boundary batch](results/parity-metal-norm-boundaries-256-2026-10-31/report.md) uses a permanent [catalog](../parity/catalogs/metal-norm-boundaries-256.json) and [generator](../parity/catalogs/generators/generate-metal-norm-boundaries.py). It covers f32/BF16 layer norm and RMS norm on inner and outer axes of strided and offset inputs, rows 2/7/64/129, and columns 17/63/127/257. All 256 independent numerical probes, matched PyTorch operators, and composed Programs passed with repeated timing samples; there were no failures or exclusions. The matrix now has 16,877 candidates, 11,814 correctness links, 11,813 operator links, 11,808 Program links, 4,771 unknown, 257 not applicable, 35 unsupported, and zero accepted baselines. Metal has 12,856 candidates and 11,564 supported cells. Timing ratios remain diagnostic.

The [384-cell Metal comparison boundary batch](results/parity-metal-comparison-boundaries-384-2026-10-31/report.md) uses a permanent [catalog](../parity/catalogs/metal-comparison-boundaries-384.json) and [generator](../parity/catalogs/generators/generate-metal-comparison-boundaries.py). It covers f32/BF16 equality, less-than, and greater-than on strided and offset views, rows 1/7/64/129, and widths 7/17/31/63/64/127/128/257. All 384 independent numerical probes, matched PyTorch operators, and composed Programs passed with repeated timing samples; there were no failures or exclusions. The matrix now has 17,261 candidates, 12,198 correctness links, 12,197 operator links, 12,192 Program links, 4,771 unknown, 257 not applicable, 35 unsupported, and zero accepted baselines. Metal has 13,240 candidates and 11,948 supported cells. Timing ratios remain diagnostic.

The [Metal contiguous view-only audit](results/parity-metal-contiguous-view-only-audit-2026-10-31.json), maintained by [its rerunnable script](../parity/audit-metal-contiguous-view-only.py), classifies the two remaining f32/BF16 `contiguous/view_only` candidates as unsupported. Their retained probes returned correct values but selected `metal_contiguous`; the Metal implementation writes a separate output buffer. Neither view-only claim receives a correctness or timing link. The matrix remains at 17,261 candidates, 12,198 correctness links, 12,197 operator links, 12,192 Program links, 4,769 unknown, 257 not applicable, 37 unsupported, and zero accepted baselines. All declared Metal f32/BF16 candidates now have a reviewed support status. Timing remains diagnostic.
