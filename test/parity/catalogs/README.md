# Catalog contracts and retained findings

These notes explain the contracts and historical findings behind named parity catalogs. See [the facility guide](../README.md) for running batches and [the artifact index](../../benchmark/results/PARITY-INDEX.md) for all catalogs and retained reports.

Quantized matmul uses `affon-quantized.ts` and `fixtures/quantized_fixture.py`. Its
catalog records M, K, N, packing bits and group size for every four-input
cell. Affon consumes packed u32 weights and floating input, scales and biases;
the independent probe decodes each packed value in Python. The PyTorch timing
peer unpacks, applies group affine parameters and multiplies on the GPU. Its
timing includes those separate PyTorch operations, so ratios describe this
specific peer implementation and remain diagnostic. Both operator and Program
comparisons check output shape, f32 output type, values and timing samples.

The `metal-quantized-views-128.json` catalog exercises a transposed input with
an offset packed-weight view, and a separate layout with offsets on all four
inputs. The Python peer applies equivalent tensor views before unpacking.

Scatter-add uses `affon-scatter.ts` and `fixtures/scatter_fixture.py`. Its three-input
contract includes a floating base, i64 indices and floating updates. The
independent reference accumulates updates into their indexed destinations;
the PyTorch peer uses `scatter_add` with equivalent views. The catalog covers
both axes, dense inputs and offset views on all three inputs.

Dense soft-target cross entropy uses `affon-dense-loss.ts` and
`fixtures/dense_loss_fixture.py`. The public `dense_cross_entropy` operation maps to
Compute’s dense cross entropy, while the existing `cross_entropy` public API
continues to accept integer class labels. Each batch cell checks the complete
scalar loss against an independent log-sum-exp reference and repeated PyTorch
soft-target timings. Offset cases slice both logits and target tensors.

The `metal-log-softmax-nll-128.json` catalog uses the same independent soft
target reference for Compute’s separate `log_softmax_nll` operation. Its inputs
are logits and dense target probabilities; the operation computes the stable
log-softmax and mean negative log likelihood internally. The batch covers
class-width boundaries 31/32/33 and 63/64/65, plus offset views.

Reduce to shape uses `affon-reduce-shape.ts` and `fixtures/reduce_shape_fixture.py`.
The `metal-reduce-to-shape-128.json` catalog covers f32 dense and offset
inputs, four row sizes, four widths, and four output shapes. The independent
reference sums source coordinates into each output coordinate. The PyTorch
peer uses `sum_to_size` on the corresponding MPS view. Each run retains
operator and composed Program correctness and repeated timing samples.

The `metal-indexed-loss-views-128.json` catalog checks indexed cross entropy
with f32/BF16 logits in strided and offset views. Its independent log-sum-exp
reference and matched PyTorch operator and Program runs cover four batch sizes
and eight class widths, including 31/32/33 and 63/64/65 boundaries.

The `metal-arg-axis-views-128.json` catalog checks f32/BF16 argmin and argmax
over the innermost axis of strided and offset views. It covers four row sizes
and four widths. The numeric result is an i64 index; operator and Program
comparisons also retain repeated matched PyTorch timing samples.

The `metal-softmax-views-128.json` catalog checks f32/BF16 softmax and
log-softmax on the innermost axis of strided and offset views. Four row sizes
and four widths exercise numerical normalization and matched PyTorch operator
and Program timing parity.

The `metal-variance-views-128.json` catalog checks f32/BF16 population
variance and standard deviation across the innermost axis of strided and
offset views. It spans four row sizes and four widths, with independent
numeric reference values and matched PyTorch operator and Program timings.

The `metal-outer-softmax-views-128.json` catalog checks f32/BF16 softmax and
log-softmax over axis zero on strided and offset views. The parity workers
recognize outer-axis variants with layout suffixes and retain full numerical
probes plus matched operator and Program timing reports for every cell.

The `metal-outer-reductions-128.json` catalog checks f32/BF16 sum, mean, min,
and max over axis zero of strided and offset views. Two row sizes and four
widths provide 128 cells with independent numerical references and repeated
matched PyTorch operator and Program timings.

The `metal-outer-stat-index-128.json` catalog checks f32/BF16 variance,
standard deviation, argmin, and argmax over axis zero on strided and offset
views. Its retained run contains one BF16 strided std case whose operator and
Program comparisons failed the numeric tolerance; both raw reports remain in
the failure directories and neither timing result is linked into the ledger.
The [reviewed rerun](../../benchmark/results/parity-metal-outer-stat-index-128-reviewed-2026-10-31/report.md)
uses f32 accumulation for the PyTorch BF16 `std_axis` peer before casting the
result back to BF16. All 128 peer comparisons pass in that retained run. The
original failed reports remain available for diagnosis.

The `metal-nonlinear-views-256.json` catalog checks f32/BF16 GELU, erf,
SiLU, and clamp on strided and offset views. Four row sizes and four widths
make 256 cells, each with an independent numerical probe and repeated matched
PyTorch operator and Program timing reports.

The `metal-binary-views-256.json` catalog checks f32/BF16 add, sub, mul, and
div on strided and offset views. Its retained run confirms 192 complete passes
and 64 f32 offset numerical probe failures. The failed cells have explicit
`probe_not_correct` timing statuses and no matched timing links.
The [reviewed rerun](../../benchmark/results/parity-metal-binary-views-256-reviewed-2026-10-31/report.md)
passed all 256 cells after the Compute Metal f32 binary FFI began passing view
offsets to its kernels. The original failure report remains retained.

The `metal-topk-192.json` catalog checks both top-k values and i64 indices for
f32/BF16 inputs, axes zero and one, and dense, strided, and offset views. Its
`affon-topk.ts` worker and `fixtures/topk_fixture.py` reference retain an explicit list
of output shapes and dtypes. The benchmark compares both outputs and includes
materialization of both in each timing sample. The composed Program doubles
the values while preserving the indices. Regenerate the catalog with
`generators/generate-metal-topk-192.py` and rerun it through `run.py`.
The `metal-topk-smallest-192.json` catalog extends the same contract to
`largest=false`, k=1 on axis zero and k=width−1 on axis one, with larger shapes
and the same three layouts. The current worker records a compact hash over every
value and index, plus sampled values and the checksum, so each result element
participates in the numerical parity decision without exceeding the worker's
JSON output limit. Regenerate it with
`generators/generate-metal-topk-smallest-192.py`.

The `metal-rms-norm-views-128.json` catalog exercises f32/BF16 RMS
normalization on inner and outer axes with strided and offset source views.
The independent numerical reference uses an explicit epsilon of `1e-5`, as
does the PyTorch peer; its default epsilon differs. Four row sizes and four
widths provide 128 rerunnable cells with both operator and composed Program
timings. Regenerate it with `generators/generate-metal-rms-norm-views-128.py`.
The `metal-rms-norm-r8-c7-recheck.json` catalog preserves the one-cell
recheck that closed the earlier f32 timing gap after the epsilon correction.

The `metal-scatter-strided-128.json` catalog covers f32/BF16 scatter-add
with row or column strides applied to the base, indices, and updates. The
independent reference uses the logical matrices; both Affon and PyTorch
consume matching physical buffers and views. Two axes, four row sizes, and
four widths produce 128 rerunnable cells. Regenerate it with
`generators/generate-metal-scatter-strided-128.py`.

The `metal-index-select-views-128.json` catalog extends f32/BF16
`index_select` coverage across both axes, strided and offset source views,
four row sizes, four widths, and matching index lengths. The source view and
the i64 index vector are matched between Affon and PyTorch; both output
correctness and repeated operator and Program timing are retained. Regenerate
it with `generators/generate-metal-index-select-views-128.py`.

The `metal-soft-loss-strided-128.json` catalog covers f32/BF16 soft-target
cross entropy and log-softmax negative log likelihood on row-strided or
column-strided logits and target views. Four batch sizes and four class
widths make 128 scalar-loss cells, each checked against the independent
log-sum-exp reference and PyTorch operator and Program timings. Regenerate
it with `generators/generate-metal-soft-loss-strided-128.py`.

The `metal-matmul-rhs-views-128.json` catalog checks f32/BF16 matrix
multiplication with transposed and column-strided right-hand matrices. It
spans four M values, four K values, and two N values. Operator parity checks
the matrix product; Program parity adds a per-column bias. Both retain
repeated PyTorch timing samples. Regenerate it with
`generators/generate-metal-matmul-rhs-views-128.py`.

The `metal-matmul-dense-grid-128.json` catalog covers f32/BF16 dense
matmul at M values 2/8/32/128, K values 5/33/65/127, and N values
7/31/63/129. Each operator comparison checks the product; the composed
Program comparison adds a per-column bias. Regenerate it with
`generators/generate-metal-matmul-dense-grid-128.py`.

The `metal-matmul-lhs-views-128.json` catalog checks f32/BF16 matmul with
transposed and column-strided left matrices. It spans four M values, four K
values, and two N values. Operator parity checks the matrix product;
Program parity adds a per-column bias. The selected operator paths include
transpose and slice views feeding both small and tiled Metal matmul.
Regenerate it with `generators/generate-metal-matmul-lhs-views-128.py`.

The `metal-cast-views-128.json` catalog checks f32 to BF16 and BF16 to f32
casts on strided and offset views. Four row sizes and eight widths cover
31/32/33 and 63/64/65 boundaries. Each cell checks the output dtype and
numerical values, with repeated matched operator and Program timings.
Regenerate it with `generators/generate-metal-cast-views-128.py`.

The `metal-layer-norm-views-128.json` catalog checks f32/BF16 layer norm on
inner and outer axes of strided and offset views. Its first retained run had
one f32 outer-axis 3×5 PyTorch MPS peer mismatch under a relative tolerance
of `1e-6`, while the independent numerical probe passed at `2e-6`. The peer
now uses the same `2e-6` f32 layer norm tolerance. The
[reviewed rerun](../../benchmark/results/parity-metal-layer-norm-views-128-reviewed-2026-10-31/report.md)
passed all 128 probes and both timing tiers; the original failed reports
remain retained. Regenerate the catalog with
`generators/generate-metal-layer-norm-views-128.py`.
The [offset-corrected rerun](../../benchmark/results/parity-metal-layer-norm-views-128-offset-corrected-2026-10-31/report.md)
fixes a catalog interpretation error: inner `_offset` variants under the
`strided` coverage regime had previously selected strided input views. The
Affon workers and Python references now give the explicit `_offset` suffix
precedence. All 128 cells passed in the corrected run, including 32 actual
inner offset views; the previous aggregate reports are archived in
`cell-peer-before-layer-norm-offset-correction-2026-10-31`.

The `metal-inner-reductions-256.json` catalog checks f32/BF16 sum, mean,
min, and max along the inner axis of strided and offset views. Four row
sizes and four widths make 256 cells, each with independent numeric probes
and repeated PyTorch operator and Program timings. Regenerate it with
`generators/generate-metal-inner-reductions-256.py`.

The `metal-transcendental-views-256.json` catalog checks f32/BF16 sin, cos,
tanh, and sigmoid on strided and offset inputs. Four row sizes and four
widths make 256 cells. Its retained batch records independent numerical
probes and repeated matched PyTorch operator and composed Program timings.
Regenerate it with `generators/generate-metal-transcendental-views-256.py`.

The `metal-math-views-192.json` catalog checks f32/BF16 exp, log, and sqrt
on strided and offset views. The first retained run passed all independent
probes but had five f32 exp PyTorch MPS peer misses at both timing tiers under
the peer's `1e-6` relative tolerance. Those probes passed the existing
`2e-6` independent tolerance. The f32 exp peer now uses `2e-6`; the
[reviewed rerun](../../benchmark/results/parity-metal-math-views-192-reviewed-2026-10-31/report.md)
passed all 192 cells and both timing tiers. The original failed reports
remain retained. Regenerate it with `generators/generate-metal-math-views-192.py`.

The `metal-comparison-views-192.json` catalog covers f32/BF16 equality,
less than, and greater than on strided and offset input views. Four row
sizes and four widths produce 192 cells. Both comparison results and
composed Programs return i64 values; each has a numerical check and
repeated matched PyTorch timings. Regenerate it with
`generators/generate-metal-comparison-views-192.py`.

The `metal-program-gaps-13.json` catalog permanently reruns thirteen
supported f32 Metal cells that lacked composed Program evidence. The first
retained run found three incorrect batched matmul plus bias Programs: the
rank-two fused Metal implementation had accepted rank-three operands. The
[reviewed rerun](../../benchmark/results/parity-metal-program-gaps-13-reviewed-2026-10-31/report.md)
passed all thirteen numerical probes and both sets of matched timings after
the fused candidate was restricted to rank-two inputs. The original failed
Program reports remain in `parity-metal-program-gaps-13-2026-10-31`.

The `metal-unknown-canonical-audit.json` catalog audits 36 declared Metal
f32/BF16 canonical gaps. Regenerate it with
`generators/generate-metal-unknown-canonical.py`. Its first retained run passed
32 cells and exposed a topk harness parsing gap in four cells. The corrected
four-cell rerun passed all probe, operator, and composed Program checks. Both
initial failure reports and corrected timing samples are retained. For a
mixed result, `promote.py BATCH --complete-only` promotes only cells with all
three checks passed; incomplete cells stay in the run manifest for review.

The `metal-batched-dense-loss-132.json` catalog covers rank-three f32/BF16
dense cross entropy and log softmax NLL with 132 batch, class and spatial
shapes. Regenerate it with `generators/generate-metal-batched-dense-loss.py`.
The initial retained run records the rank-three wrapper rejection. After
the wrapper gained rank-three support, the reviewed rerun passed numerical
probes and repeated PyTorch operator and composed Program timings for every
cell. The canonical four-cell check is retained separately.

The `metal-batched-topk-130.json` catalog checks rank-three f32/BF16 topk
along both inner axes over 130 batch, row and column shapes. Regenerate it
with `generators/generate-metal-batched-topk.py`. Every cell checks both output
values and indices against an independent reference, then records matched
PyTorch operator and composed Program timings. The canonical two-cell check
is retained separately.

The `metal-batched-scatter-130.json` catalog checks rank-three f32/BF16
scatter add along both inner axes over 130 batch, row, column, and update
shapes. Regenerate it with `generators/generate-metal-batched-scatter.py`.
Every cell retains independent numerical results and repeated matched
PyTorch operator and composed Program timings. The canonical two-cell check
is retained separately.

The 2026-11-02 BF16 scatter rerun covers all 196 linked cells: 64 basic,
64 strided, 65 batched, and three legacy cells from
`metal-unknown-canonical-audit.json`. The Metal kernel now reads the strided
base and applies updates in one dispatch. All 196 numerical, operator, and
Program checks passed; the four regression comparisons reported no
correctness regressions or timing suspects. Prior aggregate reports are
archived in the four `cell-peer-before-bf16-scatter-fused-*-2026-11-02`
directories. The retained invalid-index audit matched its earlier outcome,
including the existing BF16 probe failure.

The `metal-batched-index-select-130.json` catalog checks rank-three f32/BF16
index selection along both inner axes over 130 shapes. Regenerate it with
`generators/generate-metal-batched-index-select.py`. The dedicated
`affon-index-select.ts` worker and `fixtures/index_select_fixture.py` reference retain
independent numerical probes and repeated PyTorch operator and composed
Program timings. The canonical two-cell check is retained separately.

The `metal-ragged-dot-2.json` catalog reruns odd-width f32/BF16 vector dot
with numerical probes and matched operator and Program timings. Regenerate it
with `generators/generate-metal-ragged-dot.py`. The separate
`audit-metal-contract-exclusions.py` script records 38 reviewed exclusions
for matrix dot regimes, reserved gradient operations, and BF16
reduce-to-shape. Its report records the controlling Compute source hashes;
rerunning the audit rejects a changed contract for review.

The 2026-11-02 dot rerun covers all 258 linked f32/BF16 cells across the two
128-cell dot catalogs and the two ragged cells. BF16 vectors shorter than 128
use the direct dot encoder; longer BF16 vectors retain the parallel matmul
vector kernel. f32 vectors of length at least 128 use a parallel reduction,
declared as numerically reassociated. All numerical, operator, and Program
checks passed. The three regression comparisons found no correctness
regressions or timing suspects, and prior aggregate reports are archived in
the `cell-peer-before-dot-optimized-*-2026-11-02` directories. All 258 new
operator and Program execution medians are below the October evidence, but
those cross-session ratios also include environmental variation. The focused
same-session f32 comparison measured ratios of 0.83, 0.70, and 0.45 at widths
128, 257, and 2048 respectively; short widths stayed on the scalar path.

The `metal-remaining-runnable-audit-25.json` catalog reruns 25 declared
Metal f32/BF16 gaps with actual offset and canonical strided views. Regenerate
it with `generators/generate-metal-remaining-runnable-audit.py`. Four f32
variance/std probes remain incorrect; two `contiguous/view_only` paths
materialize and remain diagnostic. Use `promote.py BATCH --complete-only
--exclude-id metal/f32/contiguous/view_only --exclude-id
metal/bf16/contiguous/view_only` to promote only the 19 verified cells.

The `metal-scalar-reduce-to-shape.json` catalog reruns scalar-output f32
reduction with numerical and both timing tiers; regenerate it with
`generators/generate-metal-scalar-reduce-to-shape.py`. The
`metal-invalid-scatter-2.json` catalog retains f32 numerical and timing
parity alongside the BF16 Metal failure; regenerate it with
`generators/generate-metal-invalid-scatter.py`. The separate
`audit-metal-invalid-boundaries.py` runner compares both invalid scatter
cases and both invalid topk cases with PyTorch MPS and retains their raw
outcomes and source hashes.

The `metal-variance-std-corrected-4.json` catalog, generated by
`generators/generate-metal-variance-std-corrected.py`, reruns four f32
whole-tensor variance/std cells. The corrected Metal reduction sums in
64-element chunks. The original failed probe and Kahan rerun remain in
`test/benchmark/results`; the chunked rerun passed independent numerical,
operator, and Program parity with repeated timing samples.

The `metal-binary-boundaries-256.json` catalog, generated by
`generators/generate-metal-binary-boundaries.py`, adds f32/BF16 add, subtract,
multiply, and divide over strided and offset views. Rows 1/7/64/129 and
columns 17/63/127/257 cover new size boundaries. Every cell retains an
independent numerical probe and matched operator and Program timing parity.

The `metal-transcendental-boundaries-256.json` catalog, generated by
`generators/generate-metal-transcendental-boundaries.py`, adds f32/BF16 sin,
cos, tanh, and sigmoid over strided and offset views. Rows 1/7/64/129 and
columns 17/63/127/257 test additional size boundaries. Every cell retains
independent numerical and matched operator and Program timing evidence.

The `metal-inner-reduction-boundaries-256.json` catalog, generated by
`generators/generate-metal-inner-reduction-boundaries.py`, adds f32/BF16
inner-axis sum, mean, min, and max on strided and offset views. Rows
1/7/64/129 and columns 17/63/127/257 cover new reduction widths and
output sizes. Every cell retains independent numerical and matched operator
and Program timing evidence.

The `metal-outer-reduction-boundaries-256.json` catalog, generated by
`generators/generate-metal-outer-reduction-boundaries.py`, adds f32/BF16
outer-axis sum, mean, min, and max on strided and offset views. Rows
1/7/64/129 and columns 17/63/127/257 exercise new reduction lengths and
output widths. Each cell retains numerical, operator, and Program evidence
with repeated timings.

The `metal-matmul-rhs-boundaries-256.json` catalog, generated by
`generators/generate-metal-matmul-rhs-boundaries.py`, adds f32/BF16 matmul
with transposed or strided right operands at tile boundary shapes. Its
retained probe paths split evenly between `metal.matmul.small` and
`metal.matmul.tiled`; every cell also has operator and Program numerical
parity and repeated timing samples.

The 2026-11-02 BF16 single-row decode optimization uses a dedicated encoder
for rank-two matrices with a dense right operand and reduction width below
128. The first generic-kernel branch did not show a timing gain. The broader
encoder candidate slowed some strided right operands, so those keep the
existing kernel. The final rerun promoted 63 affected cells: 16 indexed
decode, 12 left-view, and 35 legacy cases. All passed numerical, operator,
and Program checks; matched regressions found no correctness regressions or
timing suspects. Prior aggregate reports are archived in the three
`cell-peer-before-bf16-m1-dense-*-2026-11-02` directories. The fresh 16-cell
decode comparison measured a 0.96 median operator execution ratio. The
current aggregate has both diagnostic gaps in 5/60 BF16 decode cells and
132/445 BF16 matmul cells. October-to-November ratios also include
environmental variation.

The `metal-softmax-boundaries-256.json` catalog, generated by
`generators/generate-metal-softmax-boundaries.py`, covers f32/BF16 softmax
and log-softmax over inner and outer axes of strided and offset views. Rows
1/7/64/129 and columns 17/63/127/257 exercise new normalization widths.
Each cell retains independent numerical, operator, and Program parity with
repeated timing samples.

The `metal-stat-boundaries-256.json` catalog, generated by
`generators/generate-metal-stat-boundaries.py`, covers f32/BF16 variance
and standard deviation on both axes of strided and offset views. The first
run passed all 256 probes and 252 complete peer comparisons. Four BF16
outer-axis variance comparisons failed under PyTorch MPS BF16 accumulation.
The peer now accumulates BF16 variance in f32 and casts the result back,
matching the existing BF16 standard deviation reference. The four-cell
`metal-stat-peer-reviewed-4.json` rerun, generated by
`generators/generate-metal-stat-peer-reviewed.py`, passed all three tiers.
Both runs and the original failures remain retained.

The `metal-arg-boundaries-256.json` catalog, generated by
`generators/generate-metal-arg-boundaries.py`, covers f32/BF16 argmin and
argmax over inner and outer axes on strided and offset views. Rows
2/7/64/129 and columns 17/63/127/257 exercise new reduction lengths.
Each cell checks i64 index values and retains matched operator and Program
correctness with repeated timing samples.

The `metal-index-boundaries-256.json` catalog, generated by
`generators/generate-metal-index-boundaries.py`, covers f32/BF16 gather and
index-select over axes zero and one on strided and offset inputs. Rows
2/8/32/65 and columns 7/31/63/127 exercise index extent and layout
boundaries. Every cell retains an independent numerical probe and matched
operator and Program correctness with repeated timings.

The `metal-indexed-loss-boundaries-256.json` catalog, generated by
`generators/generate-metal-indexed-loss-boundaries.py`, covers f32/BF16
indexed cross entropy on strided and offset views. Batch sizes 1/7/64/129
and 16 class widths from 8 through 257 exercise dispatch boundaries. Every
cell retains independent numerical, operator, and Program correctness with
repeated timing samples.

The `metal-soft-loss-boundaries-256.json` catalog, generated by
`generators/generate-metal-soft-loss-boundaries.py`, covers f32/BF16 dense
soft-target cross entropy and log-softmax NLL on row- and column-strided
inputs. Rows 2/7/33/65 and class widths 17/32/63/64/66/127/128/129
exercise new loss boundaries. Every cell retains independent numerical,
operator, and Program correctness with repeated timing samples.

The `metal-norm-boundaries-256.json` catalog, generated by
`generators/generate-metal-norm-boundaries.py`, covers f32/BF16 layer norm
and RMS norm on inner and outer axes of strided and offset inputs. Rows
2/7/64/129 and columns 17/63/127/257 add normalization size boundaries.
Every cell retains independent numerical, operator, and Program correctness
with repeated timing samples.

The `metal-comparison-boundaries-384.json` catalog, generated by
`generators/generate-metal-comparison-boundaries.py`, covers f32/BF16
equality, less-than, and greater-than on strided and offset views. Rows
1/7/64/129 and eight widths from 7 through 257 exercise comparison shape
boundaries. Every cell retains independent numerical, operator, and Program
correctness with repeated timing samples.

The remaining two `contiguous/view_only` cells are classified as
unsupported by `audit-metal-contiguous-view-only.py`. Their retained probes
select `metal_contiguous`, which writes a separate output buffer. The audit
records source and probe hashes and has no performance links for the claimed
view-only path; materializing `contiguous` cases have separate parity runs.

The `metal-bf16-matmul-strided-tiled-1.json` case checks a 256×257×256 BF16
matmul with every other right-hand column selected. On the M4, the
[baseline](../../benchmark/results/parity-metal-bf16-matmul-strided-baseline-2026-11-02/report.md)
used the scalar Metal kernel at 2.608 ms operator and 2.249 ms Program median;
the [first tiled candidate](../../benchmark/results/parity-metal-bf16-matmul-strided-tiled-2026-11-02/report.md)
passed numerical and matched peer checks at 0.994 and 0.988 ms. The
[final rebuilt binary](../../benchmark/results/parity-metal-bf16-matmul-strided-final-2026-11-02/report.md)
passed the same checks at 1.425 and 1.456 ms. Runs used 30 iterations, five
samples, and three rounds. These are local timing samples;
the smaller 64×127×64 case varied enough across reruns that it does not
establish a separate speedup. The [final regression comparison](../../benchmark/results/parity-metal-bf16-matmul-strided-final-comparison-2026-11-02/regression.md)
found no correctness regression or changed case contract.

The `metal-reduce-to-shape-scalar-parallel-1.json` catalog checks f32 scalar
reductions of 512 and 640 elements, including an offset view. On the M4,
the [baseline](../../benchmark/results/parity-metal-reduce-to-shape-scalar-baseline-2026-11-02/report.md)
used the serial kernel at 0.776–1.015 ms operator median. The
[final run](../../benchmark/results/parity-metal-reduce-to-shape-scalar-final-2026-11-02/report.md)
selected the cooperative scalar kernel at 0.194–0.196 ms. All three numerical,
operator, and Program checks passed. The
[regression comparison](../../benchmark/results/parity-metal-reduce-to-shape-scalar-comparison-2026-11-02/regression.md)
found no correctness regression or changed case contract. Runs used ten
iterations, five samples, and three rounds. The parallel candidate is marked
reassociated; `exact_only` selects the serial kernel.


## CUDA parity expansion and general loss optimization, 2026-11-05

`cuda-metal-parity.json` mirrors all **11,946** currently supported Metal
contracts, preserving shapes and options. The complete target passed numerical,
operator, and Program checks and was promoted after the scope-bound audit.
The target is registered in the master catalog and coverage policy.

The frozen full-target run has now passed all **11,946 numerical checks**:
5,650 BF16 and 6,296 f32. The retained
[numerical audit](../../benchmark/results/cuda-parity-expansion-2026-11-05/full-numerical-audit.json)
checks exact registered scope, numerical and checksum bounds, and selected
paths. All selected implementations are CUDA kernels or view operations.
The same frozen build is used for the full operator and Program
comparisons with 10 iterations, five samples and three rounds. The corrected
peer verifier and explicit reuse of unaffected reports are described below.

The frozen operator sweep exposed a top-k peer mismatch: BF16 rounding creates
ties in the deterministic input, while native PyTorch `topk` does not guarantee
stable tied indices. Affon's independent reference and CUDA implementation use
lower-index tie-breaking. The failing `[2, 16, 65]`, `k=32` input has ties in
all 32 rows; native PyTorch CPU top-k returns the same values but differs at
398 indices from stable selection. Stable sorting agrees with the independent
reference on every output. Original failure reports remain retained.

The corrected peer uses full stable sorting followed by narrowing for every
top-k input, with `tie_break=lower_index` and
`peer_algorithm=stable_sort_then_narrow` recorded in the timing contract.
It changes neither CUDA dispatch nor fixtures nor tolerances. This comparison
includes the cost of full sorting and must not be described as native PyTorch
top-k performance. Independent CPU checks pass all **408** full-output cases
across both dtypes, all axes, largest/smallest selection, equal inputs, cutoff
ties, and dense/strided/transposed layouts. The full **520-cell** corrected
`cuda-topk-stable-peer.json` GPU operator scope now passes in the corrected
full-target collection, as recorded in the
[top-k checkpoint](../../benchmark/results/cuda-parity-expansion-2026-11-05/topk-full-corrected-operator-checkpoint.json).
All 520 corrected operator and Program checks pass in the complete collection.

The original sweep was subsequently stopped after PyTorch's native scatter-add
asserted on the invalid-index contract and left its in-process CUDA context
unusable. Its 130 failure reports (20 top-k, 110 scatter including downstream
context errors) remain retained. The corrected scatter peer masks invalid
indices and zeroes their updates before scatter-add for every input, matching
Affon's skip-update semantics without cell-specific conditions. Its timing
contract records the masking cost. Independent checks pass all 120 full-output
CPU/CUDA cases across f32/BF16, all axes and three layouts; the formerly failing
GPU invalid-index operator comparison also passes with zero error.

`parity-cuda-metal-corrected-peers-2026-11-05` supersedes the queued standalone
top-k rerun. It retains the complete 11,946-cell scope and the same binary and
10/5/3 protocol. Its `seeded-evidence.json` records the original verifier identity
and hashes for 9,939 unchanged, passing operator reports after checking their
raw rounds and probe contracts. All top-k and scatter-add peers are excluded
from that reuse and were rerun under the corrected semantics. All Program
comparisons were run under the corrected verifier. This is an explicitly
mixed-verifier evidence collection; the retained reports were not all rerun
under one verifier.

The completion auditor automatically verifies `seeded-evidence.json` when
present: original and current run identities, shared build/target/protocol,
source catalog and numerical hashes, unchanged numerical rows, exclusion of
invalidated operations, and every retained source and copied report hash.
`--seed-source` permits auditing a relocated retained source batch without
changing its recorded provenance. All **9,939** reused reports pass this
provenance audit; 11 focused completion-audit tests pass, including tampering,
duplicate reuse, invalidated-operation reuse, and relocated-source checks.

The corrected collection has now passed all **11,946 operator comparisons**.
The [full operator audit](../../benchmark/results/cuda-parity-expansion-2026-11-05/full-operator-audit.json)
checks exact ID coverage, raw errors, checksums, all three rounds and five
samples, selected paths, peer identity, the actual binary hash, and every reused
report's source and destination hash. This includes 2,007 fresh operator reports
and 9,939 unchanged reports with verified provenance. All 11,946 Program
comparisons also passed. The [complete batch audit](../../benchmark/results/cuda-parity-expansion-2026-11-05/full-completion-audit.json)
binds the registered catalog, frozen binary, PyTorch 2.11.0+cu128, all raw
rounds, and seed provenance. Every selected numerical, operator, and Program
path is a CUDA kernel or view operation; no limitations or failure reports were
found. The reviewed evidence was promoted with an
[archive](../../benchmark/results/cuda-parity-expansion-2026-11-05/pre-promotion-archive-2026-11-06/reports-before.json)
of prior ledger entries, existing peer reports, and explicit missing-report
records. The current coverage ledger now supports all 11,946 mirrored cells.

The [full operator diagnostics](../../benchmark/results/cuda-parity-expansion-2026-11-05/full-operator-diagnostics.json)
and [general optimization triage](../../benchmark/results/cuda-parity-expansion-2026-11-05/general-optimization-triage.json)
retain all 11,946 timing comparisons without accepting a performance baseline.
Rank-two reduction-to-shape currently scans every input for each
output; traversing only reduced dimensions is a general candidate for follow-up.
Top-k comparisons include stable sorting and scatter-add comparisons include
invalid-index masking. These peer costs must remain explicit in any performance
interpretation. The retained Program timings span a change in GPU memory use:
the user stopped an unrelated Strata model at about Program case 9,762.
The sweep build remained frozen.

The [142-cell smoke baseline](../../benchmark/results/parity-cuda-metal-smoke-baseline-2026-11-05/report.md)
passed 134 numerical, 133 operator, and 133 Program checks. The
[parity candidate](../../benchmark/results/parity-cuda-metal-smoke-candidate-2026-11-05/report.md)
passed all 142 in each category, with [zero correctness regressions](../../benchmark/results/regression-cuda-metal-smoke-2026-11-05/regression.md).
Changes add native f32/BF16 sine and cosine, native 2/4/8-bit quantized matmul,
and the missing reduction-to-shape recipe. F32 normalization now uses centered
variance and cooperative double accumulation. Quantized matmul, normalization,
and dense losses declare reassociated arithmetic; `exact_only` rejects these
candidates. The [six-cell planner-policy rerun](../../benchmark/results/parity-cuda-metal-numerical-policy-2026-11-05/report.md)
passed numerical, operator, and Program checks on the rebuilt binary.

The [1,350-cell gap regression](../../benchmark/results/parity-cuda-metal-gap-regression-2026-11-05/capability.json)
passed all numerical checks across the changed operation families. After loss
tuning, the [784-cell dense-loss regression](../../benchmark/results/parity-cuda-metal-loss-adaptive-2026-11-05/capability.json)
also passed every numerical check. These broad regressions establish numerical
evidence; the complete target audit now also covers every operator and Program
comparison in these families.

The loss implementation uses runtime shapes, strides, offsets, and dtype.
It assigns independent warps when enough rows exist and shares a block between
power-of-two row teams when there are fewer independent rows. Long rows can
therefore use more than one warp without allocating a temporary loss tensor.
Catalog IDs and fixture values never enter dispatch.

Independent [seed A](../../benchmark/results/parity-cuda-general-adaptive-a-2026-11-05/report.json)
and [seed B](../../benchmark/results/parity-cuda-general-adaptive-b-2026-11-05/report.json)
checks cover random dimensions, signed values, large offsets, near-constant and
zero inputs, ranks two through four for normalization, and dense/offset/strided/
transposed inputs. All **736 numerical checks** compare every output against
PyTorch CPU float64 after quantizing source inputs to f32/BF16. Another **32**
checks confirm the existing public rejection of rank-four losses; these are
reported separately. The [old loss implementation](../../benchmark/results/parity-cuda-general-baseline-2026-11-05/report.json)
missed the unchanged f32 tolerance in three independent cases, in both operator
and composed Program form. The final implementation passes those same inputs.
Raw general-case inputs and outputs are retained as gzip JSON with hashes.

The independent quantized-matmul [seed A](../../benchmark/results/parity-cuda-quantized-general-a-2026-11-05/report.json)
and [seed B](../../benchmark/results/parity-cuda-quantized-general-b-2026-11-05/report.json)
also pass all **768 full-output checks**. They exercise every f32/BF16
activation/metadata pairing, 2/4/8-bit weights, odd groups (including 3 and 31)
that cross packed-word boundaries, random dimensions and values, and
dense/offset/strided/transposed views of all four operands. The expected result
uses independent PyTorch CPU float64 unpacking, affine decoding and matrix
multiplication after rounding source values to their declared dtypes. These
checks add general correctness evidence without adding timing or aggregate
coverage claims.

The [independent evidence retention audit](../../benchmark/results/cuda-parity-expansion-2026-11-05/independent-evidence-retention-audit.json)
binds all four batches to the frozen binary, verifies uncompressed input hashes,
exact operator/Program result identities, finite complete output arrays and
CUDA/view paths, and records hashes of the retained reports and outputs.
It covers 1,504 reported full-output passes and 32 separately classified API
rejections. This is an artifact audit; it does not recompute numerical references.

The [24-case baseline](../../benchmark/results/parity-cuda-loss-general-baseline-2026-11-05/report.md)
and [final comparison](../../benchmark/results/parity-cuda-loss-general-adaptive-2026-11-05/report.md)
use 30 iterations, five samples, and three rounds, covering tiny, wide,
multi-row, offset, strided, and batched cases outside the Metal catalog.
All numerical, operator, and Program checks passed. The
[regression comparison](../../benchmark/results/regression-cuda-loss-general-adaptive-2026-11-05/regression.md)
reports zero correctness regressions, changed contracts, or timing suspects
at the existing diagnostic threshold.

Example f32 cross-entropy execution-phase medians, in milliseconds:

| Logical input / layout | Before | Final | Before / final |
| --- | ---: | ---: | ---: |
| 19 × 257, strided columns | 0.6071 | 0.0651 | 9.32 |
| 11 × 1031, dense | 1.1236 | 0.1146 | 9.80 |
| 1 × 8193, dense | 0.7782 | 0.0881 | 8.84 |
| 5 × 97, offset | 0.0644 | 0.0277 | 2.33 |
| 1 × 3, dense | 0.0125 | 0.0152 | 0.82 |

The tiny f32 case costs about 3 microseconds more. Long single-row BF16
cross-entropy measured 0.0822 → 0.0879 ms; a simple warp-per-row version had
regressed to 0.2274 ms, which motivated cooperative row teams. Those raw
[earlier results](../../benchmark/results/parity-cuda-loss-general-optimized-2026-11-05/report.md)
and their [two timing suspects](../../benchmark/results/regression-cuda-loss-general-first-warp-2026-11-05/regression.md)
are retained. Timing is diagnostic on the RTX 3090. An unrelated process
retained most VRAM during the earlier comparison and part of the full sweep;
it was stopped during the Program sweep. No universal speedup or accepted performance baseline is
claimed. See the [machine-readable summary](../../benchmark/results/cuda-parity-expansion-2026-11-05/summary.json)
and [source/build context](../../benchmark/results/cuda-parity-expansion-2026-11-05/source-context.json).

## General CUDA reduce-to-shape optimization, 2026-11-06

The rank-two CUDA reduction now derives each output's base address from its
runtime shape and traverses only dimensions being summed. It uses input strides
and offsets directly and preserves row-major accumulation order, including f32
rounding after each addition. The change contains no catalog IDs, input-value
branches, or exact-shape dispatch. The scalar and higher-rank paths remain on
their existing implementations.

The [134-cell target](cuda-reduce-to-shape-reviewed-134.json) contains every
reduce-to-shape cell from the mirrored Metal catalog. The controlled
[baseline](../../benchmark/results/parity-cuda-reduce-to-shape-controlled-baseline-2026-11-06/report.md)
and [candidate](../../benchmark/results/parity-cuda-reduce-to-shape-controlled-direct-2026-11-06/report.md)
both pass all 134 numerical, operator, and Program checks with the same 10/5/3
protocol on the RTX 3090 after the Strata model was stopped. The
[regression report](../../benchmark/results/cuda-reduction-direct-2026-11-06/regression/regression.md)
finds zero changed contracts, correctness regressions, or diagnostic timing
suspects. Both batches passed full catalog, binary, PyTorch-version, raw-round,
and selected-path audits.

Two independent random seeds add 256 input cases and 512 full-output checks
across f32/i64, output ranks zero through two, identity and multi-axis
reductions, and dense, offset, strided, and transposed views. Every output
matches an independent sequential reference exactly; all selected reductions
use the CUDA path. Inputs and outputs are retained with hashes.

Across the 133 cells using the changed rank-two path, median execution-phase
speedup is **10.15×** for operator runs and **9.85×** for Program runs. Every
changed cell is faster in these controlled measurements; the scalar cell uses
the unchanged path. These are diagnostic medians for this GPU and protocol, not
an accepted performance baseline or universal speedup. The audited candidate
replaced the 134 current ledger entries, with their prior entries and reports
preserved in a fresh archive. See the [summary](../../benchmark/results/cuda-reduction-direct-2026-11-06/summary.json)
and [source/build identity](../../benchmark/results/cuda-reduction-direct-2026-11-06/source-context.json).

## General CUDA f32 scatter-add optimization, 2026-11-07

The reviewed [197-cell comparison](../../benchmark/results/cuda-scatter-f32-parallel-2026-11-07/README.md)
uses the complete f32 scatter-add subset of the frozen CUDA catalog. It routes
all rank-two views to the existing parallel output-centric CUDA kernel. A
rejected first candidate exposed an offset-address bug; the final version fixes
it and passes all 32 offset cases. The final candidate and baseline each pass all
197 numerical, operator, and Program checks. All 132 changed cells improve in
both execution-phase medians; the previous current reports are archived.

## CUDA medium-axis reduction dispatch, 2026-11-10

The [354-cell catalog](cuda-reduce-axis-threshold-354.json) covers every
reviewed CUDA case whose dispatch changes when the cooperative reduction
threshold moves from 64 to 32 elements: bf16 axis and all reductions, plus
rank-two f32 axis sum, mean, min, max, argmin, and argmax. F32 variance and
standard deviation retain the 64-element threshold. An earlier 404-cell
numerical probe exposed tolerance failures for two outer-axis strided f32
statistics cases; its [capability record](../../benchmark/results/cuda-reduce-axis-threshold-404-rejected-2026-11-10/capability.json)
is retained.

The corrected [candidate](../../benchmark/results/cuda-reduce-axis-threshold-354-candidate-2026-11-10/report.md)
and matched [control](../../benchmark/results/cuda-reduce-axis-threshold-354-control-2026-11-10/report.md)
each pass all 354 numerical, operator, and Program checks on the RTX 3090
with the same 10/5/3 protocol. The candidate's [audit](../../benchmark/results/cuda-reduce-axis-threshold-354-candidate-2026-11-10/AUDIT.json)
binds the exact catalog and binary. The [regression](../../benchmark/results/regression-cuda-reduce-axis-threshold-354-2026-11-10/regression.md)
finds zero changed cases and zero correctness regressions. Its sole timing
suspect is PyTorch timing drift: Affon's Program median for that cell fell to
0.56× control. Across all 354 cases, the largest Affon median ratio is below
1.0 in both operator and Program measurements.

After promotion, reviewed CUDA `candidate_gap` counts fell from 1,801 to
1,638 for operators and from 2,139 to 1,919 for Programs. The 354 previous
ledger entries and reports are retained in
`archive-cuda-reduce-axis-before-threshold-2026-11-10`.
