# Catalog contracts and retained findings

These notes explain the contracts and historical findings behind named parity catalogs. See [the facility guide](../README.md) for running batches and [the artifact index](../../benchmarks/results/PARITY-INDEX.md) for all catalogs and retained reports.

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
The [reviewed rerun](../../benchmarks/results/parity-metal-outer-stat-index-128-reviewed-2026-10-31/report.md)
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
The [reviewed rerun](../../benchmarks/results/parity-metal-binary-views-256-reviewed-2026-10-31/report.md)
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
[reviewed rerun](../../benchmarks/results/parity-metal-layer-norm-views-128-reviewed-2026-10-31/report.md)
passed all 128 probes and both timing tiers; the original failed reports
remain retained. Regenerate the catalog with
`generators/generate-metal-layer-norm-views-128.py`.
The [offset-corrected rerun](../../benchmarks/results/parity-metal-layer-norm-views-128-offset-corrected-2026-10-31/report.md)
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
[reviewed rerun](../../benchmarks/results/parity-metal-math-views-192-reviewed-2026-10-31/report.md)
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
[reviewed rerun](../../benchmarks/results/parity-metal-program-gaps-13-reviewed-2026-10-31/report.md)
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
`test/benchmarks/results`; the chunked rerun passed independent numerical,
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
