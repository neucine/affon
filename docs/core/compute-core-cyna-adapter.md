# Cyna compute core integration probe

Affon has an isolated Zig adapter for the Cyna `compute-core` matmul, bias,
and GELU C ABI, the f32 quantized matmul C ABI, rank-two f32 reductions, and
a synchronous shaped f32 graph ABI. The graph adapter accepts typed input
shapes and steps, packs the C arrays, and supports selected outputs of
different shapes. Its current operations are binary arithmetic, abs, neg,
ReLU, sign, exp, log, sqrt, sigmoid, SiLU, tanh, erf, GELU, sin, cos, matmul,
whole or axis sum and mean, and axis softmax or log softmax. The caller supplies a flat
output buffer with the total selected element count. Rank-zero inputs have an
empty shape and one value. Callers can choose the baseline or memory compiler
goal; the memory goal can fuse eligible multiply plus add steps.
The reduction adapter covers whole or axis sum, mean, min, max, population
variance, and standard deviation. The quantized path accepts
packed u32 weights with 2, 4, or 8 bits per weight and per-group f32 scales
and biases. The adapter accepts caller-owned slices, checks shapes and
overflow before the call, and maps C ABI status codes to Zig errors. Invalid
input lengths leave the output slice unchanged.

Run `./test/check-compute-core-cyna.sh` on Apple AArch64 with sibling
`compute-core` and an installed Cyna toolchain. The script builds the Cyna
static library, links it into a Zig test, and checks Affon's reference outputs
and invalid-shape behavior, including the selected Zig R01 quantized fixture
whole or axis reduction values, and a matmul followed by two reductions.

Affon's public `affon:core` Session still uses the Zig engine. The adapter
is a consumer integration gate for synchronous CPU workloads; it does not
yet provide persistent graph or device handles.
