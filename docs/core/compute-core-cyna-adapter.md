# Cyna compute core integration probe

Affon has an isolated Zig adapter for the Cyna `compute-core` matmul, bias,
and GELU C ABI and the f32 quantized matmul C ABI. The quantized path accepts
packed u32 weights with 2, 4, or 8 bits per weight and per-group f32 scales
and biases. The adapter accepts caller-owned slices, checks shapes and
overflow before the call, and maps C ABI status codes to Zig errors. Invalid
input lengths leave the output slice unchanged.

Run `./test/check-compute-core-cyna.sh` on Apple AArch64 with sibling
`compute-core` and an installed Cyna toolchain. The script builds the Cyna
static library, links it into a Zig test, and checks Affon's reference outputs
and invalid-shape behavior, including the selected Zig R01 quantized fixture.

Affon's public `affon:compute` Session still uses the Zig engine. The adapter
is a consumer integration gate for synchronous CPU workloads; it does not
yet provide persistent graph or device handles.
