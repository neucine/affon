# Cyna compute core integration probe

Affon has an isolated Zig adapter for the Cyna `compute-core` matmul, bias,
and GELU C ABI. The adapter accepts caller-owned f32 slices, checks shapes and
overflow before the call, and maps C ABI status codes to Zig errors. Invalid
input lengths leave the output slice unchanged.

Run `./test/check-compute-core-cyna.sh` on Apple AArch64 with sibling
`compute-core` and an installed Cyna toolchain. The script builds the Cyna
static library, links it into a Zig test, and checks Affon's reference output
and invalid-shape behavior.

Affon's public `affon:compute` Session still uses the Zig engine. The adapter
is a consumer integration gate for one synchronous CPU workload; it does not
yet provide persistent graph or device handles.
