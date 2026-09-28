# SafeTensors interoperability fixtures

These tiny synthetic fixtures contain no model weights or third-party data.
Each file is an eight-byte little-endian header length, compact JSON padded
with spaces to a multiple of eight bytes, and a tiny tensor payload (usually four zero bytes for one f32 zero).

`hf-metadata` contains `__metadata__: {"format":"pt"}` and `weight` with
`dtype: "F32"`, `shape: [1]`, `data_offsets: [0,4]`.
`partial-invalid` contains that weight followed by `bad: {}` to exercise cleanup
after a tensor has already been allocated.

The other files deliberately violate one field: a root array, numeric metadata
value, negative dimension, one-element offset list, mismatched byte count, or
unsupported F16 dtype. They must produce JavaScript exceptions, not native
traps or runtime teardown failures.

`bf16` stores exact bit patterns for signed zeros, ordinary values, a subnormal,
maximum finite value, infinities and NaN. Loading widens these to f32.
`bf16-wrong-size` stores four bytes for a one-element BF16 tensor and must fail.

`bf16-chunks` contains 32,768 BF16 ones and one final two. It exercises a full
64 KiB conversion chunk plus a partial tail, guarding against count overflow.
