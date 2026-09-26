# SafeTensors interoperability fixtures

These tiny synthetic fixtures contain no model weights or third-party data.
Each file is an eight-byte little-endian header length, compact JSON padded
with spaces to a multiple of eight bytes, and four zero bytes (one f32 zero).

`hf-metadata` contains `__metadata__: {"format":"pt"}` and `weight` with
`dtype: "F32"`, `shape: [1]`, `data_offsets: [0,4]`.
`partial-invalid` contains that weight followed by `bad: {}` to exercise cleanup
after a tensor has already been allocated.

The other files deliberately violate one field: a root array, numeric metadata
value, negative dimension, one-element offset list, mismatched byte count, or
unsupported BF16 dtype. They must produce JavaScript exceptions, not native
traps or runtime teardown failures.
