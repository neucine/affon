#!/usr/bin/env python3
"""Read an affine MLX checkpoint without modifying it; test sampled rows on CPU/Metal."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--affon", type=Path, default=Path(__file__).resolve().parents[1] / "zig-out/bin/affon")
    parser.add_argument("--tensor", help="Packed weight name; defaults to the first U32 .weight with scales/biases")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--matrix", action="store_true", help="Validate the packed matrix path with M128 and 64 spread output rows")
    args = parser.parse_args()
    if args.group_size <= 0:
        parser.error('--group-size must be positive')
    with args.checkpoint.open("rb") as checkpoint:
        header_size = struct.unpack("<Q", checkpoint.read(8))[0]
        header = json.loads(checkpoint.read(header_size))
        name = args.tensor or next(name for name, spec in header.items()
                                   if name.endswith(".weight") and spec.get("dtype") == "U32"
                                   and name[:-7] + ".scales" in header and name[:-7] + ".biases" in header)
        prefix = name[:-7]
        weight, scale, bias = (header[key] for key in [name, prefix + ".scales", prefix + ".biases"])
        assert weight["dtype"] == "U32" and scale["dtype"] == bias["dtype"] == "BF16"
        n, words = weight["shape"]
        k = words * 4  # This parity adapter deliberately selects the 8-bit scheme.
        groups = k // args.group_size
        assert args.group_size > 0 and k % args.group_size == 0
        assert scale["shape"] == bias["shape"] == [n, groups]
        selected = [i * (n - 1) // 63 for i in range(64)] if args.matrix else sorted(set([0, min(1, n - 1), n - 1]))
        if args.matrix and n < 64:
            parser.error("--matrix needs at least 64 checkpoint output rows")

        def rows(spec, width, code):
            result = []
            for row in selected:
                checkpoint.seek(8 + header_size + spec["data_offsets"][0] + row * width * struct.calcsize(code))
                values = struct.unpack("<" + code * width, checkpoint.read(width * struct.calcsize(code)))
                if code == "H":
                    values = [struct.unpack("<f", struct.pack("<I", value << 16))[0] for value in values]
                result.append(list(values))
            return result

        packed, scales, biases = rows(weight, words, "I"), rows(scale, groups, "H"), rows(bias, groups, "H")
    inputs = [[((column * 7 + row * 3) % 31 - 15) / 16 for column in range(k)] for row in range(128 if args.matrix else 3)]
    # Independent oracle uses little-endian bytes, rather than U32 shifts.
    decoded = [struct.pack("<" + "I" * words, *row) for row in packed]
    expected = [[sum(x[column] * (scales[out][column // args.group_size] * decoded[out][column]
                                  + biases[out][column // args.group_size]) for column in range(k))
                 for out in range(len(selected))] for x in inputs]
    data = json.dumps(dict(inputs=inputs, packed=packed, scales=scales, biases=biases, expected=expected))
    source = '''import { test, expect } from 'std:test'
import { Session } from 'affon:compute'
import { quantized_matmul } from 'affon:ops'
const data = DATA
for (const device of ['cpu', 'metal'] as const) test(`checkpoint parity ${device}`, () => {
  const session = new Session({ device })
  const x = session.tensor(data.inputs)
  const w = session.tensor(data.packed, { dtype: 'u32' })
  const s = session.tensor(data.scales, { dtype: 'bf16' })
  const b = session.tensor(data.biases, { dtype: 'bf16' })
  const y = quantized_matmul(x, w, s, b, { bits: 8, group_size: GROUP })
  const result = y.to_array() as number[][]
  for (let row = 0; row < result.length; row++) for (let col = 0; col < result[row].length; col++) {
    const expected = data.expected[row][col]
    expect(Math.abs(result[row][col] - expected) <= 1e-4 + 1e-4 * Math.abs(expected)).toBe(true)
  }
  y.dispose(); b.dispose(); s.dispose(); w.dispose(); x.dispose(); session.dispose()
})
'''.replace("DATA", data).replace("GROUP", str(args.group_size))
    print(json.dumps(dict(tensor=name, weight_shape=weight["shape"], parameter_shape=scale["shape"],
                          sampled_rows=selected, bits=8, group_size=args.group_size)), flush=True)
    with tempfile.TemporaryDirectory(prefix="affon-quantized-parity-") as directory:
        test = Path(directory) / "checkpoint.test.ts"
        test.write_text(source)
        subprocess.run([str(args.affon.resolve()), "test", str(test)], check=True)


if __name__ == "__main__":
    main()
