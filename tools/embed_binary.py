#!/usr/bin/env python3

import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: embed_binary.py <input> <output> <symbol>", file=sys.stderr)
        return 1

    src = pathlib.Path(sys.argv[1])
    dst = pathlib.Path(sys.argv[2])
    symbol = sys.argv[3]
    data = src.read_bytes()

    lines = [
        "#pragma once",
        "#include <stddef.h>",
        f"static const unsigned char {symbol}[] = {{",
    ]
    for i in range(0, len(data), 12):
        chunk = data[i : i + 12]
        lines.append("  " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",")
    lines.append("};")
    lines.append(f"static const size_t {symbol}_len = sizeof({symbol});")
    lines.append("")

    dst.write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
