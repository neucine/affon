#!/usr/bin/env python3

import os
import pathlib
import subprocess
import sys


prefix_value = os.environ.get("LLVM_MINGW_PREFIX")
if not prefix_value:
    raise SystemExit("LLVM_MINGW_PREFIX must point to an LLVM-MinGW installation")

prefix = pathlib.Path(prefix_value)
linker = prefix / "bin" / ("x86_64-w64-mingw32-clang.exe" if os.name == "nt" else "x86_64-w64-mingw32-clang")
if not linker.exists():
    raise SystemExit(f"LLVM-MinGW linker not found: {linker}")

arguments = []
for argument in sys.argv[1:]:
    if argument == "-lgcc":
        continue
    if argument == "-lgcc_eh":
        arguments.append("-lunwind")
        continue
    arguments.append(argument)

raise SystemExit(subprocess.call([str(linker), *arguments]))
