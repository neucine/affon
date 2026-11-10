#!/bin/sh
set -eu

affon_root=$(cd "$(dirname "$0")/.." && pwd -P)
core_root=$(cd "$affon_root/../compute-core" && pwd -P)
cyna_bin=$(command -v cyna)
cyna_root=$(cd "$(dirname "$cyna_bin")/.." && pwd -P)
runtime_libs="$cyna_root/targets/aarch64-apple-darwin/lib"

cyna build "$core_root"
zig test "$affon_root/src/support/compute_core_cyna_smoke.zig" \
    -L "$core_root/build" -lcompute-core \
    -L "$runtime_libs" -lcyna_std -lcyna -lm -lc
