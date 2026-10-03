#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AFFON_BIN="${1:-$ROOT/zig-out/bin/affon}"

export RUNTIME_PACKAGE_PATH="${RUNTIME_PACKAGE_PATH:-$ROOT/packages}"

"$AFFON_BIN" test \
  "$ROOT/test/e2e/program" \
  "$ROOT/test/e2e/dataset" \
  "$ROOT/test/e2e/checkpoint"
"$AFFON_BIN" test "$ROOT/packages/@affon/models/test"
"$AFFON_BIN" test "$ROOT/packages/@affon/tokenizers/test"
"$AFFON_BIN" test \
  "$ROOT/packages/@affon/huggingface/test" \
  "$ROOT/packages/@affon/onnx/test" \
  "$ROOT/apps/hf-inference/tests"
"$AFFON_BIN" test "$ROOT/apps/decoder-lm/test"
