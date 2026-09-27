#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AFFON_BIN="${1:-$ROOT/zig-out/bin/affon}"

export RUNTIME_PACKAGE_PATH="${RUNTIME_PACKAGE_PATH:-$ROOT/packages}"

compute_targets=()
case "$(uname -s)" in
  Darwin)
    compute_targets+=("$ROOT/test/e2e/compute")
    ;;
  Linux)
    # The two mixed files below contain CPU assertions plus unconditional Metal
    # allocation checks. macOS covers them in full until those cases are split.
    for test_file in "$ROOT"/test/e2e/compute/*.test.ts; do
      case "$(basename "$test_file")" in
        shape-contracts.test.ts|shape-ops.test.ts) continue ;;
      esac
      compute_targets+=("$test_file")
    done
    compute_targets+=("$ROOT/test/cuda")
    ;;
  *)
    echo "affon-runtime-tests: unsupported operating system: $(uname -s)" >&2
    exit 1
    ;;
esac

"$AFFON_BIN" test \
  "${compute_targets[@]}" \
  "$ROOT/test/e2e/nn" \
  "$ROOT/test/e2e/dataset" \
  "$ROOT/test/e2e/checkpoint"
"$AFFON_BIN" test "$ROOT/packages/@affon/models/test"
"$AFFON_BIN" test "$ROOT/packages/@affon/tokenizers/test"
"$AFFON_BIN" test "$ROOT/apps/decoder-lm/test"
