#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: apps/decoder-lm/run.sh <name>

Available names:
  helloworld
  tinystories
  wikitext
  wikitext-debug
EOF
}

if [ "${1:-}" = "" ]; then
  usage
  exit 1
fi

name="$1"
case "$name" in
  helloworld|tinystories|wikitext|wikitext-debug)
    ;;
  *)
    printf 'Unknown decoder training example: %s\n\n' "$name" >&2
    usage >&2
    exit 1
    ;;
esac

config="$SCRIPT_DIR/configs/train-decoder-lm-${name}.config.json"
if [ ! -f "$config" ]; then
  printf 'Missing config: %s\n' "$config" >&2
  exit 1
fi

diagnostics_default=off
case "$name" in
  *debug) diagnostics_default=error ;;
esac

AFFON_TRAIN_CONFIG="$config" \
AFFON_NATIVE_STACK_TRACE="${AFFON_NATIVE_STACK_TRACE:-1}" \
AFFON_NN_DIAGNOSTICS="${AFFON_NN_DIAGNOSTICS:-$diagnostics_default}" \
RUNTIME_PACKAGE_PATH="${RUNTIME_PACKAGE_PATH:-$ROOT/packages}" \
exec "$ROOT/zig-out/bin/affon" run "$SCRIPT_DIR/train.ts"
