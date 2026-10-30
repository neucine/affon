#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
AFFON_BIN="${1:-$ROOT/zig-out/bin/affon}"
export RUNTIME_PACKAGE_PATH="${RUNTIME_PACKAGE_PATH:-$ROOT/packages}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

cat > "$TMP_DIR/run.ts" <<'EOF'
import { Session } from 'affon:compute'
const session = new Session({ device: 'cpu' })
const value = session.tensor([2, 3], { dtype: 'f32' })
console.log(`cli-package:${value.shape.join(',')}`)
value.dispose(); session.dispose()
EOF
"$AFFON_BIN" run "$TMP_DIR/run.ts" > "$TMP_DIR/run.out" 2>&1
rg -q 'cli-package:2' "$TMP_DIR/run.out"

cat > "$TMP_DIR/pass.test.ts" <<'EOF'
import { test, expect } from 'std:test'
test('cli test process', () => expect(2 + 3).toBe(5))
EOF
"$AFFON_BIN" test "$TMP_DIR/pass.test.ts" > "$TMP_DIR/test.out" 2>&1
rg -q '1 passed, 0 failed' "$TMP_DIR/test.out"

mkdir -p "$TMP_DIR/nested/deeper"
cat > "$TMP_DIR/nested/deeper/discovery.test.ts" <<'EOF'
import { test, expect } from 'std:test'
test('nested CLI discovery', () => expect('nested').toBe('nested'))
EOF
"$AFFON_BIN" test "$TMP_DIR/nested" > "$TMP_DIR/nested.out" 2>&1
rg -q '1 passed, 0 failed' "$TMP_DIR/nested.out"
rg -q 'discovery.test.ts' "$TMP_DIR/nested.out"

cat > "$TMP_DIR/package-fail.ts" <<'EOF'
import 'affon:missing-package'
EOF
if "$AFFON_BIN" run "$TMP_DIR/package-fail.ts" > "$TMP_DIR/package-fail.out" 2>&1; then
  echo 'affon run unexpectedly resolved a missing package' >&2
  exit 1
fi
rg -q 'missing-package' "$TMP_DIR/package-fail.out"

cat > "$TMP_DIR/fail.ts" <<'EOF'
throw new Error('cli-expected-failure')
EOF
if "$AFFON_BIN" run "$TMP_DIR/fail.ts" > "$TMP_DIR/fail.out" 2>&1; then
  echo 'affon run unexpectedly succeeded for a throwing script' >&2
  exit 1
fi
rg -q 'cli-expected-failure' "$TMP_DIR/fail.out"

if "$AFFON_BIN" run "$TMP_DIR/missing.ts" > "$TMP_DIR/missing.out" 2>&1; then
  echo 'affon run unexpectedly succeeded for a missing file' >&2
  exit 1
fi
rg -q 'FileNotFound' "$TMP_DIR/missing.out"

cat > "$TMP_DIR/fail.test.ts" <<'EOF'
import { test, expect } from 'std:test'
test('cli failing test process', () => expect(2 + 3).toBe(6))
EOF
if "$AFFON_BIN" test "$TMP_DIR/fail.test.ts" > "$TMP_DIR/fail-test.out" 2>&1; then
  echo 'affon test unexpectedly succeeded for a failing assertion' >&2
  exit 1
fi
rg -q '0 passed, 1 failed' "$TMP_DIR/fail-test.out"

cat > "$TMP_DIR/malformed.test.ts" <<'EOF'
import { test } from 'std:test'
test('malformed', () => {
EOF
if "$AFFON_BIN" test "$TMP_DIR/malformed.test.ts" > "$TMP_DIR/malformed.out" 2>&1; then
  echo 'affon test unexpectedly succeeded for malformed source' >&2
  exit 1
fi
rg -q 'malformed.test.ts' "$TMP_DIR/malformed.out"

cat > "$TMP_DIR/async-fail.ts" <<'EOF'
Promise.reject(new Error('cli-async-failure'))
EOF
if "$AFFON_BIN" run "$TMP_DIR/async-fail.ts" > "$TMP_DIR/async-fail.out" 2>&1; then
  echo 'affon run unexpectedly succeeded for an unhandled rejection' >&2
  exit 1
fi
rg -q 'cli-async-failure' "$TMP_DIR/async-fail.out"

"$AFFON_BIN" run "$TMP_DIR/run.ts" > "$TMP_DIR/recovery.out" 2>&1
rg -q 'cli-package:2' "$TMP_DIR/recovery.out"
echo 'CLI process e2e passed: run, test, failure exits, package resolution, recovery'
