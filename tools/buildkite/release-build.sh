#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
export PATH="$HOME/.cargo/bin:$PATH"
if [ "$(uname -s)" = Linux ]; then
  export PATH="/opt/neucine-release-tools/cyna/bin:/opt/neucine-release-tools/bun/bin:/usr/lib/llvm-22/bin:$PATH"
elif [ "$(uname -s)" = Darwin ]; then
  export PATH="$HOME/.bun/bin:/opt/homebrew/bin:$PATH"
else
  export PATH="$HOME/.bun/bin:$HOME/.local/bin:$PATH"
fi
if [ -d "$HOME/.cyna/0.5.4-beta.local.20261122/bin" ]; then
  export PATH="$HOME/.cyna/0.5.4-beta.local.20261122/bin:$PATH"
elif [ -d "$HOME/.cyna/current/bin" ]; then
  export PATH="$HOME/.cyna/current/bin:$PATH"
fi
platform="${1:?expected fast, darwin-arm64, linux-x64, or windows-x64}"
case "$platform:$(uname -s):$(uname -m)" in
  fast:Darwin:arm64|darwin-arm64:Darwin:arm64|linux-x64:Linux:x86_64|windows-x64:MINGW*:x86_64|windows-x64:MSYS*:x86_64) ;;
  *) echo "Wrong runner for $platform" >&2; exit 1 ;;
esac
test "$(git rev-parse HEAD)" = "${BUILDKITE_COMMIT:?Buildkite commit required}"

for tool in cyna bun cargo python3 tar cc cmake; do
  command -v "$tool" >/dev/null || { echo "Missing release tool: $tool" >&2; exit 1; }
done
test "$(bun --version)" = 1.2.16 || { echo "Bun 1.2.16 required" >&2; exit 1; }

source "$root/tools/buildkite/release-deps.sh"
if [ "$platform" = fast ]; then
  for test in test_model_performance_gate test_coverage_evidence test_capability_probes test_cell_performance_summary; do
    python3 -m unittest discover -s test/parity -p "${test}.py"
  done
  for config in src test/types test/e2e/program/operations packages/@affon/models packages/@affon/tokenizers apps/decoder-lm; do
    bun x tsc -p "$config/tsconfig.json" --noEmit
  done
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) cyna build . --build windows --output build/affon ;;
    *) cyna build . --build default --output build/affon ;;
  esac
  "$root/build/affon" examples/first-program.ts
  exit 0
fi
tag="$(buildkite-agent meta-data get release_tag)"
release_commit="$(buildkite-agent meta-data get release_commit)"
[[ "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$ ]] || exit 1
test "$release_commit" = "$BUILDKITE_COMMIT"
export AFFON_RELEASE_TAG="$tag"
export AFFON_RELEASE_PLATFORM="$platform"
"$root/tools/release.sh" check
"$root/tools/release.sh" build
"$root/tools/release.sh" smoke-local
buildkite-agent artifact upload 'dist/release/*'
