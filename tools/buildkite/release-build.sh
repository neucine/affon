#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
if [ "$(uname -s)" = Linux ]; then
  export PATH="/opt/neucine-release-tools/zig-0.17.0:/opt/neucine-release-tools/bun/bin:/usr/lib/llvm-22/bin:$PATH"
fi
platform="${1:?expected fast, darwin-arm64 or linux-x64}"
case "$platform:$(uname -s):$(uname -m)" in
  fast:Darwin:arm64|darwin-arm64:Darwin:arm64|linux-x64:Linux:x86_64) ;;
  *) echo "Wrong runner for $platform" >&2; exit 1 ;;
esac
test "$(git rev-parse HEAD)" = "${BUILDKITE_COMMIT:?Buildkite commit required}"

for tool in zig bun python3 tar; do
  command -v "$tool" >/dev/null || { echo "Missing release tool: $tool" >&2; exit 1; }
done
test "$(zig version)" = 0.17.0 || { echo "Zig 0.17.0 required" >&2; exit 1; }
test "$(bun --version)" = 1.2.16 || { echo "Bun 1.2.16 required" >&2; exit 1; }

"$root/tools/buildkite/release-deps.sh"
if [ "$platform" = fast ]; then
  zig build test -Doptimize=ReleaseSafe
  zig build -Doptimize=ReleaseSafe
  "$root/zig-out/bin/affon" examples/first-program.ts
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
