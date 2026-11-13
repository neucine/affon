#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
if [ "$(uname -s)" = Linux ]; then
  export PATH="/opt/neucine-release-tools/zig-0.17.0:/opt/neucine-release-tools/bun-1.2.16:/usr/lib/llvm-22/bin:$PATH"
fi
tag="$(buildkite-agent meta-data get release_tag)"
[[ "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$ ]] || {
  echo "Invalid release tag: $tag" >&2
  exit 1
}
git fetch --quiet origin "refs/tags/$tag:refs/tags/$tag"
test "$(git rev-parse HEAD)" = "$(git rev-parse "$tag^{commit}")" || {
  echo "Build commit must equal $tag" >&2
  exit 1
}

for tool in zig bun python3 tar; do
  command -v "$tool" >/dev/null || { echo "Missing release tool: $tool" >&2; exit 1; }
done
test "$(zig version)" = 0.17.0 || { echo "Zig 0.17.0 required" >&2; exit 1; }
test "$(bun --version)" = 1.2.16 || { echo "Bun 1.2.16 required" >&2; exit 1; }

export AFFON_RELEASE_TAG="$tag"
export AFFON_RELEASE_PLATFORM="$1"
"$root/tools/buildkite/release-deps.sh"
if [ "$1" != windows-x64 ]; then
  "$root/tools/release.sh" check
fi
"$root/tools/release.sh" build
"$root/tools/release.sh" smoke-local
if [ "$1" = windows-x64 ]; then
  "$root/dist/release/affon-${tag}-windows-x64.exe" examples/first-program.ts
fi
buildkite-agent artifact upload 'dist/release/*'
