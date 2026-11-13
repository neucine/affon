#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
if [ "$(uname -s)" = Linux ]; then
  export PATH="/opt/neucine-release-tools/zig-0.17.0:/opt/neucine-release-tools/bun-1.2.16:/usr/lib/llvm-22/bin:$PATH"
fi
tag="$(buildkite-agent meta-data get release_tag)"
[[ "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$ ]] || exit 1
git fetch --quiet origin "refs/tags/$tag:refs/tags/$tag"
test "$(git rev-parse HEAD)" = "$(git rev-parse "$tag^{commit}")" || {
  echo "Build commit must equal $tag" >&2
  exit 1
}

for tool in zig python3 tar gh npx; do
  command -v "$tool" >/dev/null || { echo "Missing publish tool: $tool" >&2; exit 1; }
done
: "${GITHUB_RELEASE_TOKEN:?GITHUB_RELEASE_TOKEN is required}"
: "${CLOUDFLARE_ACCOUNT_ID:?CLOUDFLARE_ACCOUNT_ID is required}"
: "${CLOUDFLARE_API_TOKEN:?CLOUDFLARE_API_TOKEN is required}"
export GH_TOKEN="$GITHUB_RELEASE_TOKEN"
export AFFON_RELEASE_TAG="$tag" AFFON_R2_UPLOAD_TOOL=wrangler

"$root/tools/buildkite/release-deps.sh"
buildkite-agent artifact download 'dist/release/*' .
"$root/tools/release.sh" metadata

version="${tag#v}"
archive="affon-$version.tar.gz"
mkdir -p dist/source package-source verify-source
git archive --format=tar --prefix='affon/' HEAD \
  build.zig build.zig.zon LICENSE README.md THIRD_PARTY_NOTICES.md install.sh install.ps1 src tools packages/@types/affon |
  tar -C verify-source -xf -
git -C ../hao-legacy archive --format=tar --prefix='hao-legacy/' HEAD -- \
  build.zig build.zig.zon include src libs/transpiler | tar -C verify-source -xf -
git -C ../compute archive --format=tar --prefix='compute/' HEAD -- \
  build.zig build.zig.zon README.md src tools | tar -C verify-source -xf -
git -C ../zig-libs archive --format=tar --prefix='zig-libs/' HEAD -- \
  build.zig build.zig.zon README.md src .gitignore | tar -C verify-source -xf -
zig build --build-file verify-source/affon/build.zig -Doptimize=ReleaseSafe
cp -R verify-source/affon verify-source/hao-legacy verify-source/compute verify-source/zig-libs package-source/
tar -C package-source -czf "dist/source/$archive" affon hao-legacy compute zig-libs
buildkite-agent artifact upload 'dist/source/*'

"$root/tools/release.sh" upload-r2
"$root/tools/release.sh" upload-install
"$root/tools/release.sh" smoke-remote
"$root/tools/release.sh" github-release
gh release upload "$tag" "dist/source/$archive" --repo neucine/affon
