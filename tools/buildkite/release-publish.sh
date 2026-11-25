#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
test "$(uname -s)" = Darwin
export PATH="/opt/homebrew/bin:$PATH"
tag="$(buildkite-agent meta-data get release_tag)"
[[ "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$ ]] || exit 1
test "$(git rev-parse HEAD)" = "${BUILDKITE_COMMIT:?Buildkite commit required}"

for tool in python3 tar gh npx; do
  command -v "$tool" >/dev/null || { echo "Missing publish tool: $tool" >&2; exit 1; }
done
: "${GITHUB_RELEASE_TOKEN:?GITHUB_RELEASE_TOKEN is required}"
: "${CLOUDFLARE_ACCOUNT_ID:?CLOUDFLARE_ACCOUNT_ID is required}"
: "${CLOUDFLARE_API_TOKEN:?CLOUDFLARE_API_TOKEN is required}"
export GH_TOKEN="$GITHUB_RELEASE_TOKEN"
export AFFON_RELEASE_TAG="$tag" AFFON_R2_UPLOAD_TOOL=wrangler
export AFFON_SITES_REPO="${AFFON_SITES_REPO:-/Users/chao.yang/Private/sites}"
export AFFON_SITE_HAO_REPO="${AFFON_SITE_HAO_REPO:-/Users/chao.yang/Private/hao}"

source "$root/tools/buildkite/release-deps.sh"
rm -rf dist/release
buildkite-agent artifact download 'dist/release/*' . --step merged-collect
python3 tools/buildkite/ci.py publish-check
"$root/tools/release.sh" metadata

version="${tag#v}"
archive="affon-$version.tar.gz"
mkdir -p dist/source package-source verify-source
git archive --format=tar --prefix='affon/' HEAD |
  tar -C verify-source -xf -
git -C ../hao archive --format=tar --prefix='hao/' HEAD |
  tar -C verify-source -xf -
git -C ../compute-core archive --format=tar --prefix='compute-core/' HEAD |
  tar -C verify-source -xf -
cp -R verify-source/affon verify-source/hao verify-source/compute-core package-source/
tar -C package-source -czf "dist/source/$archive" affon hao compute-core
buildkite-agent artifact upload 'dist/source/*'

"$root/tools/release.sh" upload-r2
"$root/tools/release.sh" upload-install
"$root/tools/release.sh" smoke-remote
"$root/tools/release.sh" github-release
if gh release view "$tag" --repo neucine/affon --json assets --jq '.assets[].name' | grep -Fxq "$archive"; then
  download="$(mktemp -d)"
  gh release download "$tag" --repo neucine/affon --pattern "$archive" --dir "$download"
  mkdir "$download/expected" "$download/published"
  tar -xzf "dist/source/$archive" -C "$download/expected"
  tar -xzf "$download/$archive" -C "$download/published"
  diff -qr "$download/expected" "$download/published"
else
  gh release upload "$tag" "dist/source/$archive" --repo neucine/affon
fi
git -C "$AFFON_SITES_REPO" pull --ff-only origin main
"$root/tools/release.sh" sync-site
