#!/usr/bin/env bash
#
# Local release helper for Affon.
#
# Usage:
#   ./tools/release.sh check
#   ./tools/release.sh build
#   ./tools/release.sh smoke-local
#   ./tools/release.sh upload-r2
#   ./tools/release.sh smoke-remote
#   ./tools/release.sh upload-install
#   ./tools/release.sh github-release
#   ./tools/release.sh all
#
# Environment:
#   AFFON_RELEASE_TAG          Release tag to publish (default: v<build.zig.zon version>)
#   AFFON_RELEASE_SKIP_TESTS   1 to skip release tests during check/all
#   AFFON_DOWNLOAD_BASE_URL    Download origin (default: https://downloads.affon.ai)
#   AFFON_R2_ACCOUNT_ID        Cloudflare account id for aws S3-compatible uploads
#   AFFON_R2_BUCKET            R2 bucket name (default: R2_BUCKET or affon-downloads)
#   AFFON_R2_PREFIX            R2 object prefix (default: releases)
#   AFFON_R2_DRY_RUN           1 to print uploads without writing to R2
#   AFFON_R2_UPLOAD_TOOL       aws or wrangler (default: aws)
#   AFFON_GITHUB_REPO          GitHub repo slug for gh release (default: neucine/affon)
#   AWS_ACCESS_KEY_ID          R2 access key for aws uploads
#   AWS_SECRET_ACCESS_KEY      R2 secret key for aws uploads
#   AWS_PROFILE                Optional aws profile
#   CLOUDFLARE_ACCOUNT_ID      Cloudflare account id for wrangler uploads
#   CLOUDFLARE_API_TOKEN       Cloudflare API token for wrangler uploads

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$ROOT/dist/release"
SKIP_TESTS="${AFFON_RELEASE_SKIP_TESTS:-0}"
DOWNLOAD_BASE_URL="${AFFON_DOWNLOAD_BASE_URL:-https://downloads.affon.ai}"
R2_ACCOUNT_ID="${AFFON_R2_ACCOUNT_ID:-${CLOUDFLARE_ACCOUNT_ID:-}}"
R2_BUCKET_NAME="${AFFON_R2_BUCKET:-${R2_BUCKET:-affon-downloads}}"
R2_PREFIX="${AFFON_R2_PREFIX:-releases}"
R2_DRY_RUN="${AFFON_R2_DRY_RUN:-0}"
R2_UPLOAD_TOOL="${AFFON_R2_UPLOAD_TOOL:-aws}"
GITHUB_REPO="${AFFON_GITHUB_REPO:-neucine/affon}"
MODE="${1:-check}"

info() {
  printf 'affon-release: %s\n' "$*"
}

fail() {
  printf 'affon-release: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

run() {
  info "$*"
  "$@"
}

version_string() {
  python3 - <<'PY' "$ROOT/build.zig.zon"
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
match = re.search(r'\.version\s*=\s*"([^"]+)"', text)
if not match:
    raise SystemExit("failed to find .version in build.zig.zon")
print(match.group(1))
PY
}

release_tag() {
  if [ -n "${AFFON_RELEASE_TAG:-}" ]; then
    printf '%s\n' "$AFFON_RELEASE_TAG"
  else
    printf 'v%s\n' "$(version_string)"
  fi
}

host_os() {
  case "$(uname -s)" in
    Darwin) printf 'darwin\n' ;;
    Linux) printf 'linux\n' ;;
    *) fail "unsupported operating system: $(uname -s)" ;;
  esac
}

host_arch() {
  case "$(uname -m)" in
    x86_64|amd64) printf 'x64\n' ;;
    arm64|aarch64) printf 'arm64\n' ;;
    *) fail "unsupported architecture: $(uname -m)" ;;
  esac
}

asset_stem() {
  printf 'affon-%s-%s-%s\n' "$(release_tag)" "$(host_os)" "$(host_arch)"
}

release_url() {
  local name="$1"
  printf '%s/%s/%s/%s?v=%s\n' \
    "${DOWNLOAD_BASE_URL%/}" \
    "${R2_PREFIX%/}" \
    "$(release_tag)" \
    "$name" \
    "$(release_tag)"
}

assert_version_output() {
  local binary="$1"
  local version output
  version="$(version_string)"
  output="$("$binary" --version 2>&1)"
  [ "$output" = "affon $version" ] || fail "binary version mismatch: expected affon $version, got $output"
}

assert_checkout_binary() {
  [ -x "$ROOT/zig-out/bin/affon" ] || fail "missing built binary at zig-out/bin/affon"
  assert_version_output "$ROOT/zig-out/bin/affon"
}

run_release_tests() {
  export RUNTIME_PACKAGE_PATH="${RUNTIME_PACKAGE_PATH:-$ROOT/packages}"
  run zig build test -Doptimize=ReleaseSafe
  run bun x tsc -p test/types/tsconfig.json --noEmit
  run bun x tsc -p packages/@affon/models/tsconfig.json --noEmit
  run bun x tsc -p packages/@affon/tokenizers/tsconfig.json --noEmit
  run bun x tsc -p apps/decoder-lm/tsconfig.json --noEmit
  run "$ROOT/tools/test-runtime.sh" "$ROOT/zig-out/bin/affon"
}

run_checks() {
  need_cmd bun
  need_cmd python3
  need_cmd zig

  info "version: $(version_string)"
  info "tag: $(release_tag)"
  sync_install_version
  require_install_script

  run zig build -Doptimize=ReleaseSafe
  assert_checkout_binary

  if [ "$SKIP_TESTS" != "1" ]; then
    run_release_tests
  fi

  info "checks passed"
}

build_artifacts() {
  local stem package_root stage_dir tarball binary_asset
  need_cmd python3
  need_cmd shasum
  need_cmd tar
  need_cmd zig
  sync_install_version
  require_install_script

  stem="$(asset_stem)"
  tarball="$DIST_DIR/$stem.tar.gz"
  package_root="$DIST_DIR/package"
  stage_dir="$package_root/$stem"
  binary_asset="$DIST_DIR/$stem"

  run zig build -Doptimize=ReleaseFast
  assert_checkout_binary

  rm -rf "$DIST_DIR"
  mkdir -p "$stage_dir"
  cp "$ROOT/zig-out/bin/affon" "$stage_dir/affon"
  cp "$ROOT/zig-out/bin/affon" "$binary_asset"
  chmod 755 "$stage_dir/affon" "$binary_asset"
  [ ! -f "$ROOT/README.md" ] || cp "$ROOT/README.md" "$stage_dir/README.md"
  [ ! -f "$ROOT/LICENSE" ] || cp "$ROOT/LICENSE" "$stage_dir/LICENSE"

  tar -czf "$tarball" -C "$package_root" "$stem"
  rm -rf "$package_root"

  (
    cd "$DIST_DIR"
    shasum -a 256 "$stem" > "$stem.sha256"
    shasum -a 256 "$stem.tar.gz" > "$stem.tar.gz.sha256"
  )

  write_latest_json

  info "built direct binary: $binary_asset"
  info "built artifact: $tarball"
  info "wrote checksums and latest.json in $DIST_DIR"
}

require_artifacts() {
  local stem
  stem="$(asset_stem)"
  [ -f "$DIST_DIR/$stem" ] || build_artifacts
  [ -f "$DIST_DIR/$stem.tar.gz" ] || build_artifacts
  [ -f "$DIST_DIR/latest.json" ] || write_latest_json
  [ -f "$DIST_DIR/$(release_tag).json" ] || write_latest_json
}

require_install_script() {
  local version
  version="$(version_string)"
  [ -f "$ROOT/install.sh" ] || fail "missing canonical installer at install.sh"
  [ -x "$ROOT/install.sh" ] || fail "install.sh must be executable"
  grep -q "AFFON_RELEASE_VERSION=\"\${AFFON_RELEASE_VERSION:-$version}\"" "$ROOT/install.sh" || \
    fail "install.sh release version must match build.zig.zon version $version"
  grep -q 'downloads\.affon\.ai/releases/v${AFFON_RELEASE_VERSION}\.json' "$ROOT/install.sh" || \
    fail "install.sh must default to the AFFON downloads endpoint"
}

sync_install_version() {
  local version
  version="$(version_string)"
  python3 - <<'PY' "$ROOT/install.sh" "$version"
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
version = sys.argv[2]
source = path.read_text()
updated, count = re.subn(
    r'AFFON_RELEASE_VERSION="\$\{AFFON_RELEASE_VERSION:-[^}]+\}"',
    f'AFFON_RELEASE_VERSION="${{AFFON_RELEASE_VERSION:-{version}}}"',
    source,
    count=1,
)
if count != 1:
    raise SystemExit("failed to update AFFON_RELEASE_VERSION in install.sh")
if updated != source:
    path.write_text(updated)
PY
}

write_latest_json() {
  RELEASE_TAG="$(release_tag)" DIST_DIR="$DIST_DIR" R2_PREFIX="$R2_PREFIX" DOWNLOAD_BASE_URL="$DOWNLOAD_BASE_URL" python3 - <<'PY'
import json
import os
import pathlib

tag = os.environ["RELEASE_TAG"]
dist = pathlib.Path(os.environ["DIST_DIR"])
prefix = os.environ["R2_PREFIX"].strip("/")
base_url = f"{os.environ['DOWNLOAD_BASE_URL'].rstrip('/')}/{prefix}/{tag}"
assets = []

for path in sorted(dist.iterdir()):
    if path.is_file() and path.name != "latest.json":
        assets.append({
            "name": path.name,
            "browser_download_url": f"{base_url}/{path.name}?v={tag}",
        })

payload = {
    "tag_name": tag,
    "name": tag,
    "assets": assets,
}

text = json.dumps(payload, indent=2) + "\n"
(dist / "latest.json").write_text(text)
(dist / f"{tag}.json").write_text(text)
PY
}

smoke_archive_path() {
  local archive="$1"
  local stem tmp extracted
  stem="$(asset_stem)"
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/affon-release-smoke.XXXXXX")"
  trap "rm -rf '$tmp'" RETURN

  mkdir -p "$tmp/extract"
  run tar -xzf "$archive" -C "$tmp/extract"
  extracted="$tmp/extract/$stem/affon"
  [ -x "$extracted" ] || fail "archive does not contain executable $stem/affon"
  assert_version_output "$extracted"
  info "smoke passed for $archive"
}

smoke_local() {
  local stem
  require_artifacts
  stem="$(asset_stem)"
  assert_version_output "$DIST_DIR/$stem"
  smoke_archive_path "$DIST_DIR/$stem.tar.gz"
}

upload_with_aws() {
  local file="$1"
  local key="$2"
  local endpoint common

  : "${R2_ACCOUNT_ID:?missing AFFON_R2_ACCOUNT_ID or CLOUDFLARE_ACCOUNT_ID}"
  : "${AWS_ACCESS_KEY_ID:?missing AWS_ACCESS_KEY_ID}"
  : "${AWS_SECRET_ACCESS_KEY:?missing AWS_SECRET_ACCESS_KEY}"

  endpoint="https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com"
  common=(aws --endpoint-url "$endpoint")
  if [ -n "${AWS_PROFILE:-}" ]; then
    common+=(--profile "$AWS_PROFILE")
  fi

  run "${common[@]}" s3 cp "$file" "s3://$R2_BUCKET_NAME/$key"
}

upload_with_wrangler() {
  local file="$1"
  local key="$2"
  local args

  : "${CLOUDFLARE_ACCOUNT_ID:?missing CLOUDFLARE_ACCOUNT_ID}"
  : "${CLOUDFLARE_API_TOKEN:?missing CLOUDFLARE_API_TOKEN}"

  args=(npx --yes wrangler r2 object put "$R2_BUCKET_NAME/$key" --file "$file" --remote)
  if [ "${file%.json}" != "$file" ]; then
    args+=(--content-type application/json)
  fi
  run "${args[@]}"
}

upload_file_to_r2() {
  local file="$1"
  local key="$2"

  if [ "$R2_DRY_RUN" = "1" ]; then
    info "dry-run upload $file -> r2://$R2_BUCKET_NAME/$key"
    return
  fi

  case "$R2_UPLOAD_TOOL" in
    aws)
      need_cmd aws
      upload_with_aws "$file" "$key"
      ;;
    wrangler)
      need_cmd npx
      upload_with_wrangler "$file" "$key"
      ;;
    *)
      fail "unknown AFFON_R2_UPLOAD_TOOL: $R2_UPLOAD_TOOL"
      ;;
  esac
}

upload_r2() {
  local file name key
  require_artifacts

  for file in "$DIST_DIR"/*; do
    [ -f "$file" ] || continue
    name="$(basename "$file")"
    if [ "$name" = "latest.json" ] || [ "$name" = "$(release_tag).json" ]; then
      continue
    fi
    key="${R2_PREFIX%/}/$(release_tag)/$name"
    upload_file_to_r2 "$file" "$key"
  done

  upload_file_to_r2 "$DIST_DIR/latest.json" "${R2_PREFIX%/}/latest.json"
  upload_file_to_r2 "$DIST_DIR/$(release_tag).json" "${R2_PREFIX%/}/$(release_tag).json"
  info "uploaded $(release_tag) assets to R2 bucket $R2_BUCKET_NAME"
}

upload_install() {
  require_install_script
  upload_file_to_r2 "$ROOT/install.sh" "install.sh"
  upload_file_to_r2 "$ROOT/install.sh" "${R2_PREFIX%/}/install.sh"
  info "uploaded install.sh to R2 bucket $R2_BUCKET_NAME"
}

smoke_remote() {
  local stem tmp url
  need_cmd curl
  require_artifacts

  stem="$(asset_stem)"
  url="$(release_url "$stem.tar.gz")"
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/affon-release-remote.XXXXXX")"
  trap "rm -rf '$tmp'" RETURN

  info "downloading $url"
  run curl -fsSL "$url" -o "$tmp/$stem.tar.gz"
  smoke_archive_path "$tmp/$stem.tar.gz"
}

github_release_notes() {
  local tag previous range sha
  tag="$(release_tag)"
  previous="$(git -C "$ROOT" describe --tags --abbrev=0 "$tag^" 2>/dev/null || true)"
  range="$tag"
  if [ -n "$previous" ]; then
    range="$previous..$tag"
  fi
  sha="$(awk '{print $1}' "$DIST_DIR/$(asset_stem).tar.gz.sha256")"

  {
    printf '## Affon %s\n\n' "${tag#v}"
    printf '### Downloads\n\n'
    printf -- '- %s: %s\n' "$(asset_stem)" "$(release_url "$(asset_stem).tar.gz")"
    printf -- '- SHA256: `%s`\n\n' "$sha"
    printf '### Changelog\n\n'
    git -C "$ROOT" log --pretty=format:'- %s (%h)' "$range" 2>/dev/null || printf -- '- No commits found for this release.'
    printf '\n'
    if [ -n "$previous" ]; then
      printf '\nFull diff: https://github.com/%s/compare/%s...%s\n' "$GITHUB_REPO" "$previous" "$tag"
    fi
  }
}

create_github_release() {
  local tag notes
  need_cmd gh
  require_artifacts

  tag="$(release_tag)"
  if gh release view "$tag" --repo "$GITHUB_REPO" >/dev/null 2>&1; then
    fail "github release already exists: $tag"
  fi

  notes="$(mktemp "${TMPDIR:-/tmp}/affon-release-notes.XXXXXX.md")"
  github_release_notes > "$notes"
  run gh release create "$tag" "$DIST_DIR"/* \
    --repo "$GITHUB_REPO" \
    --verify-tag \
    --title "Affon ${tag#v}" \
    --notes-file "$notes"
  rm -f "$notes"
}

case "$MODE" in
  check)
    run_checks
    ;;
  build)
    build_artifacts
    ;;
  smoke-local)
    smoke_local
    ;;
  upload-r2)
    upload_r2
    ;;
  upload-install)
    upload_install
    ;;
  smoke-remote)
    smoke_remote
    ;;
  github-release)
    create_github_release
    ;;
  all)
    run_checks
    build_artifacts
    smoke_local
    upload_r2
    upload_install
    if [ "$R2_DRY_RUN" != "1" ]; then
      smoke_remote
      create_github_release
    else
      info "skipped remote smoke and GitHub release during dry run"
    fi
    ;;
  *)
    fail "unknown mode: $MODE"
    ;;
esac
