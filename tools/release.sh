#!/usr/bin/env bash
#
# Local release helper for Affon.
#
# Usage:
#   ./tools/release.sh check
#   ./tools/release.sh build
#   ./tools/release.sh smoke-local
#   ./tools/release.sh metadata
#   ./tools/release.sh upload-r2
#   ./tools/release.sh smoke-remote
#   ./tools/release.sh upload-install
#   ./tools/release.sh github-release
#   ./tools/release.sh sync-site
#   ./tools/release.sh all
#
# Environment:
#   AFFON_RELEASE_TAG          Release tag to publish (default: v<build.zig.zon version>)
#   AFFON_RELEASE_SKIP_TESTS   1 to skip release tests during check/all
#   AFFON_RELEASE_PLATFORM     Override detected asset platform
#   AFFON_WINDOWS_GNU_LIBUNWIND Path to llvm-mingw x86_64 libunwind.a
#   AFFON_DOWNLOAD_BASE_URL    Download origin (default: https://downloads.affon.ai)
#   AFFON_R2_ACCOUNT_ID        Cloudflare account id for aws S3-compatible uploads
#   AFFON_R2_BUCKET            R2 bucket name (default: R2_BUCKET or affon-downloads)
#   AFFON_R2_PREFIX            R2 object prefix (default: releases)
#   AFFON_R2_DRY_RUN           1 to print uploads without writing to R2
#   AFFON_R2_UPLOAD_TOOL       aws or wrangler (default: aws)
#   AFFON_GITHUB_REPO          GitHub repo slug for gh release (default: neucine/affon)
#   AFFON_SITES_REPO           Sites checkout (default: sibling sites directory)
#   AFFON_SITE_HAO_REPO        Pinned Hao checkout (default: sibling hao-legacy directory)
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
SITES_REPO="${AFFON_SITES_REPO:-$(dirname "$ROOT")/sites}"
SITE_HAO_REPO="${AFFON_SITE_HAO_REPO:-$(dirname "$ROOT")/hao-legacy}"
MODE="${1:-check}"
RELEASE_PLATFORM="${AFFON_RELEASE_PLATFORM:-}"

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

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1"
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1"
  else
    fail "missing required command: shasum or sha256sum"
  fi
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
    MINGW*|MSYS*|CYGWIN*) printf 'windows\n' ;;
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
  printf 'affon-%s-%s\n' "$(release_tag)" "$(release_platform)"
}

release_platform() {
  if [ -n "$RELEASE_PLATFORM" ]; then
    printf '%s\n' "$RELEASE_PLATFORM"
  else
    printf '%s-%s\n' "$(host_os)" "$(host_arch)"
  fi
}

is_windows_release() {
  [ "$(release_platform)" = "windows-x64" ]
}

binary_basename() {
  if is_windows_release; then
    printf 'affon.exe\n'
  else
    printf 'affon\n'
  fi
}

checkout_binary() {
  printf '%s/zig-out/bin/%s\n' "$ROOT" "$(binary_basename)"
}

binary_asset_name() {
  if is_windows_release; then
    printf '%s.exe\n' "$(asset_stem)"
  else
    printf '%s\n' "$(asset_stem)"
  fi
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
  local binary
  binary="$(checkout_binary)"
  [ -x "$binary" ] || fail "missing built binary at $binary"
  if is_windows_release && [ "$(host_os)" != "windows" ]; then
    need_cmd file
    file "$binary" | grep -q 'PE32+ executable' || fail "built binary is not a Windows x64 executable"
    return
  fi
  assert_version_output "$binary"
}

run_release_tests() {
  export RUNTIME_PACKAGE_PATH="${RUNTIME_PACKAGE_PATH:-$ROOT/packages}"
  run zig build test -Doptimize=ReleaseSafe
  run bun x tsc -p src/tsconfig.json --noEmit
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
  [ "$(release_tag)" = "v$(version_string)" ] || fail "release tag must match build.zig.zon"
  require_install_script

  run zig build -Doptimize=ReleaseSafe
  assert_checkout_binary

  if [ "$SKIP_TESTS" != "1" ]; then
    run_release_tests
  fi

  info "checks passed"
}

build_artifacts() {
  local stem package_root stage_dir tarball binary_asset binary_name binary_source
  local -a build_args
  need_cmd python3
  need_cmd tar
  need_cmd zig
  [ "$(release_tag)" = "v$(version_string)" ] || fail "release tag must match build.zig.zon"
  require_install_script

  stem="$(asset_stem)"
  tarball="$DIST_DIR/$stem.tar.gz"
  package_root="$DIST_DIR/package"
  stage_dir="$package_root/$stem"
  binary_name="$(binary_basename)"
  binary_source="$(checkout_binary)"
  binary_asset="$DIST_DIR/$(binary_asset_name)"

  build_args=(-Doptimize=ReleaseFast)
  if is_windows_release; then
    build_args+=(-Dtarget=x86_64-windows-gnu)
    if [ -n "${AFFON_WINDOWS_GNU_LIBUNWIND:-}" ]; then
      build_args+=("-Dwindows-gnu-libunwind=$AFFON_WINDOWS_GNU_LIBUNWIND")
    fi
  fi
  run zig build "${build_args[@]}"
  assert_checkout_binary

  rm -rf "$DIST_DIR"
  mkdir -p "$stage_dir"
  cp "$binary_source" "$stage_dir/$binary_name"
  cp "$binary_source" "$binary_asset"
  chmod 755 "$stage_dir/$binary_name" "$binary_asset"
  [ ! -f "$ROOT/README.md" ] || cp "$ROOT/README.md" "$stage_dir/README.md"
  [ ! -f "$ROOT/LICENSE" ] || cp "$ROOT/LICENSE" "$stage_dir/LICENSE"

  tar -czf "$tarball" -C "$package_root" "$stem"
  rm -rf "$package_root"

  (
    cd "$DIST_DIR"
    direct_binary="$(basename "$binary_asset")"
    sha256_file "$direct_binary" > "$direct_binary.sha256"
    sha256_file "$stem.tar.gz" > "$stem.tar.gz.sha256"
  )

  info "built direct binary: $binary_asset"
  info "built artifact: $tarball"
  info "wrote checksums in $DIST_DIR"
}

require_artifacts() {
  local stem binary_asset
  stem="$(asset_stem)"
  binary_asset="$(binary_asset_name)"
  [ -f "$DIST_DIR/$binary_asset" ] || build_artifacts
  [ -f "$DIST_DIR/$stem.tar.gz" ] || build_artifacts
}

require_install_script() {
  local version
  version="$(version_string)"
  [ -f "$ROOT/install.sh" ] || fail "missing canonical installer at install.sh"
  [ -x "$ROOT/install.sh" ] || fail "install.sh must be executable"
  [ -f "$ROOT/install.ps1" ] || fail "missing PowerShell installer at install.ps1"
  grep -q "AFFON_RELEASE_VERSION=\"\${AFFON_RELEASE_VERSION:-$version}\"" "$ROOT/install.sh" || \
    fail "install.sh release version must match build.zig.zon version $version"
  grep -q "else { \"$version\" }" "$ROOT/install.ps1" || \
    fail "install.ps1 release version must match build.zig.zon version $version"
  grep -q 'downloads\.affon\.ai/releases/v${AFFON_RELEASE_VERSION}\.json' "$ROOT/install.sh" || \
    fail "install.sh must default to the AFFON downloads endpoint"
  grep -q 'downloads\.affon\.ai/releases/v$ReleaseVersion\.json' "$ROOT/install.ps1" || \
    fail "install.ps1 must default to the AFFON downloads endpoint"
}

write_latest_json() {
  verify_release_set
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
    if path.is_file() and path.name not in ("latest.json", f"{tag}.json"):
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

verify_release_set() {
  local tag platform stem binary expected actual
  tag="$(release_tag)"
  for platform in darwin-arm64 linux-x64 windows-x64; do
    stem="affon-$tag-$platform"
    binary="$stem"
    if [ "$platform" = "windows-x64" ]; then binary="$stem.exe"; fi
    for file in "$binary" "$stem.tar.gz"; do
      [ -s "$DIST_DIR/$file" ] || fail "missing release asset: $file"
      [ -s "$DIST_DIR/$file.sha256" ] || fail "missing release checksum: $file.sha256"
      expected="$(awk 'NR == 1 { print $1 }' "$DIST_DIR/$file.sha256")"
      [[ "$expected" =~ ^[0-9a-fA-F]{64}$ ]] || fail "invalid checksum: $file.sha256"
      actual="$(sha256_file "$DIST_DIR/$file" | awk '{ print $1 }')"
      [ "$actual" = "$expected" ] || fail "checksum mismatch: $file"
    done
  done
}

smoke_archive_path() {
  local archive="$1"
  local stem tmp extracted binary_name
  stem="$(asset_stem)"
  binary_name="$(binary_basename)"
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/affon-release-smoke.XXXXXX")"
  trap "rm -rf '$tmp'" RETURN

  mkdir -p "$tmp/extract"
  run tar -xzf "$archive" -C "$tmp/extract"
  extracted="$tmp/extract/$stem/$binary_name"
  [ -x "$extracted" ] || fail "archive does not contain executable $stem/$binary_name"
  assert_version_output "$extracted"
  info "smoke passed for $archive"
}

smoke_local() {
  local stem binary_asset
  require_artifacts
  stem="$(asset_stem)"
  binary_asset="$(binary_asset_name)"
  assert_version_output "$DIST_DIR/$binary_asset"
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
  write_latest_json

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
  upload_file_to_r2 "$ROOT/install.ps1" "install.ps1"
  upload_file_to_r2 "$ROOT/install.ps1" "${R2_PREFIX%/}/install.ps1"
  info "uploaded installers to R2 bucket $R2_BUCKET_NAME"
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

sync_site() {
  local version pinned_hao current_hao pinned_hao_commit site_head remote_head hosted_installer hosted_home
  need_cmd npm
  need_cmd gh
  [ -d "$SITES_REPO/sites/affon-ai" ] || fail "missing sites checkout at $SITES_REPO"
  [ -d "$SITE_HAO_REPO" ] || fail "missing Hao checkout at $SITE_HAO_REPO"
  version="$(version_string)"
  [ "$(release_tag)" = "v$version" ] || fail "release tag must match build.zig.zon"
  gh release view "$(release_tag)" --repo "$GITHUB_REPO" >/dev/null || fail "publish the GitHub release before syncing the site"

  pinned_hao="$(awk '/repository: neucine\/hao/ { found=1 } found && /ref:/ { print $2; exit }' "$ROOT/.github/workflows/release.yml")"
  current_hao="$(git -C "$SITE_HAO_REPO" rev-parse HEAD)"
  pinned_hao_commit="$(git -C "$SITE_HAO_REPO" rev-parse "$pinned_hao")"
  [ -n "$pinned_hao_commit" ] && [ "$current_hao" = "$pinned_hao_commit" ] || \
    fail "site Hao checkout must match release ref $pinned_hao ($pinned_hao_commit), got $current_hao"

  [ -z "$(git -C "$SITES_REPO" status --porcelain)" ] || fail "sites checkout has local changes: $SITES_REPO"
  run git -C "$SITES_REPO" fetch origin main
  site_head="$(git -C "$SITES_REPO" rev-parse HEAD)"
  remote_head="$(git -C "$SITES_REPO" rev-parse origin/main)"
  [ "$site_head" = "$remote_head" ] || fail "sites checkout is not at origin/main"

  run env AFFON_REPO="$ROOT" HAO_REPO="$SITE_HAO_REPO" npm --prefix "$SITES_REPO" run sync:affon
  run env AFFON_REPO="$ROOT" HAO_REPO="$SITE_HAO_REPO" npm --prefix "$SITES_REPO" run build:affon
  grep -q "Install the latest release.*v$version" "$SITES_REPO/sites/affon-ai/dist/index.html" || \
    fail "built homepage does not show v$version"
  grep -q "AFFON_RELEASE_VERSION=\"\\\${AFFON_RELEASE_VERSION:-$version}\"" "$SITES_REPO/sites/affon-ai/dist/install.sh" || \
    fail "built site installer does not default to $version"

  run git -C "$SITES_REPO" add sites/affon-ai/content.lock.json sites/affon-ai/public/install.sh
  if ! git -C "$SITES_REPO" diff --cached --quiet; then
    run git -C "$SITES_REPO" commit -m "Sync affon.ai to $version"
    run git -C "$SITES_REPO" push origin HEAD:main
  fi
  [ -z "$(git -C "$SITES_REPO" status --porcelain)" ] || fail "sites sync left uncommitted changes"
  run env AFFON_REPO="$ROOT" HAO_REPO="$SITE_HAO_REPO" npm --prefix "$SITES_REPO" run deploy:affon
  hosted_installer="$(mktemp "${TMPDIR:-/tmp}/affon-site-installer.XXXXXX")"
  hosted_home="$(mktemp "${TMPDIR:-/tmp}/affon-site-home.XXXXXX")"
  run curl -fsSL "https://affon.ai/install.sh" -o "$hosted_installer"
  run curl -fsSL "https://affon.ai/" -o "$hosted_home"
  grep -q "AFFON_RELEASE_VERSION=\"\\\${AFFON_RELEASE_VERSION:-$version}\"" "$hosted_installer" || \
    fail "hosted site installer does not default to $version"
  grep -q "Install the latest release.*v$version" "$hosted_home" || \
    fail "hosted homepage does not show v$version"
  rm -f "$hosted_installer" "$hosted_home"
  info "affon.ai synced to $version"
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
  metadata)
    [ -d "$DIST_DIR" ] || fail "missing dist/release"
    write_latest_json
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
  sync-site)
    sync_site
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
      sync_site
    else
      info "skipped remote smoke and GitHub release during dry run"
    fi
    ;;
  *)
    fail "unknown mode: $MODE"
    ;;
esac
