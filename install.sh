#!/usr/bin/env sh

# Canonical source for the hosted installer at https://affon.ai/install.sh

set -eu

# Update this on each release so the bootstrap script installs the current
# public runtime by default.
AFFON_RELEASE_VERSION="${AFFON_RELEASE_VERSION:-0.2.0}"
AFFON_INSTALL_DIR="${AFFON_INSTALL:-$HOME/.affon}"
AFFON_BIN_DIR="${AFFON_BIN_DIR:-$AFFON_INSTALL_DIR/bin}"
AFFON_RELEASE_API_DEFAULT="https://downloads.affon.ai/releases/v${AFFON_RELEASE_VERSION}.json"
AFFON_API_URL="${AFFON_RELEASE_API:-${AFFON_GITHUB_API:-$AFFON_RELEASE_API_DEFAULT}}"

info() {
  printf 'affon-install: %s\n' "$*"
}

fail() {
  printf 'affon-install: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

cleanup() {
  if [ -n "${AFFON_TMP_DIR:-}" ] && [ -d "$AFFON_TMP_DIR" ]; then
    rm -rf "$AFFON_TMP_DIR"
  fi
}

trap cleanup EXIT INT TERM

detect_os() {
  case "$(uname -s)" in
    Darwin) printf 'darwin\n' ;;
    Linux) printf 'linux\n' ;;
    *) fail "unsupported operating system: $(uname -s)" ;;
  esac
}

detect_arch() {
  case "$(uname -m)" in
    x86_64|amd64) printf 'x64\n' ;;
    arm64|aarch64) printf 'arm64\n' ;;
    *) fail "unsupported architecture: $(uname -m)" ;;
  esac
}

download() {
  url="$1"
  out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$out"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -qO "$out" "$url"
    return
  fi
  fail "need curl or wget to download release assets"
}

pick_asset() {
  release_json="$1"
  target_os="$2"
  target_arch="$3"

  AFFON_RELEASE_JSON="$release_json" python3 - "$target_os" "$target_arch" <<'PY'
import json
import os
import sys

target_os = sys.argv[1]
target_arch = sys.argv[2]
payload = json.loads(os.environ["AFFON_RELEASE_JSON"])
assets = payload.get("assets", [])
tag = payload.get("tag_name") or payload.get("name") or "latest"

os_aliases = {
    "darwin": ["darwin", "macos", "mac", "apple-darwin"],
    "linux": ["linux", "unknown-linux", "gnu"],
}
arch_aliases = {
    "x64": ["x64", "amd64", "x86_64"],
    "arm64": ["arm64", "aarch64"],
}

archive_bonus = [".tar.gz", ".tgz", ".zip"]
binary_names = {"affon", "affon.exe"}

def score_asset(asset):
    name = (asset.get("name") or "").lower()
    url = (asset.get("browser_download_url") or "").lower()
    blob = f"{name} {url}"

    score = 0
    if "affon" in blob:
        score += 20

    os_hits = sum(alias in blob for alias in os_aliases[target_os])
    arch_hits = sum(alias in blob for alias in arch_aliases[target_arch])
    if os_hits == 0 or arch_hits == 0:
        return None

    score += 50 + os_hits * 5 + arch_hits * 5

    for suffix in archive_bonus:
        if name.endswith(suffix):
            score += 10
            break
    else:
        if os.path.basename(name) in binary_names:
            score += 5

    combos = [
        f"{target_os}-{target_arch}",
        f"{target_arch}-{target_os}",
        f"{target_os}_{target_arch}",
        f"{target_arch}_{target_os}",
    ]
    if any(combo in blob for combo in combos):
        score += 15

    return score

ranked = []
for asset in assets:
    score = score_asset(asset)
    if score is None:
        continue
    ranked.append((score, asset))

if not ranked:
    sys.stderr.write(
        "No compatible release asset found for "
        f"{target_os}/{target_arch}. Available assets: "
        + ", ".join(asset.get("name", "<unnamed>") for asset in assets)
        + "\n"
    )
    sys.exit(1)

ranked.sort(key=lambda item: item[0], reverse=True)
asset = ranked[0][1]
print(tag)
print(asset["name"])
print(asset["browser_download_url"])
PY
}

extract_binary() {
  asset_name="$1"
  asset_path="$2"
  dest_dir="$3"

  mkdir -p "$dest_dir"
  case "$asset_name" in
    *.tar.gz|*.tgz)
      need_cmd tar
      tar -xzf "$asset_path" -C "$dest_dir"
      ;;
    *.zip)
      need_cmd unzip
      unzip -q "$asset_path" -d "$dest_dir"
      ;;
    *)
      cp "$asset_path" "$dest_dir/affon"
      chmod 755 "$dest_dir/affon"
      ;;
  esac

  if [ -f "$dest_dir/affon" ]; then
    printf '%s\n' "$dest_dir/affon"
    return
  fi

  found="$(find "$dest_dir" -type f -name affon | head -n 1 || true)"
  if [ -n "$found" ]; then
    chmod 755 "$found"
    printf '%s\n' "$found"
    return
  fi

  fail "downloaded asset did not contain an affon binary"
}

shell_rc_path() {
  shell_name="$(basename "${SHELL:-}")"
  case "$shell_name" in
    zsh) printf '%s\n' "$HOME/.zshrc" ;;
    bash)
      if [ -f "$HOME/.bashrc" ] || [ ! -f "$HOME/.bash_profile" ]; then
        printf '%s\n' "$HOME/.bashrc"
      else
        printf '%s\n' "$HOME/.bash_profile"
      fi
      ;;
    fish) printf '%s\n' "$HOME/.config/fish/config.fish" ;;
    *) printf '\n' ;;
  esac
}

ensure_path() {
  case ":$PATH:" in
    *":$AFFON_BIN_DIR:"*) return ;;
  esac

  rc_path="$(shell_rc_path)"
  if [ -z "$rc_path" ]; then
    info "add $AFFON_BIN_DIR to your PATH to use affon globally"
    return
  fi

  mkdir -p "$(dirname "$rc_path")"
  touch "$rc_path"

  if grep -F "$AFFON_BIN_DIR" "$rc_path" >/dev/null 2>&1; then
    return
  fi

  case "$rc_path" in
    */config.fish)
      printf '\nfish_add_path "%s"\n' "$AFFON_BIN_DIR" >>"$rc_path"
      ;;
    *)
      printf '\nexport PATH="%s:$PATH"\n' "$AFFON_BIN_DIR" >>"$rc_path"
      ;;
  esac

  info "updated $rc_path to add $AFFON_BIN_DIR to PATH"
}

need_cmd uname
need_cmd mktemp
need_cmd mkdir
need_cmd chmod
need_cmd cp
need_cmd find
need_cmd head
need_cmd python3

TARGET_OS="$(detect_os)"
TARGET_ARCH="$(detect_arch)"
AFFON_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/affon-install.XXXXXX")"

info "fetching latest release metadata from $AFFON_API_URL"
RELEASE_JSON="$(curl -fsSL -H 'Accept: application/vnd.github+json' "$AFFON_API_URL" 2>/dev/null || true)"
if [ -z "$RELEASE_JSON" ]; then
  if command -v wget >/dev/null 2>&1; then
    RELEASE_JSON="$(wget -qO- "$AFFON_API_URL" 2>/dev/null || true)"
  fi
fi
[ -n "$RELEASE_JSON" ] || fail "unable to fetch release metadata"

ASSET_INFO="$(pick_asset "$RELEASE_JSON" "$TARGET_OS" "$TARGET_ARCH")" || exit 1
TAG_NAME="$(printf '%s\n' "$ASSET_INFO" | sed -n '1p')"
ASSET_NAME="$(printf '%s\n' "$ASSET_INFO" | sed -n '2p')"
ASSET_URL="$(printf '%s\n' "$ASSET_INFO" | sed -n '3p')"

info "downloading $ASSET_NAME from release $TAG_NAME"
ASSET_PATH="$AFFON_TMP_DIR/$ASSET_NAME"
download "$ASSET_URL" "$ASSET_PATH"

EXTRACT_DIR="$AFFON_TMP_DIR/extract"
BINARY_PATH="$(extract_binary "$ASSET_NAME" "$ASSET_PATH" "$EXTRACT_DIR")"

mkdir -p "$AFFON_BIN_DIR"
cp "$BINARY_PATH" "$AFFON_BIN_DIR/affon"
chmod 755 "$AFFON_BIN_DIR/affon"

ensure_path

info "installed affon to $AFFON_BIN_DIR/affon"
if "$AFFON_BIN_DIR/affon" --version >/dev/null 2>&1; then
  INSTALLED_VERSION="$("$AFFON_BIN_DIR/affon" --version 2>&1 || true)"
  [ -n "$INSTALLED_VERSION" ] && info "$INSTALLED_VERSION"
fi
info "restart your shell, or run:"
printf '  export PATH="%s:$PATH"\n' "$AFFON_BIN_DIR"
