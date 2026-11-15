#!/usr/bin/env bash
# Source this from an Affon Buildkite job. It creates an isolated checkout and
# pinned sibling dependencies without changing another pipeline's files.
set -euo pipefail

source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
task_id="${BUILDKITE_JOB_ID:-local-$$}"
[[ "$task_id" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid Buildkite job ID" >&2; exit 1; }
workspace="$HOME/.cache/neucine/affon-jobs/$task_id"
root="$workspace/affon"
cache_root="$HOME/.cache/neucine/affon-deps"
mkdir -p "$workspace" "$cache_root"

cleanup_affon_workspace() {
  cd "$source_root"
  git -C "$source_root" worktree remove --force "$root" 2>/dev/null || true
  if [ -n "${askpass:-}" ]; then rm -f "$askpass"; fi
  if [ -n "${download:-}" ]; then rm -rf -- "$download"; fi
  rm -rf -- "$workspace"
}
trap cleanup_affon_workspace EXIT

git -C "$source_root" worktree add --quiet --detach "$root" "${BUILDKITE_COMMIT:?Buildkite commit required}"
cd "$root"

if [ -n "${PRIVATE_REPO_READ_TOKEN:-}" ]; then
  askpass="$(mktemp)"
  cat > "$askpass" <<'EOF'
#!/usr/bin/env sh
case "$1" in
  *Username*) printf '%s\n' x-access-token ;;
  *) printf '%s\n' "$PRIVATE_REPO_READ_TOKEN" ;;
esac
EOF
  chmod 700 "$askpass"
  export GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0
fi

clone_pinned() {
  local repo="$1" ref="$2" name="$3"
  local mirror="$cache_root/$repo.git" destination="$workspace/$name"
  if [ ! -d "$mirror/objects" ]; then
    git clone --quiet --mirror "https://github.com/neucine/$repo.git" "$mirror"
  fi
  git -C "$mirror" fetch --quiet origin "$ref"
  git clone --quiet --shared --no-checkout "$mirror" "$destination"
  git -C "$destination" checkout --quiet --detach "$ref"
  test "$(git -C "$destination" rev-parse HEAD)" = "$ref"
}

clone_pinned hao 7c73a564056af4cbb47998278b5253d726e03d9f hao-legacy
clone_pinned compute dc66696c6444d4077bea9e172e1f6e1d990ed223 compute
clone_pinned zig-libs c5b61fbdd2c7c02d875212e7111a27be3aaf2f17 zig-libs

mkdir -p "$HOME/.cache/neucine/affon-build/zig" "$HOME/.cache/neucine/affon-build/cargo"
ln -s "$HOME/.cache/neucine/affon-build/zig" "$root/.zig-cache"
ln -s "$HOME/.cache/neucine/affon-build/cargo" "$workspace/hao-legacy/libs/transpiler/target"
