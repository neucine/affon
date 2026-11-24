#!/usr/bin/env bash
# Source this from an Affon Buildkite job. It creates an isolated checkout and
# pinned sibling dependencies without changing another pipeline's files.
set -euo pipefail

source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
hao_pin=42d57709117a7414547ef60808d9c6e31bf084e8
compute_pin=ca5c2e06631816bb691529c9d18b9883be42d045

check_current_pin() {
  local repo="$1" branch="$2" pin="$3" branch_tip
  branch_tip="$(git ls-remote "https://github.com/neucine/$repo.git" "refs/heads/$branch" | awk '{print $1}')"
  [ -n "$branch_tip" ] || { echo "Could not read $repo $branch branch" >&2; return 1; }
  if [ "$pin" != "$branch_tip" ]; then
    echo "$repo pin is $pin, but $branch is $branch_tip" >&2
    return 1
  fi
  echo "$repo pin matches $branch: $pin"
}

if [ "${1:-}" = --check-pins ]; then
  result=0
  check_current_pin hao main "$hao_pin" || result=1
  check_current_pin compute-core main "$compute_pin" || result=1
  exit "$result"
fi
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

export AFFON_SITE_HAO_REF="$hao_pin"
clone_pinned hao "$hao_pin" hao
clone_pinned compute-core "$compute_pin" compute-core

# Hao's Cyna manifests link the transpiler from its checkout-local target path.
# Keep Cargo's output there so the pinned dependency layout works in CI.
export CARGO_TARGET_DIR="$workspace/hao/libs/transpiler/target"
mkdir -p "$CARGO_TARGET_DIR"
(
  cd "$workspace/hao"
  bash ./build_deps.sh
  sh ./tools/prepare_std_modules.sh
)
