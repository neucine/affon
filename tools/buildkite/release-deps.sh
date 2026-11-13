#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
parent="$(dirname "$root")"


if [ -n "${PRIVATE_REPO_READ_TOKEN:-}" ]; then
  askpass="$(mktemp)"
  trap 'rm -f "$askpass"' EXIT
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
  local destination="$parent/$name"
  if [ ! -d "$destination/.git" ]; then
    git clone --quiet "https://github.com/neucine/$repo.git" "$destination"
  fi
  git -C "$destination" fetch --quiet origin "$ref"
  git -C "$destination" checkout --quiet --detach "$ref"
  test "$(git -C "$destination" rev-parse HEAD)" = "$ref"
}

clone_pinned hao 7c73a564056af4cbb47998278b5253d726e03d9f hao-legacy
clone_pinned compute dc66696c6444d4077bea9e172e1f6e1d990ed223 compute
clone_pinned zig-libs c5b61fbdd2c7c02d875212e7111a27be3aaf2f17 zig-libs
