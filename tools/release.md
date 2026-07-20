# Affon Release Flow

This document defines the maintainer release workflow for Affon runtime
releases.

The primary release path is tag-driven GitHub Actions:

```text
.github/workflows/release.yml
```

The local maintainer helper is:

```text
tools/release.sh
```

The goal is to keep every release in the same order:

1. build the runtime from the checkout
2. run the release test suite
3. verify the built runtime version
4. stage native release assets
5. smoke test the staged archive
6. publish the tag
7. let CI build, publish GitHub assets, and mirror hosted assets to R2
8. smoke test the hosted archive when doing a manual local upload
9. publish the canonical installer script

## Standard CI Release

Prepare and verify the release commit locally, then push the release commit and
tag:

```sh
git push origin HEAD
git push origin v0.2.0
```

The release workflow:

- validates that the tag matches `build.zig.zon`
- checks out Affon plus pinned `hao`, `compute`, and `zig-libs` dependencies
- builds and tests on Linux and macOS
- type-checks public TypeScript contracts and first-party packages
- packages direct native binaries, `.tar.gz` archives, and `.sha256` files
- creates a GitHub Release
- writes hosted `latest.json` and versioned release metadata
- uploads hosted assets to R2 under `releases/<tag>/`
- uploads metadata to `releases/latest.json` and `releases/<tag>.json`
- uploads `install.sh` to `install.sh` and `releases/install.sh`

CI requires these repository secrets:

- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_API_TOKEN`

## Local Release Helper

Run checks:

```sh
./tools/release.sh check
```

Build staged artifacts:

```sh
./tools/release.sh build
```

Smoke test the staged archive:

```sh
./tools/release.sh smoke-local
```

Dry-run the local R2 upload:

```sh
AFFON_R2_DRY_RUN=1 ./tools/release.sh upload-r2
```

Upload through R2's S3-compatible endpoint:

```sh
AFFON_R2_ACCOUNT_ID=... \
AFFON_R2_BUCKET=affon-downloads \
AWS_ACCESS_KEY_ID=... \
AWS_SECRET_ACCESS_KEY=... \
./tools/release.sh upload-r2
```

Smoke test the hosted archive:

```sh
./tools/release.sh smoke-remote
```

Upload only the canonical installer script:

```sh
./tools/release.sh upload-install
```

Create a GitHub Release locally after the tag exists remotely:

```sh
./tools/release.sh github-release
```

Run the full local flow:

```sh
./tools/release.sh all
```

When `AFFON_R2_DRY_RUN=1` is set, `all` stops before remote smoke testing and
GitHub Release creation.

## Local Upload Environment

Required for the default local `aws` upload path:

- `AFFON_R2_ACCOUNT_ID` or `CLOUDFLARE_ACCOUNT_ID`
- `AFFON_R2_BUCKET` or `R2_BUCKET`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

Optional:

- `AWS_PROFILE`
- `AFFON_R2_PREFIX`, default `releases`
- `AFFON_DOWNLOAD_BASE_URL`, default `https://downloads.affon.ai`
- `AFFON_R2_UPLOAD_TOOL=wrangler` to use Wrangler instead of the AWS CLI

The local helper stages assets in:

```text
dist/release/
```

For version `0.2.0` on Darwin arm64, the staged files are:

```text
affon-v0.2.0-darwin-arm64
affon-v0.2.0-darwin-arm64.sha256
affon-v0.2.0-darwin-arm64.tar.gz
affon-v0.2.0-darwin-arm64.tar.gz.sha256
latest.json
v0.2.0.json
```

The canonical installer is [install.sh](../../install.sh). It carries
`AFFON_RELEASE_VERSION`, which defaults to the current release version and points
at `https://downloads.affon.ai/releases/v${AFFON_RELEASE_VERSION}.json`. The
local helper syncs that default from `build.zig.zon`; CI verifies that the two
versions match before publishing.
