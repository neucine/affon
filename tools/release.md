# Affon Release Flow

This document defines the maintainer release workflow for Affon runtime
releases.

The GitHub Actions release workflow is available for a future CI release:

```text
.github/workflows/release.yml
```

The workflow calls the local maintainer helper for version checks, tests, native
packaging, smoke tests, and hosted metadata:

```text
tools/release.sh
```

The local release uses this order:

1. build the runtime from the checkout
2. run the release test suite
3. verify the built runtime version
4. stage native release assets
5. smoke test the staged archive
6. publish the tag
7. gather both native assets in one `dist/release/` directory
8. publish the release tag and GitHub assets
9. mirror assets and metadata to R2, then publish the canonical installer
10. smoke test both hosted archives

## Standard Local Release

Run `check`, `build`, and `smoke-local` in clean, pinned dependency checkouts
on macOS arm64 and Linux x64. Copy the four Linux assets into the macOS
`dist/release/` directory. From that checkout, publish the release commit and
tag, then run:

```sh
./tools/release.sh metadata
./tools/release.sh github-release
AFFON_R2_UPLOAD_TOOL=wrangler ./tools/release.sh upload-r2
AFFON_R2_UPLOAD_TOOL=wrangler ./tools/release.sh upload-install
./tools/release.sh smoke-remote
AFFON_SITES_REPO=/path/to/sites AFFON_SITE_HAO_REPO=/path/to/pinned/hao \
  ./tools/release.sh sync-site
```

Run `smoke-remote` on Linux as well, using the staged Linux asset there. Set
`CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` for Wrangler uploads.
`sync-site` checks the pinned Hao commit, builds affon.ai from the published
Affon commit, pushes its content lock and installer to the sites repository,
deploys the site, and verifies the hosted homepage and installer version.
The `all` mode performs this step after publishing the GitHub Release.

## Optional CI Release

Prepare and verify the release commit locally, then push the release commit and
tag. Dispatch the release workflow manually on that tag when the runners are
available:

```sh
git push origin HEAD
git push origin v0.4.0
gh workflow run release.yml --repo neucine/affon --ref v0.4.0
```

The release workflow:

- validates that the tag and installer version match `build.zig.zon`
- checks out Affon plus pinned `hao`, `compute`, and `zig-libs` dependencies
- builds and tests on Linux and macOS
- type-checks public TypeScript contracts and first-party packages
- packages direct native binaries, `.tar.gz` archives, and `.sha256` files
- creates a GitHub Release
- writes hosted `latest.json` and versioned release metadata
- uploads hosted assets to R2 under `releases/<tag>/`
- uploads metadata to `releases/latest.json` and `releases/<tag>.json`
- uploads `install.sh` to `install.sh` and `releases/install.sh`

After a CI release, run `sync-site` from a clean local checkout with the pinned
Hao source. The manual workflow currently publishes release assets only.

CI, security scans, and release jobs use the Neucine organization runners
`eq12-linux-x64` and `macbook-arm64`. The native release assets therefore target
Linux x64 and macOS arm64. Both runners must be online and available to this
repository before pushing a release tag.

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

Write hosted metadata after gathering native assets:

```sh
./tools/release.sh metadata
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

Run the full local flow on one host when only that host's asset is required:

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

For version `0.4.0` on Darwin arm64, `build` stages these files:

```text
affon-v0.4.0-darwin-arm64
affon-v0.4.0-darwin-arm64.sha256
affon-v0.4.0-darwin-arm64.tar.gz
affon-v0.4.0-darwin-arm64.tar.gz.sha256
```

The canonical installer is [install.sh](../../install.sh). It carries
`AFFON_RELEASE_VERSION`, which defaults to the current release version and points
at `https://downloads.affon.ai/releases/v${AFFON_RELEASE_VERSION}.json`. The
Update that default alongside `build.zig.zon`; the helper and CI require the two
versions to match before publishing. The `metadata` helper mode writes hosted
metadata after native assets have been gathered in `dist/release/`.
