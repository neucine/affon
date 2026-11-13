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
git push origin v0.4.1
gh workflow run release.yml --repo neucine/affon --ref v0.4.1
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
- uploads `install.sh` and `install.ps1` to the download root and `releases/`

After a CI release, run `sync-site` from a clean local checkout with the pinned
Hao source. The manual workflow currently publishes release assets only.

CI and security scans use GitHub-hosted runners. Release builds use the Neucine
organization runners `eq12-linux-x64` and `macbook-arm64`, plus a
GitHub-hosted Windows x64 runner. The native release assets target Linux x64,
macOS arm64, and Windows x64.

### Windows release status

Windows x64 is a release target. Affon cross-links with the Rust transpiler
archive built for `x86_64-pc-windows-gnullvm`, and the final PE includes the
llvm-mingw unwind runtime.

The release workflow installs LLVM-MinGW `20260908`, builds the Windows asset on
`windows-latest`, verifies `affon --version`, runs the first-program CPU demo,
and checks the packaged archive before upload. Release metadata now requires the
Windows direct `.exe`, archive, and both checksum files alongside the macOS and
Linux assets.

For a local macOS cross-build:

```sh
export LLVM_MINGW_PREFIX=/path/to/llvm-mingw
export PATH="$LLVM_MINGW_PREFIX/bin:$PATH"
export CARGO_TARGET_X86_64_PC_WINDOWS_GNULLVM_LINKER="$LLVM_MINGW_PREFIX/bin/x86_64-w64-mingw32-clang"
export AFFON_RELEASE_PLATFORM=windows-x64
export AFFON_WINDOWS_GNU_LIBUNWIND="$LLVM_MINGW_PREFIX/x86_64-w64-mingw32/lib/libunwind.a"
./tools/release.sh build
```

This validates the PE build and package layout. Run `smoke-local` on Windows to
execute the runtime test.

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

The `all` mode is retained for automation, but the two-host promotion sequence
above is the supported release procedure. `metadata` and upload now require
both native asset sets.

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

For version `0.4.1` on Darwin arm64, `build` stages these files:

```text
affon-v0.4.1-darwin-arm64
affon-v0.4.1-darwin-arm64.sha256
affon-v0.4.1-darwin-arm64.tar.gz
affon-v0.4.1-darwin-arm64.tar.gz.sha256
affon-v0.4.1-windows-x64.exe
affon-v0.4.1-windows-x64.exe.sha256
affon-v0.4.1-windows-x64.tar.gz
affon-v0.4.1-windows-x64.tar.gz.sha256
```

Unix installers use [install.sh](../../install.sh), and Windows uses
[install.ps1](../../install.ps1). Both carry
`AFFON_RELEASE_VERSION`, which defaults to the current release version and points
at `https://downloads.affon.ai/releases/v${AFFON_RELEASE_VERSION}.json`. The
Update that default alongside `build.zig.zon`; the helper and CI require the two
versions to match before publishing. The `metadata` helper mode writes hosted
metadata after native assets have been gathered in `dist/release/`.

## Promotion verification

Before `metadata`, gather **all three** platform asset sets in
`dist/release/`: `darwin-arm64`, `linux-x64`, and `windows-x64`, each with a direct binary, a
`.tar.gz` archive, and a `.sha256` file for each. `metadata` now rejects missing
files, malformed checksums, and checksum mismatches. Run `smoke-local` on each
native host; a macOS host cannot execute the Linux binary or establish CUDA
readiness. The Unix installer supports macOS arm64 and Linux x64; Windows x64
uses `install.ps1`.

The build depends on sibling `hao-legacy`, `compute`, and `zig-libs` checkouts.
The workflows pin Hao `7c73a564056af4cbb47998278b5253d726e03d9f`, Compute
`dc66696c6444d4077bea9e172e1f6e1d990ed223`, and zig-libs
`c5b61fbdd2c7c02d875212e7111a27be3aaf2f17`. Update those hashes together with
the dependency commits. `build.zig.zon` uses local paths, so an Affon source
checkout by itself is not a standalone public source installation. The prebuilt
installer is the supported clean-machine path.

The `all` command builds only the current host asset and replaces `dist/release/`
before metadata generation. It therefore cannot complete a multi-platform release
as written. Use the multi-host `check`, `build`, `smoke-local`, gather,
`metadata` sequence for promotion. Do not infer cross-platform readiness from
one host's smoke test.
