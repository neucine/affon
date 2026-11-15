# Affon release maintenance

[CI and release steps](../docs/ci.md) describe the current Buildkite flow. The
same pipeline checks release candidates on macOS and Linux, then waits for
approval before publishing the merged commit. It does not build Windows assets.

`tools/release.sh` is the local helper used by that pipeline. Run
`./tools/release.sh check` to check a prepared checkout, or see the modes at
the top of the script when diagnosing a release job. Keep dependencies and
release tools on the runner, outside the Affon checkout.
