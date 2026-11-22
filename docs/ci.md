# Affon CI and releases

The [Affon Buildkite pipeline](https://buildkite.com/neucine/affon-ci) runs a short
Mac check for ordinary branch and pull request commits. Release builds use the
Mac and the GPU Linux runner on `pc`.

## Prepare a release

Create a branch named `release/vX.Y.Z`. Set the version in
`src/main.cy` and
update the Unix installer and release notes for that version. Each push runs
the full checks on both release hosts. Fixes pushed to the branch start a new
candidate build.

Merge the release pull request into `main` after its candidate passes. The
resulting `main` build checks the merged commit again on both hosts. When both
archives pass checksum checks, approve **Approve Affon release** in that build.
The same build then tags the commit, uploads those verified archives to R2 and
GitHub, checks the remote Mac install, and updates affon.ai. Approval does not
rebuild the binaries.

A failed job can be retried on the same commit. A source fix needs a new commit
and a new build. The Windows installer remains pinned to the last Windows
release, v0.4.1.
