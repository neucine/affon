$ErrorActionPreference = 'Stop'
$bash = 'C:\Program Files\Git\bin\bash.exe'
if (-not (Test-Path $bash)) {
    throw "Git Bash is required at $bash"
}
& $bash tools/buildkite/release-build.sh windows-x64
exit $LASTEXITCODE
