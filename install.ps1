$ErrorActionPreference = "Stop"

$ReleaseVersion = if ($env:AFFON_RELEASE_VERSION) { $env:AFFON_RELEASE_VERSION } else { "0.4.1" }
$InstallDir = if ($env:AFFON_INSTALL) { $env:AFFON_INSTALL } else { Join-Path $HOME ".affon" }
$BinDir = if ($env:AFFON_BIN_DIR) { $env:AFFON_BIN_DIR } else { Join-Path $InstallDir "bin" }
$ApiUrl = if ($env:AFFON_RELEASE_API) {
    $env:AFFON_RELEASE_API
} elseif ($env:AFFON_GITHUB_API) {
    $env:AFFON_GITHUB_API
} else {
    "https://downloads.affon.ai/releases/v$ReleaseVersion.json"
}

if ($env:PROCESSOR_ARCHITECTURE -ne "AMD64") {
    throw "Affon currently supports Windows x64 only"
}

$Tag = "v$ReleaseVersion"
$AssetName = "affon-$Tag-windows-x64.exe"
$ChecksumName = "$AssetName.sha256"
$Release = Invoke-RestMethod -Uri $ApiUrl
$Asset = $Release.assets | Where-Object { $_.name -eq $AssetName } | Select-Object -First 1
$Checksum = $Release.assets | Where-Object { $_.name -eq $ChecksumName } | Select-Object -First 1
if (-not $Asset) {
    throw "Release metadata does not contain $AssetName"
}
if (-not $Checksum) {
    throw "Release metadata does not contain $ChecksumName"
}

$TemporaryDir = Join-Path ([System.IO.Path]::GetTempPath()) ("affon-install-" + [System.Guid]::NewGuid())
New-Item -ItemType Directory -Path $TemporaryDir | Out-Null
try {
    $AssetPath = Join-Path $TemporaryDir $AssetName
    $ChecksumPath = Join-Path $TemporaryDir $ChecksumName
    Invoke-WebRequest -Uri $Asset.browser_download_url -OutFile $AssetPath
    Invoke-WebRequest -Uri $Checksum.browser_download_url -OutFile $ChecksumPath

    $Expected = ((Get-Content $ChecksumPath -TotalCount 1) -split "\s+")[0].ToLowerInvariant()
    if ($Expected -notmatch "^[0-9a-f]{64}$") {
        throw "Invalid SHA-256 checksum file"
    }
    $Actual = (Get-FileHash -Algorithm SHA256 -Path $AssetPath).Hash.ToLowerInvariant()
    if ($Actual -ne $Expected) {
        throw "Affon release asset SHA-256 verification failed"
    }

    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    Copy-Item -Force $AssetPath (Join-Path $BinDir "affon.exe")

    $CurrentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $PathEntries = @($CurrentPath -split ";" | Where-Object { $_ })
    if ($PathEntries -notcontains $BinDir) {
        [Environment]::SetEnvironmentVariable("Path", (($PathEntries + $BinDir) -join ";"), "User")
    }

    Write-Host "affon-install: installed Affon $Tag to $(Join-Path $BinDir 'affon.exe')"
    Write-Host "affon-install: restart the terminal to refresh PATH"
}
finally {
    Remove-Item -Recurse -Force $TemporaryDir -ErrorAction SilentlyContinue
}
