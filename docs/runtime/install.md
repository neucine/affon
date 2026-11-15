# Install Affon

On macOS arm64 or Linux x64, run:

```bash
curl -fsSL https://affon.ai/install.sh | bash
```

The last Windows x64 release is v0.4.1. To install it, run in PowerShell:

```powershell
irm https://downloads.affon.ai/install.ps1 | iex
```

The installer downloads the release for your platform, checks its SHA-256
checksum, and installs `affon` in `~/.affon/bin` (or `%USERPROFILE%\.affon\bin`
on Windows). It adds that directory to your user `PATH`. Open a new terminal
after installation.

For manual inspection before running, download the script first:

```bash
curl -fsSL https://affon.ai/install.sh -o install.sh
less install.sh
sh install.sh
```

Check the installation:

```sh
affon --version
```

To run your first script, follow the [First Program guide](../getting-started/first-program.md).

## Path Setup

If a new shell cannot find `affon`, ensure `~/.affon/bin` is on `PATH`:

```sh
export PATH="$HOME/.affon/bin:$PATH"
```

Add that line to your shell profile if the installer could not update it
automatically.
