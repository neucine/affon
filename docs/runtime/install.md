# Installing AFFON

The recommended install path is the hosted installer:

```bash
curl -fsSL https://affon.ai/install.sh | bash
```

The installer:

- detects the current OS and CPU architecture
- downloads the latest matching release asset from the AFFON downloads endpoint
- installs `affon` into `~/.affon/bin`
- adds `~/.affon/bin` to the user's shell `PATH`
- prints the installed `affon --version`

For manual inspection before running, download the script first:

```bash
curl -fsSL https://affon.ai/install.sh -o install.sh
less install.sh
sh install.sh
```

Run a script after installation:

```sh
affon script.ts
```

Verify the installed runtime:

```sh
affon --version
```

## Path Setup

If a new shell cannot find `affon`, ensure `~/.affon/bin` is on `PATH`:

```sh
export PATH="$HOME/.affon/bin:$PATH"
```

Add that line to your shell profile if the installer could not update it
automatically.

## Install Notes

- The hosted installer is the public entry point for release asset selection
  and shell integration behavior.
- If you need to audit the installer, download it first, inspect it locally, and
  then run the downloaded copy.
