# Installing AFFON

The recommended install path is the hosted installer:

```bash
curl -fsSL https://affon.ai/install.sh | bash
```

The website copy at `https://affon.ai/install.sh` should stay byte-for-byte
aligned with the canonical installer script in this repo:

- [install.sh](../../install.sh)

The installer:

- detects the current OS and CPU architecture
- downloads the matching release asset for its bundled `AFFON_RELEASE_VERSION`
  from the AFFON downloads endpoint
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
- The installer carries a default `AFFON_RELEASE_VERSION`; override it only when
  testing a different hosted release metadata file.
- If you need to audit the installer, download it first, inspect it locally, and
  then run the downloaded copy.
