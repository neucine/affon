# Installing AFFON

The recommended install path is:

```bash
curl -fsSL https://affon.ai/install.sh | bash
```

The website copy at `https://affon.ai/install.sh` should stay byte-for-byte aligned with the canonical installer script in this repo:

- [install.sh](/Users/chao.yang/Private/affon/install.sh)

The installer:

- detects the current OS and CPU architecture
- downloads the latest matching release asset from the AFFON downloads endpoint
- installs `affon` into `~/.affon/bin`
- adds `~/.affon/bin` to the user's shell `PATH`
- prints the installed `affon --version`

For manual inspection before running, download the script first:

```bash
curl -fsSL https://affon.ai/install.sh -o install.sh
sh install.sh
```
