# Security Policy

Affon includes runtime features such as FFI, filesystem access, process execution, HTTP, and native compute backends. Please report suspected vulnerabilities privately.

## Reporting A Vulnerability

- Prefer GitHub's private vulnerability reporting for this repository if it is enabled.
- If private reporting is not available, do not open a public issue with exploit details. Open a minimal issue or discussion requesting a private contact path instead.

Please include:

- affected version or commit
- host OS and architecture
- whether the issue requires local code execution, malicious input, or a crafted package
- clear reproduction steps or a proof of concept
- impact assessment and any known mitigations

We will try to acknowledge valid reports promptly and follow up with status updates as the fix is triaged.

## Supported Versions

During the first public release cycle, the latest release on `main` receives security fixes first. Older versions may be asked to upgrade before a fix is backported.

| Version | Supported |
| --- | --- |
| `0.1.x` | Yes |
| `< 0.1.0` | No |
