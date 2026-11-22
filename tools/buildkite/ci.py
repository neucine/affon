#!/usr/bin/env python3
"""Route Affon CI and bind a release to one verified Buildkite commit."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRANCH = re.compile(r"release/v(\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?)$")
PLATFORMS = ("darwin-arm64", "linux-x64", "windows-x64")
PLATFORM_STEPS = dict(zip(PLATFORMS, ("macos", "linux", "windows")))


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def version() -> str:
    source = (ROOT / "src/affon.cy").read_text()
    match = re.search(r'^pub const version = "([^"]+)";$', source, re.MULTILINE)
    if not match:
        raise SystemExit("Cyna host has no version")
    return match.group(1)


def commit() -> str:
    sha = os.environ["BUILDKITE_COMMIT"]
    if run("git", "rev-parse", "HEAD") != sha:
        raise SystemExit("checkout differs from Buildkite commit")
    return sha


def metadata(name: str, value: str | None = None) -> str:
    if value is None:
        return run("buildkite-agent", "meta-data", "get", name)
    run("buildkite-agent", "meta-data", "set", name, value)
    return value


def merged_release(sha: str) -> tuple[str, int] | None:
    pulls = json.loads(run("gh", "api", f"repos/neucine/affon/commits/{sha}/pulls"))
    matches = [pr for pr in pulls if pr.get("merged_at") and pr.get("merge_commit_sha") == sha
               and pr["base"]["ref"] == "main" and BRANCH.fullmatch(pr["head"]["ref"])]
    if len(matches) > 1:
        raise SystemExit("more than one merged release PR points to this commit")
    if not matches:
        return None
    pr = matches[0]
    return BRANCH.fullmatch(pr["head"]["ref"]).group(1), pr["number"]


def candidate() -> None:
    branch = os.environ["BUILDKITE_BRANCH"]
    match = BRANCH.fullmatch(branch)
    if not match or version() != match.group(1):
        raise SystemExit("release branch and Cyna host versions differ")
    metadata("release_tag", f"v{match.group(1)}")
    metadata("release_commit", commit())
    print(f"Candidate v{match.group(1)} at {commit()}")


def detect() -> None:
    if os.environ["BUILDKITE_BRANCH"] != "main":
        raise SystemExit("merged release detection requires main")
    sha = commit()
    release = merged_release(sha)
    if release is None:
        print("Ordinary main build; no merged release PR")
        return
    selected, number = release
    if version() != selected:
        raise SystemExit("merged release version differs from the Cyna host")
    metadata("release_tag", f"v{selected}")
    metadata("release_commit", sha)
    metadata("release_pr", str(number))
    subprocess.run(["buildkite-agent", "pipeline", "upload", ".buildkite/merged-release.yml"], cwd=ROOT, check=True)
    print(f"Added merged release v{selected} from PR #{number}")


def checked_release() -> tuple[str, str]:
    sha = commit()
    tag = metadata("release_tag")
    if metadata("release_commit") != sha or tag != f"v{version()}":
        raise SystemExit("release metadata differs from this checkout")
    return tag, sha


def collect(lane: str) -> None:
    if lane not in ("candidate", "merged"):
        raise SystemExit("collect lane must be candidate or merged")
    tag, sha = checked_release()
    dist = ROOT / "dist/release"
    shutil.rmtree(dist, ignore_errors=True)
    dist.mkdir(parents=True, exist_ok=True)
    assets = []
    for platform in PLATFORMS:
        step = f"{lane}-{PLATFORM_STEPS[platform]}"
        subprocess.run(["buildkite-agent", "artifact", "download", "dist/release/*", ".", "--step", step], cwd=ROOT, check=True)
        stem = f"affon-{tag}-{platform}"
        for name in (stem + (".exe" if platform == "windows-x64" else ""), f"{stem}.tar.gz"):
            path = dist / name
            checksum = dist / f"{name}.sha256"
            if not path.is_file() or not checksum.is_file():
                raise SystemExit(f"missing {platform} release asset or checksum: {name}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if checksum.read_text().split()[0].lower() != digest:
                raise SystemExit(f"asset checksum differs: {name}")
            assets.append({"name": name, "sha256": digest, "size": path.stat().st_size})
    manifest = dist / "verified.json"
    manifest.write_text(json.dumps({"tag": tag, "commit": sha, "platforms": list(PLATFORMS), "assets": assets}, indent=2) + "\n")
    subprocess.run(["buildkite-agent", "artifact", "upload", "dist/release/*"], cwd=ROOT, check=True)
    print(f"Verified {len(assets)} assets from {sha}")


def publish_check() -> None:
    if os.environ["BUILDKITE_BRANCH"] != "main":
        raise SystemExit("publication requires main")
    tag, sha = checked_release()
    release = merged_release(sha)
    if release != (version(), int(metadata("release_pr"))):
        raise SystemExit("release PR no longer matches this commit")
    subprocess.run(["git", "fetch", "origin", "main"], cwd=ROOT, check=True)
    if run("git", "rev-parse", "origin/main") != sha:
        raise SystemExit("main moved after this release build started")
    manifest = json.loads((ROOT / "dist/release/verified.json").read_text())
    if manifest["tag"] != tag or manifest["commit"] != sha or manifest["platforms"] != list(PLATFORMS):
        raise SystemExit("verified assets belong to another release")
    expected_names = {name for platform in PLATFORMS for name in
                      (f"affon-{tag}-{platform}" + (".exe" if platform == "windows-x64" else ""), f"affon-{tag}-{platform}.tar.gz")}
    if {asset["name"] for asset in manifest["assets"]} != expected_names or len(manifest["assets"]) != len(expected_names):
        raise SystemExit("verified release must contain exactly three platform binaries and archives")
    for asset in manifest["assets"]:
        path = ROOT / "dist/release" / asset["name"]
        if not path.is_file() or path.stat().st_size != asset["size"] or hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
            raise SystemExit(f"verified asset changed: {asset['name']}")
    existing = run("git", "ls-remote", "origin", f"refs/tags/{tag}^{{}}")
    if existing and existing.split()[0] != sha:
        raise SystemExit(f"{tag} points to another commit")
    if not existing:
        # GitHub's API uses the scoped release token and avoids runner Git
        # credential helper state. The tag is created only after approval.
        created = json.loads(run("gh", "api", "-X", "POST", "repos/neucine/affon/git/tags",
                                 "-f", f"tag={tag}", "-f", f"message=Affon {version()}",
                                 "-f", f"object={sha}", "-f", "type=commit"))
        run("gh", "api", "-X", "POST", "repos/neucine/affon/git/refs",
            "-f", f"ref=refs/tags/{tag}", "-f", f"sha={created['sha']}")
    subprocess.run(["git", "fetch", "--quiet", "origin", f"refs/tags/{tag}:refs/tags/{tag}"], cwd=ROOT, check=True)
    print(f"Approved {tag} at {sha} with {len(manifest['assets'])} verified assets")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action == "candidate":
        candidate()
    elif action == "detect":
        detect()
    elif action == "collect" and len(sys.argv) == 3:
        collect(sys.argv[2])
    elif action == "publish-check":
        publish_check()
    else:
        raise SystemExit("usage: ci.py candidate|detect|collect candidate|collect merged|publish-check")
