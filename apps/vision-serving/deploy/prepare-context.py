"""Copy explicit source directories into a Docker build context; excludes models and caches."""
import argparse
import os
from pathlib import Path
import shutil

repo = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output")
parser.add_argument(
    "--cyna-distribution",
    type=Path,
    default=(
        Path(os.environ["CYNA_DISTRIBUTION_DIR"])
        if os.environ.get("CYNA_DISTRIBUTION_DIR")
        else None
    ),
    help="Cyna distribution directory for the container's target architecture",
)
args = parser.parse_args()
output = Path(args.output).resolve()
if output.exists():
    parser.error("Use a new output directory")
if args.cyna_distribution is None:
    parser.error(
        "Set --cyna-distribution or CYNA_DISTRIBUTION_DIR to a Linux Cyna "
        "distribution"
    )
if not (args.cyna_distribution / "bin/cyna").is_file():
    parser.error(f"Cyna distribution is missing bin/cyna: {args.cyna_distribution}")
output.mkdir(parents=True)
ignore = shutil.ignore_patterns("target", "__pycache__", ".DS_Store")
for name, paths in {
    "affon": ["cyna.toml", "cyna.lock", "src"],
    "hao": [
        "cyna.toml",
        "build_deps.sh",
        "src",
        "include",
        "libs/transpiler",
        "patches",
        "tools/prepare_std_modules.sh",
    ],
    "compute-core": ["cyna.toml", "src"],
}.items():
    for relative in paths:
        source, dest = repo.parent / name / relative, output / "source" / name / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, dest, ignore=ignore)
        else:
            shutil.copyfile(source, dest)
for source in (repo / "packages/@affon").iterdir():
    if not source.is_dir():
        continue
    for relative in ["src", "package.json"]:
        path = source / relative
        if not path.exists():
            continue
        dest = output / "source/affon/packages/@affon" / source.name / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, dest, ignore=ignore)
        else:
            shutil.copyfile(path, dest)
shutil.copytree(repo / "apps/vision-serving/src", output / "source/affon/apps/vision-serving/src")
shutil.copytree(args.cyna_distribution, output / "cyna", ignore=ignore)
shutil.copyfile(Path(__file__).with_name("Dockerfile"), output / "Dockerfile")
print(output)
