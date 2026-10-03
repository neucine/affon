"""Copy explicit source directories into a Docker build context; excludes models and caches."""
import argparse
from pathlib import Path
import shutil

repo = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output")
args = parser.parse_args()
output = Path(args.output).resolve()
if output.exists():
    parser.error("Use a new output directory")
output.mkdir(parents=True)
ignore = shutil.ignore_patterns("target", "__pycache__", ".DS_Store")
for name, paths in {
    "affon": ["build.zig", "build.zig.zon", "src", "tools"],
    "hao": ["build.zig", "build.zig.zon", "src", "include", "libs/transpiler"],
    "compute": ["build.zig", "build.zig.zon", "src", "tools"],
    "zig-libs": ["build.zig", "build.zig.zon", "src"],
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
shutil.copyfile(Path(__file__).with_name("Dockerfile"), output / "Dockerfile")
print(output)
