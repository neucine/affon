"""Local model packaging experiment. Python standard library only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


SCHEMA = "affon-model-package/v1"
ROOT = Path(__file__).resolve().parents[2]
# Contracts describe packages. Engines are selected by the caller.
ADAPTERS = {
    "affon/prepared-onnx/v1": ("graph.json", "weights.safetensors"),
    "affon/llama-tokens/v1": ("config.json", "model.safetensors"),
}
ONNX_CONTRACT = "onnx/tensors-f32/v1"


def file_digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def validate(value):
    if not isinstance(value, dict) or set(value) != {"schema", "contract", "entrypoint", "artifacts"}:
        raise ValueError("Expected schema, contract, entrypoint and artifacts")
    if value["schema"] != SCHEMA:
        raise ValueError("Unsupported package schema")
    if not isinstance(value["contract"], str) or not value["contract"]:
        raise ValueError("Expected a versioned model contract identifier")
    if not isinstance(value["artifacts"], list) or not value["artifacts"]:
        raise ValueError("Expected nonempty artifacts")
    names = set()
    for item in value["artifacts"]:
        if not isinstance(item, dict) or set(item) != {"path", "size", "digest"}:
            raise ValueError("Expected artifact path, size and digest")
        name = item["path"]
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
                or name == "manifest.json" or name in names):
            raise ValueError("Artifact paths must be unique flat filenames")
        names.add(name)
        if type(item["size"]) is not int or item["size"] < 0:
            raise ValueError("Invalid artifact size")
        if not isinstance(item["digest"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", item["digest"]):
            raise ValueError("Invalid artifact digest")
    if not isinstance(value["entrypoint"], str) or value["entrypoint"] not in names:
        raise ValueError("Entrypoint must reference a declared artifact")
    return value


def read_manifest(directory):
    return validate(json.loads((Path(directory) / "manifest.json").read_text()))


def verify(directory, manifest=None):
    directory = Path(directory)
    manifest = read_manifest(directory) if manifest is None else validate(manifest)
    for item in manifest["artifacts"]:
        path = directory / item["path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or nonregular artifact: {item['path']}")
        if path.stat().st_size != item["size"] or file_digest(path) != item["digest"]:
            raise ValueError(f"Integrity failure: {item['path']}")
    return manifest


def pack(source, destination, contract, files, entrypoint):
    """Package arbitrary local files; no architecture interpretation here."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError("Destination already exists")
    manifest = {"schema": SCHEMA, "contract": contract, "entrypoint": entrypoint, "artifacts": [
        {"path": name, "size": 0, "digest": "sha256:" + "0" * 64} for name in sorted(files)
    ]}
    validate(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary)
        for item in manifest["artifacts"]:
            original, copied = source / item["path"], stage / item["path"]
            if original.is_symlink() or not original.is_file():
                raise ValueError(f"Missing or nonregular source: {item['path']}")
            shutil.copyfile(original, copied)
            item.update(size=copied.stat().st_size, digest=file_digest(copied))
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify(stage)
        stage.rename(destination)
    return manifest


def execute(directory, request=None, device="cpu", executable="affon", *, engine, cache_dir=None, explain=False):
    manifest = read_manifest(directory)
    contract = manifest["contract"]
    raw_onnx = contract == ONNX_CONTRACT
    required = (manifest["entrypoint"],) if raw_onnx else ADAPTERS.get(contract)
    if required is None:
        raise ValueError(f"Unsupported model contract: {contract}")
    if engine not in (("affon", "onnxruntime") if raw_onnx else ("affon",)):
        raise ValueError(f"Engine {engine} does not support contract {contract}")
    if set(required) != {item["path"] for item in manifest["artifacts"]}:
        raise ValueError("Artifacts do not match the runtime adapter's file contract")
    if not raw_onnx and manifest["entrypoint"] != required[0]:
        raise ValueError("Entrypoint does not match the model contract")
    if device not in ("cpu", "metal"):
        raise ValueError("This experiment supports cpu and metal only")
    if engine == "onnxruntime" and device != "cpu":
        raise ValueError("ONNX Runtime adapter supports CPU only; no device fallback")
    cache = Path(cache_dir or Path(tempfile.gettempdir()) / "affon-model-package-cache").resolve()
    if cache.is_relative_to(Path(directory).resolve()):
        raise ValueError("Derived cache must be outside the model package")
    # Execute a private verified snapshot; undeclared neighboring files cannot
    # change what the model loader sees. Copying is deliberately simple in v0.
    with tempfile.TemporaryDirectory(prefix="affon-model-package-") as temporary:
        stage = Path(temporary)
        model = stage / "model"
        model.mkdir()
        for item in manifest["artifacts"]:
            original = Path(directory) / item["path"]
            if original.is_symlink() or not original.is_file():
                raise ValueError(f"Missing or nonregular artifact: {item['path']}")
            shutil.copyfile(original, model / item["path"])
        verify(model, manifest)
        if not explain:
            (stage / "request.json").write_text(json.dumps(request, allow_nan=False))
        if raw_onnx:
            mode = "prepare" if engine == "affon" else ("explain" if explain else "run")
            completed = subprocess.run([
                sys.executable, str(ROOT / "apps/model-package/onnx-engine.py"), mode,
                str(model / manifest["entrypoint"]), str(stage), str(cache),
            ], cwd=stage, capture_output=True, text=True)
            if completed.returncode:
                raise ValueError(f"{engine}: {completed.stderr.strip()}")
            details = json.loads((stage / "engine.json").read_text())
            if explain:
                return {"contract": contract, "engine": engine, "device": device, **details}
            if engine == "onnxruntime":
                return json.loads((stage / "response.json").read_text())
            model = stage / "prepared"
        elif explain:
            return {"contract": contract, "engine": engine, "device": device,
                    "artifacts_verified": True, "execution_checked": False,
                    "note": "Model-specific compatibility is checked during loading"}
        env = {**os.environ, "AFFON_DEVICE": device, "MODEL_PACKAGE_STAGE": str(stage),
               "MODEL_PACKAGE_MODEL_DIR": str(model),
               "MODEL_PACKAGE_RUNTIME": "affon/prepared-onnx/v1" if raw_onnx else contract}
        subprocess.run([executable, str(ROOT / "apps/model-package/src/run.ts")],
                       cwd=ROOT, env=env, check=True)
        return json.loads((stage / "response.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("pack")
    build.add_argument("source")
    build.add_argument("destination")
    build.add_argument("--contract", required=True)
    build.add_argument("--entrypoint", required=True)
    build.add_argument("--files", nargs="+", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("directory")
    run = commands.add_parser("run")
    run.add_argument("directory")
    run.add_argument("--input", required=True)
    run.add_argument("--device", choices=["cpu", "metal"], default="cpu")
    run.add_argument("--affon", default="affon")
    run.add_argument("--engine", choices=["affon", "onnxruntime"], required=True)
    run.add_argument("--cache-dir")
    explain = commands.add_parser("explain")
    explain.add_argument("directory")
    explain.add_argument("--engine", choices=["affon", "onnxruntime"], required=True)
    explain.add_argument("--device", choices=["cpu", "metal"], default="cpu")
    explain.add_argument("--cache-dir")
    args = parser.parse_args()
    try:
        if args.command == "pack":
            result = pack(args.source, args.destination, args.contract, args.files, args.entrypoint)
        elif args.command == "inspect":
            result = verify(args.directory)
        elif args.command == "explain":
            result = execute(args.directory, engine=args.engine, device=args.device,
                             cache_dir=args.cache_dir, explain=True)
        else:
            result = execute(args.directory, json.loads(Path(args.input).read_text()), args.device, args.affon,
                             engine=args.engine, cache_dir=args.cache_dir)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"model-package: {error}\n")


if __name__ == "__main__":
    main()
