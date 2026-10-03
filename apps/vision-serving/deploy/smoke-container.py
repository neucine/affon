"""Check the Linux serving image on loopback, then remove its test container."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

app = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(app))
from benchmark import boundary_checks, cases, encode, request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--image", default="affon-vision:experiment")
parser.add_argument("--model-dir", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
source, out = Path(args.model_dir).resolve(), Path(args.output).resolve()
out.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="affon-container-model-") as temp:
    temp = Path(temp)
    # Model artifacts are mounted read-only and served as UID 65532.
    temp.chmod(0o755)
    for name in ["graph.json", "weights.safetensors"]:
        shutil.copyfile(source / name, temp / name)
        (temp / name).chmod(0o644)
    (temp / "source").mkdir()
    shutil.copyfile(source / "source/preprocessor_config.json", temp / "source/preprocessor_config.json")
    (temp / "source/preprocessor_config.json").chmod(0o644)
    container = subprocess.check_output([
        "docker", "run", "-d", "--rm", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "-p", "127.0.0.1::8080",
        "--mount", f"type=bind,src={temp},dst=/mnt/models,readonly", args.image,
    ], text=True).strip()
    try:
        port = int(subprocess.check_output(["docker", "port", container, "8080/tcp"], text=True).strip().rsplit(":", 1)[1])
        start = time.perf_counter()
        while True:
            try:
                if request(port, "GET", "/v2/health/ready")[0] == 200:
                    break
            except (OSError, ConnectionError):
                pass
            running = subprocess.check_output(["docker", "inspect", "-f", "{{.State.Running}}", container], text=True).strip()
            if running != "true" or time.perf_counter() - start > 120:
                raise RuntimeError("Container did not become ready")
            time.sleep(.1)
        checks = boundary_checks(port)
        sample = cases()[1]
        status, actual = request(port, "POST", "/v2/models/vision/infer", encode(sample))
        assert status == 200 and actual["id"] == sample["id"], actual
        os.environ["MODEL_DIR"] = str(source)
        # Import the independent baseline outside the checkout: ORT may create
        # a local telemetry file during import on this version.
        os.chdir(temp)
        spec = importlib.util.spec_from_file_location("vision_baseline", app / "baseline.py")
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        shape, pixels = baseline.validate(sample)
        expected = baseline.session.run(None, {"pixels": baseline.preprocess(np.asarray(pixels, dtype=np.uint8).reshape(shape[1:]))})[0]
        native = np.asarray(actual["outputs"][0]["data"])
        passed = actual["outputs"][0]["shape"] == list(expected.shape) and np.allclose(native, expected.reshape(-1), atol=1e-4, rtol=1e-4)
        report = {"image": args.image, "image_id": subprocess.check_output(["docker", "inspect", "-f", "{{.Image}}", container], text=True).strip(),
                  "boundary_checks": checks, "parity_passed": bool(passed),
                  "max_absolute_error": float(np.abs(native-expected.reshape(-1)).max()),
                  "top1": int(native.argmax()), "kubernetes_deployed": False}
        (out / "container-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if not passed:
            raise AssertionError("Container output mismatch")
    finally:
        logs = subprocess.run(["docker", "logs", container], capture_output=True, text=True)
        (out / "container.log").write_text(logs.stdout + logs.stderr)
        subprocess.run(["docker", "stop", "-t", "2", container], capture_output=True)
