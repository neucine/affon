"""Sequential HTTP correctness and serving measurements; no cluster required."""
import argparse
import copy
import hashlib
import http.client
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
APP = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def request(port, method, path, body=None, content_type="application/json", declared_length=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    try:
        try:
            headers = {"Content-Type": content_type, "Connection": "close"}
            if declared_length is not None:
                headers["Content-Length"] = str(declared_length)
            connection.request(method, path, body=body, headers=headers)
        except BrokenPipeError:
            # A server may send 413 and close before the oversized body finishes.
            pass
        result = connection.getresponse()
        raw = result.read()
        try:
            value = json.loads(raw) if raw else None
        except ValueError:
            value = raw.decode(errors="replace")
        return result.status, value
    finally:
        connection.close()


def encode(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


def rss(pid):
    # Sampled resident memory; this is deliberately not described as peak RSS.
    result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=True)
    return int(result.stdout.strip()) * 1024


def cases():
    result = []
    for i, (height, width) in enumerate([(224, 224), (260, 320), (320, 260)]):
        y, x, c = np.indices((height, width, 3))
        pixels = ((x * 3 + y * 5 + c * 47) % 256).astype(np.uint8)
        result.append({"id": f"case-{i}", "inputs": [{"name": "rgb", "datatype": "UINT8", "shape": [1, height, width, 3], "data": pixels.reshape(-1).tolist()}]})
    return result


def boundary_checks(port):
    count = 0
    def check(method, path, expected, payload=None, content_type="application/json"):
        nonlocal count
        status, response = request(port, method, path, payload, content_type)
        if status != expected:
            raise AssertionError(f"{method} {path}: expected {expected}, got {status}: {response}")
        count += 1
        return response
    for path in ["/v2/health/live", "/v2/health/ready", "/v2/models/vision/ready"]:
        check("GET", path, 200)
    metadata = check("GET", "/v2/models/vision", 200)
    assert metadata["inputs"] == [{"name": "rgb", "datatype": "UINT8", "shape": [1, -1, -1, 3]}]
    check("GET", "/v2/models/missing/ready", 404)
    check("GET", "/v2/models/vision/infer", 405)
    check("POST", "/v2/models/vision/infer", 415, b"{}", "text/plain")
    for bad in [b"{", b"null", b"[]", b"{}"]:
        check("POST", "/v2/models/vision/infer", 400, bad)
    valid = {"inputs": [{"name": "rgb", "datatype": "UINT8", "shape": [1, 1, 1, 3], "data": [0, 0, 0]}]}
    for change in [{"name": "pixels"}, {"datatype": "FP32"}, {"shape": [2, 1, 1, 3]},
                   {"shape": [1, 513, 1, 3]}, {"shape": [1, 1, 512, 3]}, {"data": [0]}, {"data": [-1, 0, 0]},
                   {"data": [True, 0, 0]}, {"data": [1.5, 0, 0]}]:
        bad = copy.deepcopy(valid)
        bad["inputs"][0].update(change)
        check("POST", "/v2/models/vision/infer", 400, encode(bad))
    # Verify header-based early rejection without racing a continuing upload.
    status, _ = request(port, "POST", "/v2/models/vision/infer", declared_length=4 * 1024 * 1024 + 1)
    assert status == 413, f"Expected early 413, got {status}"
    count += 1
    # Rejections must not poison subsequent valid inference or the listener.
    check("GET", "/v2/health/ready", 200)
    return count


def run_engine(engine, command, model, temporary, out, samples, iterations, warmups, threads=4, memory_iterations=0):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "MODEL_DIR": str(model), "MODEL_NAME": "vision", "HOST": "127.0.0.1", "PORT": str(port),
           "AFFON_DEVICE": "cpu", "RUNTIME_PACKAGE_PATH": str(REPO / "packages"),
           "AFFON_CPU_THREADS": str(threads), "ORT_THREADS": str(threads), "VECLIB_MAXIMUM_THREADS": str(threads), "OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads)}
    with (out / f"{engine}.log").open("w") as log:
        start = time.perf_counter()
        process = subprocess.Popen(command, cwd=temporary, env=env, stdout=log, stderr=log)
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{engine} exited during startup; see {out / (engine + '.log')}")
                try:
                    if request(port, "GET", "/v2/health/ready")[0] == 200:
                        break
                except (OSError, http.client.HTTPException):
                    pass
                if time.perf_counter() - start > 120:
                    raise RuntimeError(f"{engine} startup timed out")
                time.sleep(.01)
            report = {"startup_to_ready_ms": (time.perf_counter() - start) * 1000, "rss_ready_bytes": rss(process.pid)}
            report["boundary_checks"] = boundary_checks(port)
            outputs = []
            for sample in samples:
                status, result = request(port, "POST", "/v2/models/vision/infer", encode(sample))
                if status != 200:
                    raise AssertionError(f"{engine} inference: {status}: {result}")
                assert result["id"] == sample["id"] and result["model_name"] == "vision"
                assert result["outputs"][0]["datatype"] == "FP32" and result["outputs"][0]["name"] == "logits"
                outputs.append(result["outputs"][0])
            # Warm and measure the landscape image with identical serialized input.
            body = encode(samples[1])
            for _ in range(warmups):
                if request(port, "POST", "/v2/models/vision/infer", body)[0] != 200:
                    raise AssertionError("Warmup failed")
            report["rss_warm_bytes"] = rss(process.pid)
            latency, preprocess, inference = [], [], []
            beginning = time.perf_counter()
            for _ in range(iterations):
                tick = time.perf_counter()
                status, result = request(port, "POST", "/v2/models/vision/infer", body)
                latency.append((time.perf_counter() - tick) * 1000)
                if status != 200:
                    raise AssertionError(f"Measured request failed: {status}: {result}")
                preprocess.append(result["parameters"]["preprocess_ms"])
                inference.append(result["parameters"]["inference_ms"])
            elapsed = time.perf_counter() - beginning
            report.update(iterations=iterations, warmups=warmups, request_bytes=len(body),
                          latency_ms=latency, median_ms=float(np.median(latency)), p95_ms=float(np.percentile(latency, 95)),
                          serial_requests_per_second=iterations / elapsed,
                          median_preprocess_ms=float(np.median(preprocess)), median_inference_ms=float(np.median(inference)),
                          rss_after_bytes=rss(process.pid))
            if memory_iterations:
                memory = [{"iteration": 0, "rss_bytes": rss(process.pid)}]
                for i in range(1, memory_iterations + 1):
                    status, _ = request(port, "POST", "/v2/models/vision/infer", body)
                    if status != 200:
                        raise AssertionError("Memory probe request failed")
                    if i % 50 == 0 or i == memory_iterations:
                        memory.append({"iteration": i, "rss_bytes": rss(process.pid)})
                report["memory_probe_rss"] = memory
            return report, outputs
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--affon", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--memory-iterations", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmups < 0 or args.threads < 1 or args.memory_iterations < 0:
        parser.error("iterations must be positive and warmups nonnegative")
    source, binary, out = Path(args.model_dir).resolve(), Path(args.affon).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"platform": platform.platform(), "python": sys.version, "affon_sha256": digest(binary),
              "model_sha256": digest(source / "model.onnx"), "processor_sha256": digest(source / "source/preprocessor_config.json"),
              "versions": {key: importlib.metadata.version(key) for key in ["onnx", "onnxruntime", "numpy", "pillow", "safetensors"]},
              "source_sha256": {str(p.relative_to(REPO)): digest(p) for p in [APP / "src/server.ts", APP / "baseline.py", Path(__file__), REPO / "packages/@affon/onnx/tools/convert.py"]},
              "device": "cpu", "threads_requested": args.threads, "runs": {}, "parity": []}
    samples = cases()
    with tempfile.TemporaryDirectory(prefix="affon-vision-serving-") as temporary:
        temp = Path(temporary)
        prepared = temp / "prepared"
        tick = time.perf_counter()
        conversion = subprocess.run([sys.executable, str(REPO / "packages/@affon/onnx/tools/convert.py"),
                                     str(source / "model.onnx"), str(prepared)], cwd=temp, capture_output=True, text=True)
        if conversion.returncode:
            raise RuntimeError(conversion.stderr)
        report["affon_preparation_ms"] = (time.perf_counter() - tick) * 1000
        shutil.copytree(source / "source", prepared / "source", ignore=shutil.ignore_patterns("*.safetensors"))
        report["affon_prepared_bytes"] = sum((prepared / name).stat().st_size for name in ["graph.json", "weights.safetensors"])
        all_outputs = {}
        for engine, command, model in [
            ("onnxruntime", [sys.executable, str(APP / "baseline.py")], source),
            ("affon", [str(binary), str(APP / "src/server.ts")], prepared),
        ]:
            print(f"Starting {engine}", flush=True)
            report["runs"][engine], all_outputs[engine] = run_engine(engine, command, model, temp, out, samples, args.iterations, args.warmups, args.threads, args.memory_iterations)
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        for i, sample in enumerate(samples):
            native, baseline = all_outputs["affon"][i], all_outputs["onnxruntime"][i]
            a, b = np.asarray(native["data"]), np.asarray(baseline["data"])
            passed = native["shape"] == baseline["shape"] and a.shape == b.shape and np.isfinite(a).all() and np.allclose(a, b, atol=1e-4, rtol=1e-4)
            report["parity"].append({"case": sample["id"], "input_shape": sample["inputs"][0]["shape"],
                                     "passed": bool(passed), "max_absolute_error": float(np.abs(a-b).max()),
                                     "affon_top1": int(a.argmax()), "onnxruntime_top1": int(b.argmax())})
        report["passed"] = all(case["passed"] for case in report["parity"])
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit("End-to-end output parity failed")


if __name__ == "__main__":
    main()
