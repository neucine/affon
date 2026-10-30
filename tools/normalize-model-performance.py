#!/usr/bin/env python3
"""Convert existing model benchmark reports to the shared comparison envelope."""
import argparse
import json
from pathlib import Path


def vision(path, target):
    raw = json.loads(Path(path).read_text())
    if raw.get("passed") is not True or set(raw.get("runs", {})) != {"affon", "onnxruntime"}:
        raise ValueError("vision report lacks complete passing parity and both engines")
    if not raw.get("parity") or not all(row.get("passed") is True for row in raw["parity"]):
        raise ValueError("vision report lacks passing per-input parity")
    outputs = {}
    for engine in ("affon", "onnxruntime"):
        run = raw["runs"][engine]
        contract = {"model_sha256": raw["model_sha256"], "processor_sha256": raw["processor_sha256"],
                    "phase": "http_request", "timing_scope": "serialized RGB request to decoded JSON response",
                    "input_shape": [1, 260, 320, 3], "batch": 1, "dtype": "rgb8",
                    "threads_requested": raw["threads_requested"], "iterations": run["iterations"],
                    "warmups": run["warmups"], "request_bytes": run["request_bytes"]}
        outputs[engine] = {"schema_version": 1, "suite": "affon_model_performance", "target": target,
                           "engine": engine, "provenance": {"platform": raw.get("platform"),
                           "model_sha256": raw["model_sha256"], "affon_sha256": raw.get("affon_sha256"),
                           "versions": raw.get("versions")},
                           "cases": [{"id": "mobilenetv2/cpu/http/260x320",
                           "correct": True, "contract": contract, "samples_ms": run["latency_ms"]}]}
    return outputs


def hf(directory, target):
    root = Path(directory)
    run = json.loads((root / "run.json").read_text())
    cases = []
    for item in run["runs"]:
        report = json.loads((root / item["report"]).read_text())
        family, device = report["family"], report["device"]
        metric = "generate_ms" if family == "gpt2" else "forward_with_readback_ms"
        samples = report["metrics"][metric]["samples"]
        if report.get("format") != "affon-inference-benchmark/v1" or report["measured_runs"] != len(samples):
            raise ValueError(f"incomplete HF report: {item['report']}")
        cases.append({"id": f"{family}/{device}/{metric}", "correct": True,
                      "contract": {"family": family, "device": device, "model_id": report["model_id"],
                                   "revision": report["revision"], "workload": report["workload"],
                                   "timing_scope": metric, "build_profile": run["build_profile"],
                                   "measured_runs": report["measured_runs"], "warmups": 1},
                      "samples_ms": samples})
    if len(cases) != len(run["runs"]) or not cases:
        raise ValueError("incomplete HF run")
    return {"affon": {"schema_version": 1, "suite": "affon_model_performance", "target": target,
                      "engine": "affon", "provenance": {"platform": run.get("platform"),
                      "affon_sha256": run.get("affon_sha256"),
                      "gpt2_source_sha256": run.get("gpt2_source_sha256")}, "cases": cases}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, choices=["vision", "hf"])
    p.add_argument("--input", required=True)
    p.add_argument("--target", required=True, help="Stable ID for this exact machine configuration")
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    outputs = vision(a.input, a.target) if a.source == "vision" else hf(a.input, a.target)
    destination = Path(a.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for engine, report in outputs.items():
        path = destination / f"{engine}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()
