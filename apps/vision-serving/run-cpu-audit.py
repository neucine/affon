"""Repeat fixed MobileNetV2 CPU phases in fresh processes at one and four threads."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

APP = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-dir", required=True)
parser.add_argument("--affon", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--runs", type=int, default=30)
parser.add_argument("--batch", type=int, default=10)
args = parser.parse_args()
if min(args.repeats, args.runs, args.batch) < 1:
    parser.error("repeats, runs and batch must be positive")
out = Path(args.output).resolve()
out.mkdir(parents=True, exist_ok=True)
records = []
for repeat in range(1, args.repeats + 1):
    for threads in (1, 4):
        directory = out / f"t{threads}-run{repeat}"
        command = [
            sys.executable,
            str(APP / "profile-cpu.py"),
            "--model-dir", args.model_dir,
            "--affon", args.affon,
            "--output", str(directory),
            "--threads", str(threads),
            "--runs", str(args.runs),
            "--batch", str(args.batch),
        ]
        print(f"Profiling {threads} thread(s), run {repeat}", flush=True)
        with (out / f"t{threads}-run{repeat}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((directory / "summary.json").read_text())
        affon = result["affon"]["graph_ms"]["median"]
        ort = result["onnxruntime"]["graph_ms"]["median"]
        records.append({
            "threads": threads,
            "run": repeat,
            "graph_ms": {"affon": affon, "onnxruntime": ort},
            "graph_ratio": affon / ort,
            "preprocessing_ms": {
                "affon": result["affon"]["preprocessing_ms"]["median"],
                "onnxruntime": result["onnxruntime"]["preprocessing_ms"]["median"],
            },
            "max_absolute_error": result["max_absolute_error"],
            "parity": result["parity"],
            "native_peak_rss_bytes": result["native_process_usage"]["peak_rss_bytes"],
        })
    (out / "audit-summary.json").write_text(json.dumps(records, indent=2) + "\n")
summary = {
    "records": records,
    "graph_gate": {
        str(threads): {
            "max_ratio": max(r["graph_ratio"] for r in records if r["threads"] == threads),
            "passed": all(r["graph_ratio"] <= 2 and r["parity"] for r in records if r["threads"] == threads),
            "median_affon_ms": statistics.median(r["graph_ms"]["affon"] for r in records if r["threads"] == threads),
            "median_onnxruntime_ms": statistics.median(r["graph_ms"]["onnxruntime"] for r in records if r["threads"] == threads),
        }
        for threads in (1, 4)
    },
}
(out / "audit-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary["graph_gate"], indent=2))
if not all(item["parity"] for item in records):
    raise SystemExit("Numerical parity failed")
if not all(gate["passed"] for gate in summary["graph_gate"].values()):
    raise SystemExit("CPU graph performance gate failed")
