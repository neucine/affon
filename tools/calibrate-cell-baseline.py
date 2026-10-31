#!/usr/bin/env python3
"""Measure repeatability of resident Affon cell timings across fresh processes."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def machine_state():
    power = subprocess.check_output(["pmset", "-g", "ps"], text=True).splitlines()[0]
    thermal = subprocess.check_output(["pmset", "-g", "therm"], text=True).strip()
    return {"power_source": power, "thermal_status": thermal}


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / "tools" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calibrate(binary, case, runs, samples, iterations, maximum_ratio, maximum_absolute_ms, warmup_calls=20):
    started_at = datetime.now(timezone.utc).isoformat()
    state_at_start = machine_state()
    benchmark = load_tool("benchmark-cells-pytorch.py")
    probe = load_tool("probe-capabilities.py")
    reference = probe.expected(case)
    results = []
    selected_path = None
    for index in range(runs):
        result = benchmark.affon(binary, case, iterations, samples, warmup_calls=warmup_calls)
        if selected_path is None:
            selected_path = result["selected_path"]
        if result["selected_path"] != selected_path:
            raise ValueError(f"{case['id']}: selected path changed between baseline runs")
        if result["output_count"] != len(reference):
            raise ValueError(f"{case['id']}: output length changed")
        tolerance = 1e-6 * max(1, max(abs(value) for value in reference))
        error = max(abs(item["value"] - reference[item["index"]]) for item in result["output_samples"])
        checksum_error = abs(result["output_checksum"] - sum(reference))
        if error > tolerance or checksum_error > tolerance * len(reference):
            raise ValueError(f"{case['id']}: baseline output failed independent reference")
        results.append({"run": index, "samples_ms": result["samples_ms"],
                        "median_ms": statistics.median(result["samples_ms"]),
                        "execution_median_ms": statistics.median(result["execution_ms"]),
                        "readback_median_ms": statistics.median(result["readback_ms"]),
                        "max_abs_error": error, "checksum_error": checksum_error})
    medians = [row["median_ms"] for row in results]
    spread_ms = max(medians) - min(medians)
    spread_ratio = max(medians) / min(medians)
    stable = spread_ratio <= maximum_ratio or spread_ms <= maximum_absolute_ms
    state_at_end = machine_state()
    return {"schema_version": 1, "suite": "affon_cell_baseline_calibration",
            "started_at": started_at, "ended_at": datetime.now(timezone.utc).isoformat(),
            "machine_state_at_start": state_at_start, "machine_state_at_end": state_at_end,
            "cell_id": case["id"], "target": f"{subprocess.check_output(['sysctl', '-n', 'hw.model'], text=True).strip()}/{platform.mac_ver()[0]}/{platform.machine()}",
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "harness_sha256": hashlib.sha256((ROOT / "test/parity/workers/affon-benchmark.ts").read_bytes()).hexdigest(),
            "contract": {"shape": case["shape"], "dtype": case["dtype"], "layout": "dense_zero_offset",
                         "compile_options": {"dense_inputs": bool(case.get("dense_inputs", False))},
                         "timing_scope": "resident operation, synchronized full host materialization, output allocation and disposal included",
                         "iterations_per_sample": iterations, "samples_per_run": samples,
                         "warmup_calls": warmup_calls, "independent_processes": runs},
            "selected_path": selected_path, "runs": results,
            "calibration": {"stable": stable, "maximum_allowed_ratio": maximum_ratio,
                            "maximum_allowed_absolute_ms": maximum_absolute_ms,
                            "observed_run_median_ratio": spread_ratio,
                            "observed_run_median_span_ms": spread_ms}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "zig-out/bin/affon")
    parser.add_argument("--catalog", type=Path, default=ROOT / "test/benchmarks/capability-probes.json")
    parser.add_argument("--only", required=True, help="Comma-separated cell IDs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup-calls", type=int, default=20)
    parser.add_argument("--maximum-ratio", type=float, default=1.10)
    parser.add_argument("--maximum-absolute-ms", type=float, default=0.05)
    args = parser.parse_args()
    if args.runs < 3 or args.samples < 3 or args.iterations < 1 or args.warmup_calls < 0 or not math.isfinite(args.maximum_ratio) or args.maximum_ratio < 1 or not math.isfinite(args.maximum_absolute_ms) or args.maximum_absolute_ms < 0:
        parser.error("need three runs and samples, positive iterations, and valid spread limits")
    if not args.binary.is_file():
        parser.error("Affon binary missing")
    catalog = json.loads(args.catalog.read_text())
    wanted = args.only.split(",")
    cases = {case["id"]: case for case in catalog["cases"]}
    if len(set(wanted)) != len(wanted) or any(key not in cases for key in wanted):
        parser.error("unknown or duplicate cell ID")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in wanted:
        report = calibrate(args.binary, cases[key], args.runs, args.samples, args.iterations,
                           args.maximum_ratio, args.maximum_absolute_ms, args.warmup_calls)
        path = args.output_dir / (key.replace("/", "__") + ".json")
        path.write_text(json.dumps(report, indent=2) + "\n")
        state = "stable" if report["calibration"]["stable"] else "unstable"
        print(f"{key}: {state}, run median spread {report['calibration']['observed_run_median_ratio']:.2f}x / {report['calibration']['observed_run_median_span_ms']:.4f} ms", flush=True)


if __name__ == "__main__":
    main()
