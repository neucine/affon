#!/usr/bin/env python3
"""Compare host and Metal command timing spread across fresh Affon processes."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import plistlib
import statistics
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
INTERPOSER = ROOT / "tools/metal-command-timestamps.m"


def machine_snapshot():
    registry = plistlib.loads(subprocess.check_output(["ioreg", "-a", "-r", "-c", "AGXAccelerator", "-l"]))
    gpu = registry[0].get("PerformanceStatistics", {}) if registry else {}
    return {"recorded_at": datetime.now(timezone.utc).isoformat(),
            "power_source": subprocess.check_output(["pmset", "-g", "ps"], text=True).splitlines()[0],
            "thermal_status": subprocess.check_output(["pmset", "-g", "therm"], text=True).strip(),
            "gpu_utilization_percent": gpu.get("Device Utilization %"),
            "gpu_recovery_count": gpu.get("recoveryCount")}


def benchmark_module():
    spec = importlib.util.spec_from_file_location("cell_benchmark", ROOT / "tools/benchmark-cells-pytorch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile(binary, case, runs, samples, iterations, warmup_calls):
    benchmark = benchmark_module()
    spec = importlib.util.spec_from_file_location("cell_probe", ROOT / "tools/probe-capabilities.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    reference = probe.expected(case)
    with tempfile.TemporaryDirectory(prefix="affon-cell-gpu-") as folder:
        folder = Path(folder)
        library = folder / "timestamps.dylib"
        subprocess.run(["xcrun", "clang", "-dynamiclib", "-O2", "-fobjc-arc",
                        "-framework", "Foundation", "-framework", "Metal",
                        str(INTERPOSER), "-o", str(library)], check=True)
        rows = []
        for index in range(runs):
            state_before = machine_snapshot()
            events_path = folder / f"run-{index}.jsonl"
            result = benchmark.affon(binary, case, iterations, samples,
                                     warmup_calls=warmup_calls,
                                     extra_env={"AFFON_GPU_PROFILE_PATH": str(events_path),
                                                "DYLD_INSERT_LIBRARIES": str(library)})
            if result["output_count"] != len(reference):
                raise ValueError(f"{case['id']}: output length mismatch")
            tolerance = 1e-6 * max(1, max(abs(value) for value in reference))
            error = max(abs(item["value"] - reference[item["index"]]) for item in result["output_samples"])
            if error > tolerance or abs(result["output_checksum"] - sum(reference)) > tolerance * len(reference):
                raise ValueError(f"{case['id']}: output failed independent reference")
            events = sorted((json.loads(line) for line in events_path.read_text().splitlines()),
                            key=lambda row: row["commit_ns"])
            expected = 1 + warmup_calls + samples * iterations
            if len(events) != expected or any(not math.isfinite(row["gpu_ns"]) or row["gpu_ns"] <= 0 for row in events):
                raise ValueError(f"{case['id']}: expected {expected} completed Metal commands, got {len(events)}")
            gpu_samples = [row["gpu_ns"] / 1e6 for row in events[-samples * iterations:]]
            state_after = machine_snapshot()
            rows.append({"run": index, "gpu_command_median_ms": statistics.median(gpu_samples),
                         "host_total_median_ms": statistics.median(result["samples_ms"]),
                         "host_execute_median_ms": statistics.median(result["execution_ms"]),
                         "host_readback_median_ms": statistics.median(result["readback_ms"]),
                         "command_count": len(events), "selected_path": result["selected_path"],
                         "machine_state_before": state_before, "machine_state_after": state_after})
    if len({json.dumps(row["selected_path"], sort_keys=True) for row in rows}) != 1:
        raise ValueError(f"{case['id']}: selected path changed across processes")
    def spread(key):
        values = [row[key] for row in rows]
        return {"smallest_ms": min(values), "largest_ms": max(values),
                "largest_over_smallest": max(values) / min(values)}
    return {"schema_version": 1, "suite": "affon_cell_gpu_variability",
            "cell_id": case["id"], "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "interposer_sha256": hashlib.sha256(INTERPOSER.read_bytes()).hexdigest(),
            "harness_sha256": hashlib.sha256((ROOT / "test/parity/workers/affon-benchmark.ts").read_bytes()).hexdigest(),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "power_source_at_end": rows[-1]["machine_state_after"]["power_source"],
            "thermal_status_at_end": rows[-1]["machine_state_after"]["thermal_status"],
            "contract": {"shape": case["shape"], "dtype": case["dtype"],
                         "iterations_per_sample": iterations, "samples_per_run": samples,
                         "warmup_calls": warmup_calls, "fresh_processes": runs},
            "scope": "Metal command buffer GPUStartTime to GPUEndTime; host totals include run and full output materialization. Profiling callback overhead affects host times; use separate wall benchmark for peer comparisons.",
            "runs": rows,
            "spread": {key: spread(key) for key in ("gpu_command_median_ms", "host_total_median_ms",
                                                    "host_execute_median_ms", "host_readback_median_ms")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "zig-out/bin/affon")
    parser.add_argument("--catalog", type=Path, default=ROOT / "test/benchmark/capability-probes.json")
    parser.add_argument("--cell", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup-calls", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 3 or args.samples < 3 or args.iterations < 1 or args.warmup_calls < 0:
        parser.error("need at least three runs and samples, positive iterations, and nonnegative warmup")
    cases = {item["id"]: item for item in json.loads(args.catalog.read_text())["cases"]}
    if args.cell not in cases or not args.cell.startswith("metal/"):
        parser.error("cell must be a declared Metal capability case")
    report = profile(args.binary.resolve(), cases[args.cell], args.runs, args.samples,
                     args.iterations, args.warmup_calls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "gpu_spread": report["spread"]["gpu_command_median_ms"]["largest_over_smallest"],
                      "host_spread": report["spread"]["host_total_median_ms"]["largest_over_smallest"]}))


if __name__ == "__main__":
    main()
