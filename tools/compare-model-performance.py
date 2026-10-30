#!/usr/bin/env python3
"""Compare normalized model timing reports; reject unmatched workloads."""
import argparse
import json
import math
from pathlib import Path
import statistics


def load(path):
    report = json.loads(Path(path).read_text())
    if report.get("schema_version") != 1 or report.get("suite") != "affon_model_performance":
        raise ValueError(f"{path}: unsupported model performance report")
    if not report.get("target") or not isinstance(report.get("cases"), list) or not report["cases"]:
        raise ValueError(f"{path}: missing target or cases")
    ids = set()
    for case in report["cases"]:
        key = case.get("id")
        if not isinstance(key, str) or not key or key in ids:
            raise ValueError(f"{path}: missing or duplicate case ID")
        ids.add(key)
        if case.get("correct") is not True or not isinstance(case.get("contract"), dict) or not case["contract"]:
            raise ValueError(f"{path}: {key}: correctness or workload contract missing")
        samples = case.get("samples_ms")
        if not isinstance(samples, list) or len(samples) < 3 or any(
            not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in samples
        ):
            raise ValueError(f"{path}: {key}: need at least three positive finite samples")
    return report


def compare(current, reference, *, kind, maximum_ratio, absolute_ms):
    if current["target"] != reference["target"]:
        raise ValueError("target mismatch")
    if current.get("engine") != "affon" or (kind == "self" and reference.get("engine") != "affon") or (kind == "peer" and reference.get("engine") == "affon"):
        raise ValueError("engine roles mismatch")
    left = {case["id"]: case for case in current["cases"]}
    right = {case["id"]: case for case in reference["cases"]}
    if left.keys() != right.keys():
        raise ValueError(f"case matrix mismatch: missing={sorted(right.keys()-left.keys())}, added={sorted(left.keys()-right.keys())}")
    rows = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if a["contract"] != b["contract"]:
            raise ValueError(f"{key}: workload or timing contract mismatch")
        am, bm = statistics.median(a["samples_ms"]), statistics.median(b["samples_ms"])
        ratio = am / bm
        # A noise allowance applies to self-regression only, never to a peer target.
        passed = ratio <= maximum_ratio or (kind == "self" and am - bm <= absolute_ms)
        rows.append({"id": key, "current_median_ms": am, "reference_median_ms": bm,
                     "ratio": ratio, "pass": passed})
    return {"schema_version": 1, "suite": "affon_model_performance_comparison", "kind": kind,
            "target": current["target"], "maximum_ratio": maximum_ratio,
            "absolute_noise_ms": absolute_ms if kind == "self" else None,
            "cases": rows, "gate_pass": all(row["pass"] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--kind", required=True, choices=["self", "peer"])
    parser.add_argument("--maximum-ratio", type=float, required=True)
    parser.add_argument("--absolute-noise-ms", type=float, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not math.isfinite(args.maximum_ratio) or args.maximum_ratio <= 0 or not math.isfinite(args.absolute_noise_ms) or args.absolute_noise_ms < 0:
        parser.error("limits must be finite and nonnegative; ratio must be positive")
    if args.kind == "peer" and args.absolute_noise_ms:
        parser.error("absolute noise allowance is only valid for self comparisons")
    result = compare(load(args.current), load(args.reference), kind=args.kind,
                     maximum_ratio=args.maximum_ratio, absolute_ms=args.absolute_noise_ms)
    encoded = json.dumps(result, indent=2) + "\n"
    print(encoded, end="")
    if args.output:
        Path(args.output).write_text(encoded)
    if not result["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
