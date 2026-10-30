#!/usr/bin/env python3
"""Evaluate independent calibration sessions before accepting baseline evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def spread(values):
    return {"ratio": max(values) / min(values), "span_ms": max(values) - min(values)}


def within_limit(measurement, ratio_limit, span_limit):
    return measurement["ratio"] <= ratio_limit or measurement["span_ms"] <= span_limit


def evaluate(reports, ratio_limit=1.10, span_limit=0.05, sources=None):
    if len(reports) < 2:
        raise ValueError("at least two independent sessions are required")
    if not math.isfinite(ratio_limit) or ratio_limit < 1 or not math.isfinite(span_limit) or span_limit < 0:
        raise ValueError("invalid spread limits")
    reasons = []
    expected = reports[0]
    keys = ("cell_id", "target", "binary_sha256", "harness_sha256", "contract", "selected_path")
    for index, report in enumerate(reports):
        if report.get("suite") != "affon_cell_baseline_calibration":
            raise ValueError(f"session {index + 1}: unexpected suite")
        for key in keys:
            if report.get(key) is None or report.get(key) != expected.get(key):
                reasons.append(f"session {index + 1}: missing or mismatched {key}")
        if len(report.get("runs", [])) < 3:
            reasons.append(f"session {index + 1}: fewer than three process runs")
        state_start = report.get("machine_state_at_start")
        state_end = report.get("machine_state_at_end")
        if not state_start or not state_end or state_start != state_end:
            reasons.append(f"session {index + 1}: power or thermal state missing or changed")
        elif state_start != expected.get("machine_state_at_start"):
            reasons.append(f"session {index + 1}: power or thermal state differs between sessions")
        if not report.get("started_at") or not report.get("ended_at"):
            reasons.append(f"session {index + 1}: session timestamps missing")
        medians = [row.get("median_ms") for row in report.get("runs", [])]
        if not medians or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in medians):
            raise ValueError(f"session {index + 1}: invalid process medians")
        if not within_limit(spread(medians), ratio_limit, span_limit):
            reasons.append(f"session {index + 1}: process medians exceed repeatability limit")
    timestamps = [report.get("started_at") for report in reports]
    if len(set(timestamps)) != len(timestamps):
        reasons.append("session timestamps are not distinct")
    session_medians = [statistics.median(row["median_ms"] for row in report["runs"]) for report in reports]
    between = spread(session_medians)
    if not within_limit(between, ratio_limit, span_limit):
        reasons.append("session medians exceed repeatability limit")
    if sources is not None and len(sources) != len(reports):
        raise ValueError("session source count differs from reports")
    artifacts = ([{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for path in sources] if sources is not None else [])
    return {"schema_version": 1, "suite": "affon_cell_baseline_evaluation",
            "cell_id": expected.get("cell_id"), "accepted": not reasons,
            "target": expected.get("target"), "binary_sha256": expected.get("binary_sha256"),
            "harness_sha256": expected.get("harness_sha256"),
            "contract": expected.get("contract"), "selected_path": expected.get("selected_path"),
            "session_medians_ms": session_medians, "between_session": between,
            "session_artifacts": artifacts,
            "limits": {"maximum_ratio": ratio_limit, "maximum_span_ms": span_limit},
            "reasons": reasons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path, help="JSON calibration files for one cell")
    parser.add_argument("--maximum-ratio", type=float, default=1.10)
    parser.add_argument("--maximum-absolute-ms", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.sessions]
    result = evaluate(reports, args.maximum_ratio, args.maximum_absolute_ms, args.sessions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['cell_id']}: {'accepted' if result['accepted'] else 'rejected'}; "
          f"between-session spread {result['between_session']['ratio']:.2f}x / "
          f"{result['between_session']['span_ms']:.4f} ms")
    for reason in result["reasons"]:
        print(f"  - {reason}")


if __name__ == "__main__":
    main()
