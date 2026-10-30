#!/usr/bin/env python3
"""Summarize repeated cell comparisons without treating them as release gates."""
import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(reports, ratio_limit, kind="operator_peer"):
    if not math.isfinite(ratio_limit) or ratio_limit <= 0:
        raise ValueError("ratio limit must be positive and finite")
    rows, seen = [], set()
    for report in reports:
        if report.get("suite") != "affon_cell_benchmark" or report.get("kind") != kind:
            raise ValueError("unexpected cell report")
        cell = report["cell_id"]
        if cell in seen:
            raise ValueError(f"duplicate cell report: {cell}")
        seen.add(cell)
        if report.get("correctness", {}).get("passed") is not True:
            raise ValueError(f"{cell}: correctness failed")
        rounds = report.get("rounds")
        if not isinstance(rounds, list) or len(rounds) < 2:
            raise ValueError(f"{cell}: independent rounds missing")
        paired = []
        for item in rounds:
            a, b = item.get("affon_samples_ms"), item.get("peer_samples_ms")
            if not isinstance(a, list) or not isinstance(b, list) or len(a) < 3 or len(b) < 3 or any(
                not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0 for x in a + b
            ):
                raise ValueError(f"{cell}: raw samples missing")
            paired.append((statistics.median(a), statistics.median(b)))
        ratios = [a / b for a, b in paired]
        a_median, b_median = statistics.median(a for a, _ in paired), statistics.median(b for _, b in paired)
        status = "candidate_gap" if all(value > ratio_limit for value in ratios) else (
            "mixed_rounds" if any(value > ratio_limit for value in ratios) else "within_diagnostic_target")
        rows.append({"id": cell, "status": status, "affon_ms": a_median, "peer_ms": b_median,
                     "delta_ms": a_median - b_median, "ratio": a_median / b_median,
                     "round_ratios": ratios, "rounds": len(rounds)})
    rows.sort(key=lambda row: (row["status"] != "candidate_gap", -max(0, row["delta_ms"]), -row["ratio"]))
    return {"schema_version": 1, "suite": "affon_cell_diagnostic_summary", "kind": kind, "ratio_limit": ratio_limit,
            "caveat": "Candidate gaps need calibrated same-target noise and larger workload shapes before optimization priority is accepted.",
            "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ratio-limit", type=float, default=1.25)
    parser.add_argument("--kind", choices=["operator_peer", "program_peer"], default="operator_peer")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in sorted(args.directory.glob("*.json"))]
    if not reports:
        parser.error("no cell reports found")
    report = summarize(reports, args.ratio_limit, args.kind)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
