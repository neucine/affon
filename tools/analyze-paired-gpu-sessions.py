#!/usr/bin/env python3
"""Compare alternating Metal cell sessions without treating them as baselines."""
import argparse
import json
import math
from pathlib import Path
import statistics


def analyze(paths):
    if len(paths) < 4 or len(paths) % 2:
        raise ValueError("provide at least two alternating pairs")
    reports = [json.loads(path.read_text()) for path in paths]
    first_by_cell = {}
    cells = []
    rows = []
    for index, (path, report) in enumerate(zip(paths, reports)):
        if report.get("suite") != "affon_cell_gpu_variability":
            raise ValueError(f"unexpected suite: {path}")
        cell = report["cell_id"]
        if index < 2:
            if cell in cells:
                raise ValueError("the first pair must contain two different cells")
            cells.append(cell)
        elif cell != cells[index % 2]:
            raise ValueError("sessions must alternate the same two cells")
        if cell in first_by_cell:
            first = first_by_cell[cell]
            for key in ("binary_sha256", "interposer_sha256", "harness_sha256", "contract"):
                if report.get(key) != first.get(key):
                    raise ValueError(f"{cell}: {key} changed")
        else:
            first_by_cell[cell] = report
        paths_selected = {json.dumps(run["selected_path"], sort_keys=True) for run in report["runs"]}
        if len(paths_selected) != 1 or (cell in first_by_cell and paths_selected != {
                json.dumps(run["selected_path"], sort_keys=True) for run in first_by_cell[cell]["runs"]}):
            raise ValueError(f"{cell}: selected path changed")
        gpu = [run["gpu_command_median_ms"] for run in report["runs"]]
        host = [run["host_total_median_ms"] for run in report["runs"]]
        if any(not math.isfinite(value) or value <= 0 for value in gpu + host):
            raise ValueError(f"{cell}: invalid timing")
        rows.append({"pair": index // 2 + 1, "cell_id": cell, "source": str(path),
                     "gpu_median_ms": statistics.median(gpu), "host_median_ms": statistics.median(host),
                     "gpu_process_ratio": max(gpu) / min(gpu),
                     "recorded_at": report.get("recorded_at")})
    changes = []
    for cell in cells:
        values = [row["gpu_median_ms"] for row in rows if row["cell_id"] == cell]
        changes.append({"cell_id": cell, "gpu_session_ratio": max(values) / min(values)})
    return {"schema_version": 1, "suite": "affon_paired_gpu_diagnostic",
            "scope": "GPU command timing diagnostic; sessions are sequential and do not establish a simultaneous cause or accepted baseline.",
            "cells": cells, "sessions": rows, "changes": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path, help="Alternating A B A B session JSON files")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.sessions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for change in result["changes"]:
        print(f"{change['cell_id']}: {change['gpu_session_ratio']:.3f}x GPU session spread")


if __name__ == "__main__":
    main()
