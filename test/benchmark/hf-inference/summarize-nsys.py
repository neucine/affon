#!/usr/bin/env python3
"""Summarize repeated cached decode windows from an Nsight Systems SQLite export."""
import argparse
import hashlib
import json
import sqlite3
import statistics
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path


def busy_ns(rows):
    start, end = rows[0][0], rows[0][1]
    total = 0
    for current_start, current_end, _ in rows[1:]:
        if current_start > end:
            total += end - start
            start, end = current_start, current_end
        else:
            end = max(end, current_end)
    return total + end - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--kernels-per-step", type=int, required=True)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    db = sqlite3.connect(args.sqlite)
    names = dict(db.execute("select id,value from StringIds"))
    kernels = db.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    runtime = db.execute("select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start").fetchall()
    runtime_starts = [row[0] for row in runtime]
    period, count = args.kernels_per_step, args.steps
    first = len(kernels) - period * count
    if first < 0:
        raise ValueError("too few kernels for cached windows")
    signature = [name for _, _, name in kernels[first:first + period]]
    if any([name for _, _, name in kernels[first + i * period:first + (i + 1) * period]] != signature
           for i in range(1, count)):
        raise ValueError("cached windows do not repeat the same kernel sequence")
    steps = []
    totals = defaultdict(lambda: [0, 0])
    for i in range(count):
        rows = kernels[first + i * period:first + (i + 1) * period]
        span = rows[-1][1] - rows[0][0]
        work = sum(end - start for start, end, _ in rows)
        busy = busy_ns(rows)
        left = bisect_left(runtime_starts, rows[0][0])
        right = bisect_left(runtime_starts, rows[-1][1])
        calls = runtime[left:right]
        launch_api = sum(end - start for start, end, name in calls if "LaunchKernel" in names[name])
        sync_api = sum(end - start for start, end, name in calls if "Synchronize" in names[name])
        steps.append({"kernel_count": period, "kernel_ms": work / 1e6,
                      "kernel_span_ms": span / 1e6, "gpu_busy_ms": busy / 1e6,
                      "between_kernel_ms": (span - busy) / 1e6,
                      "launch_api_ms": launch_api / 1e6, "synchronize_api_ms": sync_api / 1e6,
                      "launch_api_calls": sum("LaunchKernel" in names[name] for _, _, name in calls)})
        for start, end, name in rows:
            totals[names[name]][0] += 1
            totals[names[name]][1] += end - start
    top = sorted(totals.items(), key=lambda item: item[1][1], reverse=True)[:25]
    result = {
        "suite": "apertus_cached_cuda_trace", "source_sqlite_sha256": hashlib.sha256(args.sqlite.read_bytes()).hexdigest(),
        "whole_run_kernel_count": len(kernels), "prefill_kernel_count": first,
        "cached_steps": count, "kernels_per_cached_step": period,
        "median_kernel_ms": statistics.median(step["kernel_ms"] for step in steps),
        "median_kernel_span_ms": statistics.median(step["kernel_span_ms"] for step in steps),
        "median_gpu_busy_ms": statistics.median(step["gpu_busy_ms"] for step in steps),
        "median_between_kernel_ms": statistics.median(step["between_kernel_ms"] for step in steps),
        "median_launch_api_ms": statistics.median(step["launch_api_ms"] for step in steps),
        "median_synchronize_api_ms": statistics.median(step["synchronize_api_ms"] for step in steps),
        "top_cached_kernels": [{"name": name, "launches": values[0], "total_ms": values[1] / 1e6}
                               for name, values in top],
        "steps": steps,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("kernels_per_cached_step", "median_kernel_ms",
                                                  "median_kernel_span_ms", "median_between_kernel_ms")}))


if __name__ == "__main__":
    main()
