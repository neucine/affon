#!/usr/bin/env python3
"""Join Compute NVTX invocation ranges to CUDA activities in an Nsight SQLite export."""

import argparse
import heapq
import json
import re
import sqlite3
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

RUN_LABEL = re.compile(r"^affon run=(\d+) executable=(\d+) step=(\d+) invocation=(\d+)$")
EXECUTION_LABEL = re.compile(r"^affon execution run=(\d+) executable=(\d+)$")
READBACK_LABEL = re.compile(
    r"^affon readback tensor=(\d+) storage=(\d+) operation=([A-Za-z0-9_]+)$"
)
CUBLAS_LABEL = re.compile(r"^cublas(?:Lt)?[A-Z][A-Za-z0-9_]*$")
CUBLAS_LIFECYCLE_PREFIXES = (
    "cublasCreate",
    "cublasDestroy",
    "cublasGetProperty",
    "cublasGetVersion",
    "cublasGetMathMode",
    "cublasSetMathMode",
    "cublasSetStream",
    "cublasGetStream",
)


def table_exists(db, name):
    return db.execute("select 1 from sqlite_master where type='table' and name=?", (name,)).fetchone() is not None


def columns(db, name):
    return {row[1] for row in db.execute(f"pragma table_info({name})")}


def union_ns(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = []
    start, end = intervals[0]
    for left, right in intervals[1:]:
        if left > end:
            merged.append((start, end))
            start, end = left, right
        else:
            end = max(end, right)
    merged.append((start, end))
    return merged


def inactive_intervals(intervals):
    merged = merge_intervals(intervals)
    return [
        (right, next_left)
        for (_, right), (next_left, _) in zip(merged, merged[1:])
        if right < next_left
    ]


def cublas_name(name):
    return name if CUBLAS_LABEL.fullmatch(name or "") else None


def cublas_lifecycle(name):
    return any((name or "").startswith(prefix) for prefix in CUBLAS_LIFECYCLE_PREFIXES)


def clipped_intervals(start, end, intervals):
    clipped = []
    for left, right, *_ in intervals:
        if left < end and start < right:
            clipped.append((max(start, left), min(end, right)))
    return clipped


def interval_coverage_by_window(windows, intervals, name_index=0):
    ordered = sorted(intervals, key=lambda interval: (interval[0], interval[1]))
    active = []
    next_index = 0
    sequence = 0
    results = []
    for start, end in windows:
        while next_index < len(ordered) and ordered[next_index][0] < end:
            left, right, *payload = ordered[next_index]
            heapq.heappush(active, (right, sequence, left, payload))
            sequence += 1
            next_index += 1
        while active and active[0][0] <= start:
            heapq.heappop(active)
        clipped = []
        names = defaultdict(lambda: {"coverage_ns": 0, "interval_count": 0})
        for right, _sequence, left, payload in active:
            overlap = min(end, right) - max(start, left)
            if overlap <= 0:
                continue
            clipped.append((max(start, left), min(end, right)))
            name = payload[name_index] if len(payload) > name_index else "<unnamed>"
            names[name]["coverage_ns"] += overlap
            names[name]["interval_count"] += 1
        results.append({
            "coverage_ns": union_ns(clipped),
            "intervals": clipped,
            "names": names,
        })
    return results


def compact_gap_transitions(gaps):
    transitions = {}
    for gap in gaps:
        preceding = sorted(
            {
                json.dumps(point, sort_keys=True): point
                for point in gap["preceding"]
            }.values(),
            key=lambda point: json.dumps(point, sort_keys=True),
        )
        following = sorted(
            {
                json.dumps(point, sort_keys=True): point
                for point in gap["following"]
            }.values(),
            key=lambda point: json.dumps(point, sort_keys=True),
        )
        key = (
            tuple(json.dumps(point, sort_keys=True) for point in preceding),
            tuple(json.dumps(point, sort_keys=True) for point in following),
        )
        bucket = transitions.setdefault(
            key,
            {
                "preceding": preceding,
                "following": following,
                "gap_count": 0,
                "duration_ns": 0,
                "synchronization_ns": 0,
                "cublas_api_ns": 0,
                "cuda_api_ns": 0,
                "host_delay_ns": 0,
            },
        )
        bucket["gap_count"] += 1
        bucket["duration_ns"] += gap["duration_ns"]
        bucket["synchronization_ns"] += gap["synchronization_ns"]
        bucket["cublas_api_ns"] += gap["cublas_api_ns"]
        bucket["cuda_api_ns"] += gap["cuda_api_ns"]
        bucket["host_delay_ns"] += gap["host_delay_ns"]
    return sorted(
        transitions.values(),
        key=lambda item: (
            item["host_delay_ns"],
            item["duration_ns"],
        ),
        reverse=True,
    )


def classify_gaps(entries, activities, largest_limit=10):
    gaps = inactive_intervals([(start, end) for start, end, *_ in activities])
    api_intervals = [
        interval
        for entry in entries
        for interval in entry.get("api_intervals", [])
    ]
    sync_intervals = [
        interval
        for entry in entries
        for interval in entry.get("sync_intervals", [])
    ]
    cublas_intervals = [
        interval
        for entry in entries
        for interval in entry.get("cublas_intervals", [])
    ]
    api_windows = interval_coverage_by_window(gaps, api_intervals)
    sync_windows = interval_coverage_by_window(gaps, sync_intervals, name_index=1)
    cublas_windows = interval_coverage_by_window(gaps, cublas_intervals)
    api_coverage = defaultdict(lambda: {"coverage_ns": 0, "interval_count": 0})
    sync_coverage = defaultdict(lambda: {"coverage_ns": 0, "interval_count": 0})
    cublas_coverage = defaultdict(lambda: {"coverage_ns": 0, "interval_count": 0})
    classification = {
        "gap_count": len(gaps),
        "duration_ns": 0,
        "synchronization_ns": 0,
        "cublas_api_ns": 0,
        "cuda_api_ns": 0,
        "host_delay_ns": 0,
    }
    ends = defaultdict(list)
    starts = defaultdict(list)
    for entry in entries:
        if entry["kind"] == "run":
            point = {
                "step_index": entry["step_index"],
                "invocation_id": entry["invocation_id"],
                "invocation_id_text": entry["invocation_id_text"],
            }
        elif entry["kind"] == "execution":
            point = {
                "run_id": entry["run_id"],
                "run_id_text": entry["run_id_text"],
                "executable_id": entry["executable_id"],
                "executable_id_text": entry["executable_id_text"],
            }
        else:
            point = {
                "tensor_id": entry["tensor_id"],
                "tensor_id_text": entry["tensor_id_text"],
                "storage_id": entry["storage_id"],
                "storage_id_text": entry["storage_id_text"],
                "operation": entry["operation"],
            }
        for activity in entry["activities"]:
            start, end, *_ = activity
            ends[end].append(point)
            starts[start].append(point)

    detailed = []
    for (start, end), api_window, sync_window, cublas_window in zip(
        gaps, api_windows, sync_windows, cublas_windows
    ):
        duration = end - start
        sync_ns = sync_window["coverage_ns"]
        cublas_ns = union_ns(
            clipped_intervals(
                start, end, cublas_window["intervals"] + sync_window["intervals"]
            )
        ) - sync_ns
        recorded_ns = union_ns(
            api_window["intervals"]
            + sync_window["intervals"]
            + cublas_window["intervals"]
        )
        host_ns = duration - recorded_ns
        exclusive_api_ns = recorded_ns - sync_ns - cublas_ns
        for name, values in api_window["names"].items():
            api_coverage[name]["coverage_ns"] += values["coverage_ns"]
            api_coverage[name]["interval_count"] += values["interval_count"]
        for name, values in sync_window["names"].items():
            sync_coverage[name]["coverage_ns"] += values["coverage_ns"]
            sync_coverage[name]["interval_count"] += values["interval_count"]
        for name, values in cublas_window["names"].items():
            cublas_coverage[name]["coverage_ns"] += values["coverage_ns"]
            cublas_coverage[name]["interval_count"] += values["interval_count"]
        classification["duration_ns"] += duration
        classification["synchronization_ns"] += sync_ns
        classification["cublas_api_ns"] += cublas_ns
        classification["cuda_api_ns"] += exclusive_api_ns
        classification["host_delay_ns"] += host_ns
        detailed.append({
            "start_ns": start,
            "end_ns": end,
            "duration_ns": duration,
            "preceding": ends.get(start, []),
            "following": starts.get(end, []),
            "synchronization_ns": sync_ns,
            "cublas_api_ns": cublas_ns,
            "cuda_api_ns": exclusive_api_ns,
            "host_delay_ns": host_ns,
        })

    detailed.sort(key=lambda gap: gap["duration_ns"], reverse=True)
    return {
        "classification": classification,
        "apis": [
            {"name": name, **values}
            for name, values in sorted(
                api_coverage.items(),
                key=lambda item: item[1]["coverage_ns"],
                reverse=True,
            )
        ],
        "synchronization_apis": [
            {"name": name, **values}
            for name, values in sorted(
                sync_coverage.items(),
                key=lambda item: item[1]["coverage_ns"],
                reverse=True,
            )
        ],
        "cublas_apis": [
            {"name": name, **values}
            for name, values in sorted(
                cublas_coverage.items(),
                key=lambda item: item[1]["coverage_ns"],
                reverse=True,
            )
        ],
        "largest_gaps": detailed[:largest_limit],
        "gap_transitions": compact_gap_transitions(detailed),
    }


def analyze(db):
    required = {"NVTX_EVENTS", "StringIds", "CUPTI_ACTIVITY_KIND_KERNEL"}
    missing = sorted(name for name in required if not table_exists(db, name))
    if missing:
        raise ValueError(f"missing Nsight tables: {', '.join(missing)}")
    nvtx_columns = columns(db, "NVTX_EVENTS")
    if not {"start", "end", "text", "textId", "globalTid"} <= nvtx_columns:
        raise ValueError("NVTX_EVENTS lacks range text, time, or thread columns")
    ranges = []
    for start, end, text, tid in db.execute("""
        select n.start, n.end, coalesce(n.text, s.value), n.globalTid
        from NVTX_EVENTS n left join StringIds s on n.textId=s.id
        where n.end is not null order by n.start
    """):
        text = text or ""
        match = RUN_LABEL.fullmatch(text)
        if match:
            ranges.append({
                "kind": "run",
                "start": start,
                "end": end,
                "tid": tid,
                "run_id": int(match[1]),
                "run_id_text": match[1],
                "executable_id": int(match[2]),
                "executable_id_text": match[2],
                "step_index": int(match[3]),
                "invocation_id": int(match[4]),
                "invocation_id_text": match[4],
                "api_ns": 0,
                "api_count": 0,
                "api_intervals": [],
                "sync_intervals": [],
                "cublas_ns": 0,
                "cublas_count": 0,
                "cublas_intervals": [],
                "activities": [],
                "library_activities": [],
                "custom_activities": [],
            })
            continue
        match = EXECUTION_LABEL.fullmatch(text)
        if match:
            ranges.append({
                "kind": "execution",
                "start": start,
                "end": end,
                "tid": tid,
                "run_id": int(match[1]),
                "run_id_text": match[1],
                "executable_id": int(match[2]),
                "executable_id_text": match[2],
                "api_ns": 0,
                "api_count": 0,
                "api_intervals": [],
                "sync_intervals": [],
                "cublas_ns": 0,
                "cublas_count": 0,
                "cublas_intervals": [],
                "activities": [],
                "library_activities": [],
                "custom_activities": [],
            })
            continue
        match = READBACK_LABEL.fullmatch(text)
        if match:
            ranges.append({
                "kind": "readback",
                "start": start,
                "end": end,
                "tid": tid,
                "tensor_id": int(match[1]),
                "tensor_id_text": match[1],
                "storage_id": int(match[2]),
                "storage_id_text": match[2],
                "operation": match[3],
                "api_ns": 0,
                "api_count": 0,
                "api_intervals": [],
                "sync_intervals": [],
                "cublas_ns": 0,
                "cublas_count": 0,
                "cublas_intervals": [],
                "activities": [],
                "library_activities": [],
                "custom_activities": [],
            })
    if not ranges:
        raise ValueError("no Affon NVTX invocation or readback ranges found")
    ranges_by_tid = defaultdict(list)
    for entry in ranges:
        ranges_by_tid[entry["tid"]].append(entry)
    starts_by_tid = {tid: [entry["start"] for entry in entries] for tid, entries in ranges_by_tid.items()}
    total_cublas_activities = 0
    total_cublas_lifecycle_activities = 0
    unmatched_cublas_activities = 0
    unmatched_cublas_details = []
    lifecycle_cublas_details = []
    for start, end, name, tid in db.execute("""
        select n.start, n.end, coalesce(n.text, s.value), n.globalTid
        from NVTX_EVENTS n left join StringIds s on n.textId=s.id
        where n.end is not null and lower(coalesce(n.text, s.value)) like 'cublas%'
        order by n.start
    """):
        name = cublas_name(name)
        if name is None:
            continue
        total_cublas_activities += 1
        if cublas_lifecycle(name):
            total_cublas_lifecycle_activities += 1
            if len(lifecycle_cublas_details) < 20:
                lifecycle_cublas_details.append({
                    "start_ns": start,
                    "end_ns": end,
                    "api": name,
                })
            continue
        thread_ranges = ranges_by_tid.get(tid)
        index = (
            bisect_right(starts_by_tid.get(tid, []), start) - 1
            if thread_ranges
            else -1
        )
        entry = thread_ranges[index] if index >= 0 else None
        if entry is None or end > entry["end"]:
            unmatched_cublas_activities += 1
            if len(unmatched_cublas_details) < 20:
                unmatched_cublas_details.append({
                    "start_ns": start,
                    "end_ns": end,
                    "api": name,
                })
            continue
        entry["cublas_ns"] += end - start
        entry["cublas_count"] += 1
        entry["cublas_intervals"].append((start, end, name))
    api_by_correlation = {}
    api_name_by_correlation = {}
    api_thread_by_correlation = {}
    api_interval_by_correlation = {}
    library_api_correlations = set()
    conflicting_correlations = set()
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if not table_exists(db, table):
            continue
        if not {"start", "end", "globalTid", "correlationId"} <= columns(db, table):
            raise ValueError(f"{table} lacks required correlation columns")
        table_columns = columns(db, table)
        name_column = "coalesce(s.value, '')" if "nameId" in table_columns else "''"
        join = "left join StringIds s on a.nameId=s.id" if "nameId" in table_columns else ""
        for start, end, tid, correlation, name in db.execute(f"""
            select a.start,a.end,a.globalTid,a.correlationId,{name_column}
            from {table} a {join}
            where a.correlationId is not null
        """):
            if name and correlation not in api_name_by_correlation:
                api_name_by_correlation[correlation] = name
            api_thread_by_correlation[correlation] = tid
            api_interval_by_correlation[correlation] = (start, end)
            thread_ranges = ranges_by_tid.get(tid)
            if not thread_ranges:
                continue
            index = bisect_right(starts_by_tid[tid], start) - 1
            if index < 0:
                continue
            entry = thread_ranges[index]
            if end > entry["end"]:
                continue
            existing = api_by_correlation.get(correlation)
            if existing is not None and existing is not entry:
                conflicting_correlations.add(correlation)
                continue
            if existing is None:
                api_by_correlation[correlation] = entry
                entry["api_ns"] += end - start
                entry["api_count"] += 1
                entry["api_intervals"].append((start, end, name or "<unnamed>"))
                if any(
                    left <= start and end <= right
                    for left, right, _ in entry["cublas_intervals"]
                ):
                    library_api_correlations.add(correlation)
    for correlation in conflicting_correlations:
        api_by_correlation.pop(correlation, None)
    total_synchronizations = 0
    unmatched_synchronizations = 0
    outside_synchronizations = 0
    unmatched_synchronization_details = []
    outside_synchronization_details = []
    outside_synchronization_apis = defaultdict(int)
    if table_exists(db, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"):
        if not {"start", "end", "correlationId", "syncType"} <= columns(
            db, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"
        ):
            raise ValueError(
                "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION lacks correlation or type columns"
            )
        for start, end, correlation, sync_type in db.execute("""
            select start,end,correlationId,syncType
            from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
        """):
            total_synchronizations += 1
            entry = api_by_correlation.get(correlation)
            if entry is None:
                api_interval = api_interval_by_correlation.get(correlation)
                api_tid = api_thread_by_correlation.get(correlation)
                if api_interval is not None and api_tid is not None:
                    api_start, api_end = api_interval
                    thread_ranges = ranges_by_tid.get(api_tid, [])
                    containing = [
                        candidate
                        for candidate in thread_ranges
                        if candidate["start"] <= api_start
                        and api_end <= candidate["end"]
                    ]
                    if containing:
                        entry = max(
                            containing,
                            key=lambda candidate: candidate["start"],
                        )
            if entry is None:
                api_tid = api_thread_by_correlation.get(correlation)
                thread_ranges = (
                    ranges_by_tid.get(api_tid, [])
                    if api_tid is not None
                    else [
                        candidate
                        for entries in ranges_by_tid.values()
                        for candidate in entries
                    ]
                )
                inside_affon_range = any(
                    candidate["start"] <= start and end <= candidate["end"]
                    for candidate in thread_ranges
                    if candidate["start"] <= start
                )
                if not inside_affon_range:
                    outside_synchronizations += 1
                    api_name = api_name_by_correlation.get(
                        correlation, "<unnamed>"
                    )
                    outside_synchronization_apis[api_name] += 1
                    if len(outside_synchronization_details) < 20:
                        outside_synchronization_details.append({
                            "start_ns": start,
                            "end_ns": end,
                            "correlation_id": correlation,
                            "api": api_name,
                            "sync_type": sync_type,
                        })
                    continue
                unmatched_synchronizations += 1
                if len(unmatched_synchronization_details) < 20:
                    unmatched_synchronization_details.append({
                        "start_ns": start,
                        "end_ns": end,
                        "correlation_id": correlation,
                        "api": api_name_by_correlation.get(correlation, "<unnamed>"),
                        "sync_type": sync_type,
                    })
                continue
            name = next(
                (
                    api_name
                    for api_start, api_end, api_name in entry["api_intervals"]
                    if api_start <= start and end <= api_end
                ),
                "<unnamed>",
            )
            entry["sync_intervals"].append((start, end, sync_type, name))
    total_activities = 0
    unmatched_activities = 0
    unmatched_activity_details = []
    for table, kind in (
        ("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "memset"),
    ):
        if not table_exists(db, table):
            continue
        if not {"start", "end", "correlationId"} <= columns(db, table):
            raise ValueError(f"{table} lacks required activity columns")
        has_bytes = "bytes" in columns(db, table)
        query = f"select start,end,correlationId,{'bytes' if has_bytes else '0'} from {table}"
        for start, end, correlation, byte_count in db.execute(query):
            total_activities += 1
            entry = api_by_correlation.get(correlation)
            if entry is None:
                unmatched_activities += 1
                if len(unmatched_activity_details) < 20:
                    unmatched_activity_details.append({
                        "start_ns": start,
                        "end_ns": end,
                        "kind": kind,
                        "correlation_id": correlation,
                        "bytes": byte_count or 0,
                    })
            else:
                activity = (start, end, kind, byte_count or 0)
                entry["activities"].append(activity)
                if correlation in library_api_correlations:
                    entry["library_activities"].append(activity)
                else:
                    entry["custom_activities"].append(activity)
    by_run = defaultdict(list)
    for entry in ranges:
        if entry["kind"] in ("run", "execution"):
            by_run[(entry["run_id_text"], entry["executable_id_text"])].append(entry)
    runs = []
    for (run_id_text, executable_id_text), entries in sorted(by_run.items()):
        activities = [activity for entry in entries for activity in entry["activities"]]
        intervals = [(start, end) for start, end, _, _ in activities]
        span_ns = max((end for _, end in intervals), default=0) - min((start for start, _ in intervals), default=0) if intervals else 0
        gap_status = classify_gaps(entries, activities)
        steps = defaultdict(list)
        for entry in entries:
            if entry["kind"] == "run":
                steps[entry["step_index"]].append(entry)
        runs.append({"run_id": int(run_id_text), "run_id_text": run_id_text,
                     "executable_id": int(executable_id_text), "executable_id_text": executable_id_text,
                     "execution_range_count": sum(entry["kind"] == "execution" for entry in entries),
                     "invocation_count": sum(entry["kind"] == "run" for entry in entries),
                     "cuda_api_count": sum(entry["api_count"] for entry in entries),
                     "invocation_ids": sorted(entry["invocation_id"] for entry in entries if entry["kind"] == "run"),
                     "invocation_ids_text": sorted(entry["invocation_id_text"] for entry in entries if entry["kind"] == "run"),
                     "cuda_api_ns": sum(entry["api_ns"] for entry in entries),
                     "cublas_api_count": sum(entry["cublas_count"] for entry in entries),
                     "cublas_api_ns": sum(entry["cublas_ns"] for entry in entries),
                     "kernel_count": sum(kind == "kernel" for _, _, kind, _ in activities),
                     "library_kernel_count": sum(
                         kind == "kernel"
                         for entry in entries
                         for _, _, kind, _ in entry["library_activities"]
                     ),
                     "custom_kernel_count": sum(
                         kind == "kernel"
                         for entry in entries
                         for _, _, kind, _ in entry["custom_activities"]
                     ),
                     "memcpy_count": sum(kind == "memcpy" for _, _, kind, _ in activities),
                     "memcpy_bytes": sum(byte_count for _, _, kind, byte_count in activities if kind == "memcpy"),
                     "memset_count": sum(kind == "memset" for _, _, kind, _ in activities),
                     "memset_bytes": sum(byte_count for _, _, kind, byte_count in activities if kind == "memset"),
                     "device_active_ns": union_ns(intervals), "device_span_ns": span_ns,
                     "device_gap_ns": span_ns - union_ns(intervals),
                     **gap_status,
                     "steps": [{"step_index": step, "invocation_count": len(items),
                                "invocation_ids": sorted(item["invocation_id"] for item in items),
                                "invocation_ids_text": sorted(item["invocation_id_text"] for item in items),
                                "cuda_api_ns": sum(item["api_ns"] for item in items),
                                "cublas_api_count": sum(item["cublas_count"] for item in items),
                                "cublas_api_ns": sum(item["cublas_ns"] for item in items),
                                "kernel_count": sum(kind == "kernel" for item in items for _, _, kind, _ in item["activities"]),
                                "library_kernel_count": sum(
                                    kind == "kernel"
                                    for item in items
                                    for _, _, kind, _ in item["library_activities"]
                                ),
                                "custom_kernel_count": sum(
                                    kind == "kernel"
                                    for item in items
                                    for _, _, kind, _ in item["custom_activities"]
                                ),
                                "memcpy_count": sum(kind == "memcpy" for item in items for _, _, kind, _ in item["activities"]),
                                "memcpy_bytes": sum(byte_count for item in items for _, _, kind, byte_count in item["activities"] if kind == "memcpy"),
                                "memset_count": sum(kind == "memset" for item in items for _, _, kind, _ in item["activities"]),
                                "memset_bytes": sum(byte_count for item in items for _, _, kind, byte_count in item["activities"] if kind == "memset"),
                                "device_active_ns": union_ns([(start, end) for item in items for start, end, _, _ in item["activities"]])}
                               for step, items in sorted(steps.items())]})
    readback_groups = defaultdict(list)
    for entry in ranges:
        if entry["kind"] == "readback":
            readback_groups[(entry["tensor_id"], entry["storage_id"], entry["operation"])].append(entry)
    readbacks = []
    for (tensor_id, storage_id, operation), entries in sorted(readback_groups.items()):
        activities = [activity for entry in entries for activity in entry["activities"]]
        intervals = [(start, end) for start, end, _, _ in activities]
        span_ns = max((end for _, end in intervals), default=0) - min((start for start, _ in intervals), default=0) if intervals else 0
        gap_status = classify_gaps(entries, activities)
        readbacks.append({
            "tensor_id": tensor_id,
            "tensor_id_text": str(tensor_id),
            "storage_id": storage_id,
            "storage_id_text": str(storage_id),
            "operation": operation,
            "range_count": len(entries),
            "cuda_api_count": sum(entry["api_count"] for entry in entries),
            "cuda_api_ns": sum(entry["api_ns"] for entry in entries),
            "cublas_api_count": sum(entry["cublas_count"] for entry in entries),
            "cublas_api_ns": sum(entry["cublas_ns"] for entry in entries),
            "kernel_count": sum(kind == "kernel" for _, _, kind, _ in activities),
            "memcpy_count": sum(kind == "memcpy" for _, _, kind, _ in activities),
            "memcpy_bytes": sum(byte_count for _, _, kind, byte_count in activities if kind == "memcpy"),
            "memset_count": sum(kind == "memset" for _, _, kind, _ in activities),
            "memset_bytes": sum(byte_count for _, _, kind, byte_count in activities if kind == "memset"),
            "device_active_ns": union_ns(intervals),
            "device_span_ns": span_ns,
            "device_gap_ns": span_ns - union_ns(intervals),
            **gap_status,
        })
    return {"schema_version": 3, "source": "nsight_sqlite_nvtx_cupti", "runs": runs, "readbacks": readbacks,
            "total_device_activities": total_activities, "unmatched_device_activities": unmatched_activities,
            "unmatched_device_activity_details": unmatched_activity_details,
            "total_synchronization_activities": total_synchronizations,
            "unmatched_synchronization_activities": unmatched_synchronizations,
            "unmatched_synchronizations": unmatched_synchronization_details,
            "outside_synchronization_activities": outside_synchronizations,
            "outside_synchronizations": outside_synchronization_details,
            "outside_synchronization_apis": [
                {"name": name, "activity_count": count}
                for name, count in sorted(
                    outside_synchronization_apis.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ],
            "total_cublas_activities": total_cublas_activities,
            "total_cublas_lifecycle_activities": total_cublas_lifecycle_activities,
            "lifecycle_cublas": lifecycle_cublas_details,
            "unmatched_cublas_activities": unmatched_cublas_activities,
            "unmatched_cublas": unmatched_cublas_details,
            "conflicting_correlation_ids": len(conflicting_correlations),
            "gap_definition": {
                "synchronization_ns": "Gap time covered by a correlated CUPTI synchronization activity.",
                "cublas_api_ns": "Remaining gap time covered by a cublas-verbose NVTX API range outside synchronization.",
                "cuda_api_ns": "Remaining gap time covered by another correlated CUDA runtime or driver API call outside synchronization and cuBLAS API ranges.",
                "host_delay_ns": "Gap time with no recorded synchronization, cuBLAS API, or CUDA API coverage.",
            },
            "limitations": "Gap classification uses CUDA runtime and synchronization events plus cublas-verbose API ranges nested in the same Affon NVTX range. Categorized host work can overlap GPU work. Host delay means no recorded API coverage and does not by itself prove dependency starvation. Activities outside Affon invocation and readback NVTX ranges remain unmatched."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with sqlite3.connect(args.sqlite) as db:
        result = analyze(db)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"runs": len(result["runs"]), "unmatched_device_activities": result["unmatched_device_activities"]}))


if __name__ == "__main__":
    main()
