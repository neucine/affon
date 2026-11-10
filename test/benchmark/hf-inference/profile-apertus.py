#!/usr/bin/env python3
"""Capture one warmed Apertus prefill or decode step as one joined artifact."""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORKER = Path(__file__).with_name("apertus-profile.ts")
CORRELATOR = ROOT / "tools" / "correlate-nsys-compute.py"
BASE_REQUIRED_SPANS = {
    "affon.executable.run",
    "affon.bridge.run",
    "affon.bridge.readback",
    "compute.execution",
    "inference.generation.inputs",
    "inference.generation.input_cleanup",
    "inference.generation.selection",
    "inference.generation.cache_release",
    "inference.generation.cleanup",
}


def selected_span_name(workload):
    if workload not in ("prefill", "decode"):
        raise ValueError("workload must be prefill or decode")
    return f"apertus.cached_{workload}.selected"


def required_spans(workload):
    return {selected_span_name(workload), *BASE_REQUIRED_SPANS}


def find_qdstrm_importer(nsys):
    nsys_path = Path(nsys).resolve()
    candidates = [
        nsys_path.parent.parent / "host-linux-x64" / "QdstrmImporter",
        Path("/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter"),
        Path("/usr/lib/nsight-compute/host/linux-desktop-glibc_2_11_3-x64/QdstrmImporter"),
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    discovered = shutil.which("QdstrmImporter")
    return Path(discovered) if discovered else None


def selected_capture_status(
    records, require_profiler_markers, workload="decode"
):
    selected_name = selected_span_name(workload)
    selected = [
        record
        for record in records
        if record.get("kind") == "span_start"
        and record.get("name") == selected_name
    ]
    trace_ids = {
        record["trace_id"] for record in selected if record.get("trace_id")
    }
    selected_records = [
        record for record in records if record.get("trace_id") in trace_ids
    ]
    started = {
        record["name"]
        for record in selected_records
        if record.get("kind") == "span_start"
    }
    marker_available = any(
        record.get("kind") == "event"
        and record.get("name") == "compute.profiler_marker_availability"
        and record.get("attributes", {}).get("compute.available") is True
        for record in selected_records
    )
    return {
        "selected_span_count": len(selected),
        "selected_trace_ids": sorted(trace_ids),
        "missing_spans": sorted(required_spans(workload) - started),
        "profiler_marker_available": marker_available,
        "profiler_marker_required": require_profiler_markers,
    }


def joined_identity_status(
    report, records, timeline=None, workload="decode"
):
    selected_name = selected_span_name(workload)
    def exact_id_text(attributes, key):
        text_key = f"{key}_text"
        value = attributes.get(text_key, attributes.get(key))
        return None if value is None else str(value)

    def exact_id(attributes, key):
        return int(exact_id_text(attributes, key))

    selected = [
        record
        for record in records
        if record.get("kind") == "span_start"
        and record.get("name") == selected_name
    ]
    trace_ids = {
        record["trace_id"] for record in selected if record.get("trace_id")
    }
    selected_records = [
        record for record in records if record.get("trace_id") in trace_ids
    ]
    executions = [
        record
        for record in selected_records
        if record.get("kind") == "span_start"
        and record.get("name") == "compute.execution"
    ]
    run_id_texts = sorted(
        {
            exact_id_text(record["attributes"], "compute.run_id")
            for record in executions
            if "compute.run_id" in record.get("attributes", {})
            or "compute.run_id_text" in record.get("attributes", {})
        }
    )
    executable_id_texts = sorted(
        {
            exact_id_text(record["attributes"], "compute.executable_id")
            for record in executions
            if "compute.executable_id" in record.get("attributes", {})
            or "compute.executable_id_text" in record.get("attributes", {})
        }
    )

    telemetry_steps = {}
    malformed_invocations = 0
    for record in selected_records:
        if record.get("kind") != "event" or record.get("name") != "compute.invocation":
            continue
        attributes = record.get("attributes", {})
        if (
            "compute.step_index" not in attributes
            or (
                "compute.invocation_id" not in attributes
                and "compute.invocation_id_text" not in attributes
            )
        ):
            malformed_invocations += 1
            continue
        step = int(attributes["compute.step_index"])
        invocation = exact_id(attributes, "compute.invocation_id")
        telemetry_steps.setdefault(step, set()).add(invocation)
    telemetry_steps = {
        step: sorted(invocations)
        for step, invocations in sorted(telemetry_steps.items())
    }
    telemetry_invocation_count = sum(len(items) for items in telemetry_steps.values())

    evidence = report.get("runtime_evidence", {})
    evidence_executable_id_text = str(
        evidence.get("executable_id_text", evidence.get("executable_id"))
    )
    evidence_steps = {
        int(entry["step_index"]): int(entry["invocation_count"])
        for entry in evidence.get("entries", [])
    }
    telemetry_counts = {
        step: len(invocations) for step, invocations in telemetry_steps.items()
    }
    evidence_count_mismatches = sum(
        count != evidence_steps.get(step, 0)
        for step, count in telemetry_counts.items()
    ) + sum(
        count != telemetry_counts.get(step, 0)
        for step, count in evidence_steps.items()
    )
    evidence_invocation_count = sum(evidence_steps.values())

    problems = []
    if len(executions) != 1:
        problems.append(
            f"selected {workload} trace does not have exactly one Compute execution"
        )
    if len(run_id_texts) != 1:
        problems.append(
            f"selected {workload} trace does not have exactly one run ID"
        )
    if len(executable_id_texts) != 1:
        problems.append(
            f"selected {workload} trace does not have exactly one executable ID"
        )
    if malformed_invocations:
        problems.append(
            f"selected {workload} trace has malformed invocation telemetry"
        )
    if evidence_count_mismatches:
        problems.append(
            "runtime evidence invocation counts do not match selected telemetry"
        )
    if executable_id_texts and evidence_executable_id_text != executable_id_texts[0]:
        problems.append("runtime evidence and telemetry executable IDs differ")

    nsys_invocation_count = None
    nsys_steps = None
    if timeline is not None:
        matching_runs = [
            run
            for run in timeline.get("runs", [])
            if str(run.get("run_id_text", run.get("run_id"))) in run_id_texts
            and str(run.get("executable_id_text", run.get("executable_id")))
            in executable_id_texts
        ]
        if len(matching_runs) != 1:
            problems.append("Nsight timeline does not have exactly one matching Affon run")
        else:
            nsys_steps = {
                int(step["step_index"]): sorted(
                    str(invocation)
                    for invocation in step.get(
                        "invocation_ids_text", step.get("invocation_ids", [])
                    )
                )
                for step in matching_runs[0].get("steps", [])
            }
            nsys_invocation_count = sum(
                len(invocations) for invocations in nsys_steps.values()
            )
            if nsys_steps != {
                step: [str(invocation) for invocation in invocations]
                for step, invocations in telemetry_steps.items()
            }:
                problems.append(
                    "Nsight invocation IDs or step assignments differ from telemetry"
                )

    summary = {
        "run_id": int(run_id_texts[0]) if len(run_id_texts) == 1 else None,
        "run_id_text": run_id_texts[0] if len(run_id_texts) == 1 else None,
        "executable_id": (
            int(executable_id_texts[0]) if len(executable_id_texts) == 1 else None
        ),
        "executable_id_text": (
            executable_id_texts[0] if len(executable_id_texts) == 1 else None
        ),
        "step_count": len(telemetry_steps),
        "telemetry_invocation_count": telemetry_invocation_count,
        "runtime_evidence_invocation_count": evidence_invocation_count,
        "runtime_evidence_count_mismatches": evidence_count_mismatches,
        "nsys_invocation_count": nsys_invocation_count,
    }
    return summary, problems


def gap_attribution_status(selected_run, runtime_report):
    entries = {
        int(entry["step_index"]): entry
        for entry in runtime_report.get("runtime_evidence", {}).get("entries", [])
        if "step_index" in entry
    }

    def point_steps(points):
        return sorted(
            {
                int(point["step_index"])
                for point in points
                if "step_index" in point
            }
        )

    def values_for_steps(steps, key, fallback):
        values = []
        for step in steps:
            entry = entries.get(step, {})
            value = entry.get(key)
            if value not in (None, "") and value not in values:
                values.append(value)
        return values or [fallback]

    grouped = {}
    for transition in selected_run.get("gap_transitions", []):
        preceding_steps = point_steps(transition.get("preceding", []))
        following_steps = point_steps(transition.get("following", []))
        preceding_operations = values_for_steps(
            preceding_steps, "operation", "<unattributed>"
        )
        following_operations = values_for_steps(
            following_steps, "operation", "<unattributed>"
        )
        preceding_implementations = values_for_steps(
            preceding_steps, "physical_implementation", "<unattributed>"
        )
        following_implementations = values_for_steps(
            following_steps, "physical_implementation", "<unattributed>"
        )
        key = (
            tuple(preceding_operations),
            tuple(following_operations),
            tuple(preceding_implementations),
            tuple(following_implementations),
        )
        bucket = grouped.setdefault(
            key,
            {
                "preceding_operations": preceding_operations,
                "following_operations": following_operations,
                "preceding_implementations": preceding_implementations,
                "following_implementations": following_implementations,
                "gap_count": 0,
                "duration_ns": 0,
                "synchronization_ns": 0,
                "cublas_api_ns": 0,
                "cuda_api_ns": 0,
                "host_delay_ns": 0,
                "_preceding_steps": set(),
                "_following_steps": set(),
            },
        )
        bucket["_preceding_steps"].update(preceding_steps)
        bucket["_following_steps"].update(following_steps)
        for field in (
            "gap_count",
            "duration_ns",
            "synchronization_ns",
            "cublas_api_ns",
            "cuda_api_ns",
            "host_delay_ns",
        ):
            bucket[field] += transition.get(field, 0)

    operation_transitions = []
    for bucket in grouped.values():
        preceding_steps = sorted(bucket.pop("_preceding_steps"))
        following_steps = sorted(bucket.pop("_following_steps"))
        bucket["preceding_step_indices"] = preceding_steps
        bucket["following_step_indices"] = following_steps
        bucket["preceding_submission_ns"] = sum(
            int(entries.get(step, {}).get("invocation_host_total_ns", 0))
            for step in preceding_steps
        )
        bucket["preceding_invocation_ns"] = sum(
            int(entries.get(step, {}).get("total_ns", 0))
            for step in preceding_steps
        )
        operation_transitions.append(bucket)
    operation_transitions.sort(
        key=lambda item: (item["host_delay_ns"], item["duration_ns"]),
        reverse=True,
    )

    by_preceding_operation = {}
    by_following_operation = {}
    for transition in operation_transitions:
        for key, field, target in (
            (
                " + ".join(transition["preceding_operations"]),
                "preceding_operations",
                by_preceding_operation,
            ),
            (
                " + ".join(transition["following_operations"]),
                "following_operations",
                by_following_operation,
            ),
        ):
            bucket = target.setdefault(
                key,
                {
                    field: transition[field],
                    "gap_count": 0,
                    "duration_ns": 0,
                    "synchronization_ns": 0,
                    "cublas_api_ns": 0,
                    "cuda_api_ns": 0,
                    "host_delay_ns": 0,
                },
            )
            for value in (
                "gap_count",
                "duration_ns",
                "synchronization_ns",
                "cublas_api_ns",
                "cuda_api_ns",
                "host_delay_ns",
            ):
                bucket[value] += transition[value]

    submission_by_operation = {}
    for entry in entries.values():
        elapsed = int(entry.get("invocation_host_total_ns", 0))
        if elapsed == 0:
            continue
        operation = entry.get("operation") or "<view>"
        bucket = submission_by_operation.setdefault(
            operation,
            {
                "operation": operation,
                "invocation_count": 0,
                "submission_ns": 0,
                "library_call_count": 0,
                "custom_dispatch_count": 0,
                "dispatch_count_sample_count": 0,
            },
        )
        bucket["invocation_count"] += int(entry.get("invocation_host_timing_sample_count", 0))
        bucket["submission_ns"] += elapsed
        bucket["library_call_count"] += int(entry.get("library_call_count", 0))
        bucket["custom_dispatch_count"] += int(
            entry.get("custom_dispatch_count", 0)
        )
        bucket["dispatch_count_sample_count"] += int(
            entry.get("dispatch_count_sample_count", 0)
        )

    classification = selected_run.get("classification") or {}
    classified_host_ns = int(classification.get("host_delay_ns", 0))
    attributed_host_ns = sum(
        item["host_delay_ns"] for item in operation_transitions
    )
    problems = []
    if classified_host_ns != attributed_host_ns:
        problems.append(
            "operation-attributed host delay does not cover the selected run's host delay"
        )
    return {
        "source": "selected_run_gap_transitions_joined_to_runtime_operations",
        "classified_host_delay_ns": classified_host_ns,
        "attributed_host_delay_ns": attributed_host_ns,
        "operation_transitions": operation_transitions,
        "host_delay_by_preceding_operation": sorted(
            by_preceding_operation.values(),
            key=lambda item: item["host_delay_ns"],
            reverse=True,
        ),
        "host_delay_by_following_operation": sorted(
            by_following_operation.values(),
            key=lambda item: item["host_delay_ns"],
            reverse=True,
        ),
        "submission_cost_by_operation": sorted(
            submission_by_operation.values(),
            key=lambda item: item["submission_ns"],
            reverse=True,
        ),
    }, problems


def trace_phase_status(records, workload="decode"):
    selected_name = selected_span_name(workload)
    selected = [
        record
        for record in records
        if record.get("kind") == "span_start"
        and record.get("name") == selected_name
    ]
    if len(selected) != 1:
        return {
            "wall_ns": None,
            "covered_ns": None,
            "unattributed_ns": None,
            "coverage_ratio": None,
            "phases": [],
        }, [f"expected exactly one selected {workload} trace"]

    root = selected[0]
    trace_id = root.get("trace_id")
    span_id = root.get("span_id")
    ends = [
        record
        for record in records
        if record.get("kind") == "span_end"
        and record.get("trace_id") == trace_id
        and record.get("span_id") == span_id
    ]
    if len(ends) != 1:
        return {
            "wall_ns": None,
            "covered_ns": None,
            "unattributed_ns": None,
            "coverage_ratio": None,
            "phases": [],
        }, [
            f"selected {workload} trace does not have exactly one end record"
        ]

    root_start = int(root["timestamp_ns"])
    root_end = int(ends[0]["timestamp_ns"])
    problems = []
    intervals = []
    phases = {}
    for child in records:
        if (
            child.get("kind") != "span_start"
            or child.get("trace_id") != trace_id
            or child.get("parent_span_id") != span_id
        ):
            continue
        child_ends = [
            record
            for record in records
            if record.get("kind") == "span_end"
            and record.get("trace_id") == trace_id
            and record.get("span_id") == child.get("span_id")
        ]
        if len(child_ends) != 1:
            problems.append(f"{child.get('name', 'unknown')} has no unique end")
            continue
        start = int(child["timestamp_ns"])
        end = int(child_ends[0]["timestamp_ns"])
        if start < root_start or end > root_end or end < start:
            problems.append(f"{child.get('name', 'unknown')} falls outside selected trace")
            continue
        intervals.append((start, end))
        name = child.get("name", "unknown")
        phases[name] = phases.get(name, 0) + end - start

    wall_ns = root_end - root_start
    covered_ns = union_ns(intervals)
    unattributed_ns = max(0, wall_ns - covered_ns)
    return {
        "wall_ns": wall_ns,
        "covered_ns": covered_ns,
        "unattributed_ns": unattributed_ns,
        "coverage_ratio": covered_ns / wall_ns if wall_ns else None,
        "phases": [
            {"name": name, "elapsed_ns": elapsed}
            for name, elapsed in sorted(phases.items())
        ],
    }, problems


def union_ns(intervals):
    if not intervals:
        return 0
    intervals = sorted(intervals)
    start, end = intervals[0]
    total = 0
    for left, right in intervals[1:]:
        if left > end:
            total += end - start
            start, end = left, right
        else:
            end = max(end, right)
    return total + end - start


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def collect_telemetry(proc, port, report_path, log_path, timeout):
    records, cursor, missed = [], 0, False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/traces?since={cursor}", timeout=2
            ) as response:
                batch = json.load(response)
            records.extend(batch["records"])
            cursor = batch["next_cursor"]
            missed |= batch["missed"]
        except OSError:
            if proc.poll() is not None:
                break
        if proc.poll() is not None:
            # Drain records written while the worker was shutting down.
            for _ in range(3):
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/api/traces?since={cursor}",
                        timeout=2,
                    ) as response:
                        batch = json.load(response)
                except OSError:
                    break
                if not batch["records"]:
                    break
                records.extend(batch["records"])
                cursor = batch["next_cursor"]
                missed |= batch["missed"]
            break
        time.sleep(0.001)
    else:
        proc.terminate()
        proc.wait(timeout=10)
        raise RuntimeError(f"profile timed out; inspect {log_path}")
    return records, missed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affon", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda", "metal", "cpu"))
    parser.add_argument(
        "--workload", default="decode", choices=("prefill", "decode")
    )
    parser.add_argument(
        "--prompt",
        default="1,1784,8961,1307,32265,1395",
        help="comma-separated token IDs used for the fixed-cache profile",
    )
    parser.add_argument("--capacity", default=320, type=int)
    parser.add_argument("--budget", default=4, type=int)
    parser.add_argument("--capture-records", default=262144, type=int)
    parser.add_argument("--telemetry-records", default=131072, type=int)
    parser.add_argument("--nsys", type=Path)
    parser.add_argument("--no-nsys", action="store_true")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    try:
        prompt = [int(token) for token in args.prompt.split(",")]
    except ValueError as error:
        raise RuntimeError("--prompt must contain comma-separated integers") from error
    if not prompt or any(token < 0 for token in prompt):
        raise RuntimeError("--prompt must contain nonnegative token IDs")
    if args.capacity < 2:
        raise RuntimeError("--capacity must be at least 2")
    minimum_budget = 3 if args.workload == "decode" else 1
    if args.budget < minimum_budget:
        raise RuntimeError(
            "--budget must include prefill, one warmup, and one selected token"
            if args.workload == "decode"
            else "--budget must include one selected prefill token"
        )
    if len(prompt) + args.budget > args.capacity:
        raise RuntimeError("prompt and budget must fit the fixed cache capacity")
    if args.capture_records < 1 or args.capture_records > 4194304:
        raise RuntimeError("--capture-records must be between 1 and 4194304")
    if args.telemetry_records < 512 or args.telemetry_records > 1048576:
        raise RuntimeError("--telemetry-records must be between 512 and 1048576")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "apertus-profile.json"
    traces_path = output / "telemetry-traces.json"
    log_path = output / "run.log"
    report_path.unlink(missing_ok=True)
    traces_path.unlink(missing_ok=True)

    nsys = None
    if not args.no_nsys:
        nsys = args.nsys or (Path(shutil.which("nsys")) if shutil.which("nsys") else None)
        if nsys is None:
            raise RuntimeError("nsys was not found; pass --nsys or --no-nsys")

    port = args.port or free_port()
    env = {
        **os.environ,
        "AFFON_APERTUS_CHECKPOINT": str(args.checkpoint.resolve()),
        "AFFON_APERTUS_PROFILE_PATH": str(report_path),
        "AFFON_PROFILE_DEVICE": args.device,
        "AFFON_PROFILE_WORKLOAD": args.workload,
        "AFFON_PROFILE_PROMPT": ",".join(str(token) for token in prompt),
        "AFFON_PROFILE_CAPACITY": str(args.capacity),
        "AFFON_PROFILE_BUDGET": str(args.budget),
        "AFFON_PROFILE_CAPTURE_RECORDS": str(args.capture_records),
        "AFFON_PROFILE_NSYS": "1" if nsys else "0",
        "RUNTIME_TELEMETRY_CONSOLE": "1",
        "RUNTIME_TELEMETRY_CONSOLE_PORT": str(port),
        "RUNTIME_TELEMETRY_TRACE_RECORDS": str(args.telemetry_records),
    }
    command = [str(args.affon.resolve()), str(WORKER)]
    if nsys:
        profile_base = output / "apertus-profile"
        command = [
            str(nsys),
            "profile",
            "--force-overwrite=true",
            "--trace=cublas-verbose,cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--output",
            str(profile_base),
            *command,
        ]

    with log_path.open("w") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log)
        records, missed = collect_telemetry(proc, port, report_path, log_path, args.timeout_seconds)
    returncode = proc.wait()
    traces_path.write_text(json.dumps({"missed": missed, "records": records}, indent=2) + "\n")
    if returncode != 0:
        raise RuntimeError(f"profile worker failed with status {returncode}; inspect {log_path}")
    if not report_path.exists():
        raise RuntimeError(f"profile worker did not write {report_path}")

    report = json.loads(report_path.read_text())
    capture_status = selected_capture_status(
        records, nsys is not None, args.workload
    )
    metadata = {
        "schema_version": 1,
        "suite": "apertus_affon_profile_capture",
        "workload": args.workload,
        "command": command,
        "telemetry": {
            "record_count": len(records),
            "missed": missed,
            "required_spans": sorted(required_spans(args.workload)),
            **capture_status,
        },
        "runtime_evidence_dropped_event_count": report["runtime_evidence"][
            "dropped_event_count"
        ],
        "nsys": None,
    }
    timeline = None

    if nsys:
        nsys_report = output / "apertus-profile.nsys-rep"
        qdstrm_path = output / "apertus-profile.qdstrm"
        sqlite_path = output / "profile.sqlite"
        timeline_path = output / "compute-timeline.json"
        sqlite_path.unlink(missing_ok=True)
        timeline_path.unlink(missing_ok=True)
        if not nsys_report.exists() and qdstrm_path.exists():
            importer = find_qdstrm_importer(nsys)
            if importer is None:
                raise RuntimeError(
                    "Nsight produced a QDSTRM but QdstrmImporter was not found"
                )
            subprocess.run(
                [
                    str(importer),
                    "--input-file",
                    str(qdstrm_path),
                    "--output-file",
                    str(nsys_report),
                    "--force-overwrite",
                ],
                cwd=ROOT,
                check=True,
            )
        if not nsys_report.exists():
            raise RuntimeError(
                f"Nsight did not produce a report or importable QDSTRM; inspect {log_path}"
            )
        subprocess.run(
            [
                str(nsys),
                "export",
                "--type=sqlite",
                "--force-overwrite=true",
                "--output",
                str(sqlite_path),
                str(nsys_report),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(CORRELATOR),
                "--sqlite",
                str(sqlite_path),
                "--output",
                str(timeline_path),
            ],
            cwd=ROOT,
            check=True,
        )
        timeline = json.loads(timeline_path.read_text())
        metadata["nsys"] = {
            "report": str(nsys_report),
            "sqlite": str(sqlite_path),
            "timeline": str(timeline_path),
            "run_count": len(timeline.get("runs", [])),
            "readback_count": len(timeline.get("readbacks", [])),
            "total_device_activities": timeline.get("total_device_activities"),
            "unmatched_device_activities": timeline["unmatched_device_activities"],
            "total_synchronization_activities": timeline.get(
                "total_synchronization_activities"
            ),
            "unmatched_synchronization_activities": timeline.get(
                "unmatched_synchronization_activities"
            ),
            "total_cublas_activities": timeline.get("total_cublas_activities"),
            "unmatched_cublas_activities": timeline.get(
                "unmatched_cublas_activities"
            ),
            "unmatched_cublas": timeline.get("unmatched_cublas", []),
        }

    identity, identity_problems = joined_identity_status(
        report, records, timeline, args.workload
    )
    metadata["identity"] = identity
    timeline_problems = []
    if timeline:
        selected_runs = [
            run
            for run in timeline.get("runs", [])
            if str(run.get("run_id_text", run.get("run_id")))
            == identity["run_id_text"]
            and str(run.get("executable_id_text", run.get("executable_id")))
            == identity["executable_id_text"]
        ]
        if len(selected_runs) == 1:
            selected_run = selected_runs[0]
            metadata["nsys"]["selected_run"] = {
                "run_id": selected_run.get("run_id"),
                "run_id_text": selected_run.get("run_id_text"),
                "executable_id": selected_run.get("executable_id"),
                "executable_id_text": selected_run.get("executable_id_text"),
                "device_active_ns": selected_run.get("device_active_ns"),
                "device_span_ns": selected_run.get("device_span_ns"),
                "device_gap_ns": selected_run.get("device_gap_ns"),
                "classification": selected_run.get("classification"),
                "apis": selected_run.get("apis"),
                "cublas_api_count": selected_run.get("cublas_api_count"),
                "cublas_api_ns": selected_run.get("cublas_api_ns"),
                "cublas_apis": selected_run.get("cublas_apis"),
                "synchronization_apis": selected_run.get(
                    "synchronization_apis"
                ),
                "largest_gaps": selected_run.get("largest_gaps"),
            }
            classified_ns = (selected_run.get("classification") or {}).get(
                "duration_ns"
            )
            if classified_ns != selected_run.get("device_gap_ns"):
                timeline_problems.append(
                    "Nsight selected run gap classification does not cover its device gap"
                )
            if selected_run.get("cublas_api_count", 0) == 0:
                timeline_problems.append(
                    "Nsight selected run has no cuBLAS API evidence"
                )
            gap_attribution, gap_problems = gap_attribution_status(
                selected_run, report
            )
            metadata["gap_attribution"] = gap_attribution
            timeline_problems.extend(gap_problems)
    phases, phase_problems = trace_phase_status(records, args.workload)
    metadata["phases"] = phases
    metadata_path = output / "capture.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    problems = (
        list(identity_problems)
        + list(phase_problems)
        + list(timeline_problems)
    )
    if missed:
        problems.append("telemetry records were missed")
    if capture_status["selected_span_count"] != 1:
        problems.append(
            f"expected exactly one selected {args.workload} trace"
        )
    if capture_status["missing_spans"]:
        problems.append(
            f"selected {args.workload} trace is missing required spans"
        )
    if capture_status["profiler_marker_required"] and not capture_status[
        "profiler_marker_available"
    ]:
        problems.append(
            f"selected {args.workload} trace has no available NVTX markers"
        )
    if metadata["runtime_evidence_dropped_event_count"] != 0:
        problems.append("runtime evidence dropped events")
    if metadata["nsys"] and metadata["nsys"]["run_count"] == 0:
        problems.append("Nsight capture has no correlated Affon run")
    if problems:
        raise RuntimeError(
            f"capture is incomplete ({'; '.join(problems)}); inspect {metadata_path}"
        )


if __name__ == "__main__":
    main()
