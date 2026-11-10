#!/usr/bin/env python3
"""Capture one warmed GPT-2 inference Program as one joined artifact."""

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
WORKER = Path(__file__).with_name("inference-program-profile.ts")
CORRELATOR = ROOT / "tools" / "correlate-nsys-compute.py"
SELECTED_SPAN = "gpt2.prefix_inference.selected"
REQUIRED_SPANS = {
    SELECTED_SPAN,
    "affon.executable.run",
    "affon.bridge.run",
    "affon.bridge.readback",
    "compute.execution",
}
VALID_KERNEL_COUNT_EVIDENCE = {
    "exact",
    "custom_dispatches_exact_library_fanout_unknown",
}


def exact_id_text(attributes, key):
    value = attributes.get(f"{key}_text", attributes.get(key))
    return None if value is None else str(value)


def telemetry_scope(records):
    selected = [
        record
        for record in records
        if record.get("kind") == "span_start"
        and record.get("name") == SELECTED_SPAN
    ]
    trace_ids = {
        record["trace_id"] for record in selected if record.get("trace_id")
    }
    return selected, [
        record for record in records if record.get("trace_id") in trace_ids
    ]


def selected_capture_status(records, require_profiler_markers):
    selected, selected_records = telemetry_scope(records)
    started = {
        record.get("name")
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
        "selected_trace_ids": sorted(
            {
                record["trace_id"]
                for record in selected
                if record.get("trace_id")
            }
        ),
        "missing_spans": sorted(REQUIRED_SPANS - started),
        "profiler_marker_available": marker_available,
        "profiler_marker_required": require_profiler_markers,
    }


def execution_records(selected_records):
    executions = {}
    problems = []
    for record in selected_records:
        if (
            record.get("kind") != "span_start"
            or record.get("name") != "compute.execution"
        ):
            continue
        attributes = record.get("attributes", {})
        run_id_text = exact_id_text(attributes, "compute.run_id")
        executable_id_text = exact_id_text(
            attributes, "compute.executable_id"
        )
        span_id = record.get("span_id")
        if run_id_text is None or executable_id_text is None or span_id is None:
            problems.append("selected inference trace has an incomplete execution")
            continue
        if run_id_text in executions:
            problems.append(
                f"selected inference trace repeats run ID {run_id_text}"
            )
            continue
        executions[run_id_text] = {
            "run_id_text": run_id_text,
            "runtime_executable_id_text": executable_id_text,
            "span_id": span_id,
        }
    return executions, problems


def evidence_steps(evidence):
    return {
        int(entry["step_index"]): int(entry.get("invocation_count", 0))
        for entry in evidence.get("entries", [])
        if "step_index" in entry
    }


def telemetry_steps(selected_records, execution):
    steps = {}
    malformed = 0
    for record in selected_records:
        if (
            record.get("kind") != "event"
            or record.get("name") != "compute.invocation"
            or record.get("span_id") != execution["span_id"]
        ):
            continue
        attributes = record.get("attributes", {})
        invocation_id_text = exact_id_text(
            attributes, "compute.invocation_id"
        )
        if (
            "compute.step_index" not in attributes
            or invocation_id_text is None
        ):
            malformed += 1
            continue
        steps.setdefault(int(attributes["compute.step_index"]), set()).add(
            invocation_id_text
        )
    return {
        step: sorted(invocations)
        for step, invocations in sorted(steps.items())
    }, malformed


def timeline_steps(run):
    return {
        int(step["step_index"]): sorted(
            str(invocation)
            for invocation in step.get(
                "invocation_ids_text", step.get("invocation_ids", [])
            )
        )
        for step in run.get("steps", [])
    }


def identity_status(report, records, timeline=None):
    selected, selected_records = telemetry_scope(records)
    problems = []
    if len(selected) != 1:
        problems.append("expected exactly one selected inference trace")

    executions, execution_problems = execution_records(selected_records)
    problems.extend(execution_problems)
    evidence = report.get("runtime_evidence", {})
    if evidence.get("schema_version") != 3:
        problems.append("runtime evidence must use aggregate schema version 3")
    kernel_count_evidence = evidence.get("kernel_count_evidence")
    if kernel_count_evidence not in VALID_KERNEL_COUNT_EVIDENCE:
        problems.append("runtime evidence has invalid kernel count evidence")
    run_ids = sorted(
        {
            str(memory["run_id"])
            for memory in evidence.get("phase_memory", [])
            if "run_id" in memory
        }
    )
    run_id_text = run_ids[0] if len(run_ids) == 1 else None
    if len(run_ids) != 1:
        problems.append(
            "runtime evidence does not identify exactly one selected run"
        )
    execution = executions.get(run_id_text) if run_id_text else None
    if run_id_text and execution is None:
        problems.append(
            f"runtime evidence run {run_id_text} is absent from telemetry"
        )
    if len(executions) != 1:
        problems.append(
            "selected inference trace does not have exactly one execution"
        )

    evidence_executable_id_text = str(
        evidence.get("executable_id_text", evidence.get("executable_id"))
    )
    runtime_executable_id_text = (
        execution["runtime_executable_id_text"] if execution else None
    )
    if (
        runtime_executable_id_text is not None
        and evidence_executable_id_text != runtime_executable_id_text
    ):
        problems.append(
            "runtime evidence and telemetry executable IDs differ"
        )

    evidence_counts = evidence_steps(evidence)
    telemetry_steps_by_span = {}
    malformed_invocations = None
    if execution:
        telemetry_steps_by_span, malformed_invocations = telemetry_steps(
            selected_records, execution
        )
    telemetry_counts = {
        step: len(invocations)
        for step, invocations in telemetry_steps_by_span.items()
    }
    compared_steps = set(evidence_counts) | set(telemetry_counts)
    if any(
        evidence_counts.get(step, 0) != telemetry_counts.get(step, 0)
        for step in compared_steps
    ):
        problems.append(
            "runtime evidence invocation counts differ from telemetry"
        )
    if malformed_invocations:
        problems.append("selected inference trace has malformed invocation telemetry")
    if int(evidence.get("dropped_event_count", 0)) != 0:
        problems.append("runtime evidence dropped events")

    custom_dispatch_count = sum(
        int(entry.get("custom_dispatch_count", 0))
        for entry in evidence.get("entries", [])
    )
    library_call_count = sum(
        int(entry.get("library_call_count", 0))
        for entry in evidence.get("entries", [])
    )
    expected_kernel_count_evidence = (
        "custom_dispatches_exact_library_fanout_unknown"
        if library_call_count > 0
        else "exact"
    )
    if kernel_count_evidence != expected_kernel_count_evidence:
        problems.append(
            "kernel count evidence disagrees with observed library calls"
        )
    runtime_kernel_count = sum(
        int(entry.get("kernel_count", 0))
        for entry in evidence.get("entries", [])
    )

    nsys = None
    if timeline is not None and run_id_text and runtime_executable_id_text:
        matching_runs = [
            run
            for run in timeline.get("runs", [])
            if str(run.get("run_id_text", run.get("run_id"))) == run_id_text
            and str(
                run.get(
                    "executable_id_text", run.get("executable_id")
                )
            )
            == runtime_executable_id_text
        ]
        if len(matching_runs) != 1:
            problems.append(
                "Nsight timeline does not have exactly one matching Affon run"
            )
        else:
            run = matching_runs[0]
            if timeline_steps(run) != {
                step: [str(invocation) for invocation in invocations]
                for step, invocations in telemetry_steps_by_span.items()
            }:
                problems.append(
                    "Nsight invocation IDs or step assignments differ from telemetry"
                )
            if int(run.get("custom_kernel_count", 0)) != custom_dispatch_count:
                problems.append(
                    "Nsight custom kernel count differs from runtime evidence"
                )
            if int(run.get("cublas_api_count", 0)) != library_call_count:
                problems.append(
                    "Nsight library API count differs from runtime evidence"
                )
            nsys = {
                "invocation_count": int(run.get("invocation_count", 0)),
                "kernel_count": int(run.get("kernel_count", 0)),
                "custom_kernel_count": int(
                    run.get("custom_kernel_count", 0)
                ),
                "library_kernel_count": int(
                    run.get("library_kernel_count", 0)
                ),
                "cublas_api_count": int(run.get("cublas_api_count", 0)),
                "memcpy_count": int(run.get("memcpy_count", 0)),
                "memset_count": int(run.get("memset_count", 0)),
                "device_active_ns": run.get("device_active_ns"),
                "device_span_ns": run.get("device_span_ns"),
                "device_gap_ns": run.get("device_gap_ns"),
                "classification": run.get("classification"),
            }
            nsys["kernel_count_accounting_gap"] = (
                nsys["kernel_count"] - runtime_kernel_count
            )
            nsys["library_kernel_fanout"] = (
                nsys["library_kernel_count"]
                - nsys["cublas_api_count"]
            )
            if (
                nsys["kernel_count_accounting_gap"] != 0
                and kernel_count_evidence == "exact"
            ):
                problems.append(
                    "exact runtime kernel count differs from Nsight"
                )

    return {
        "run_id_text": run_id_text,
        "runtime_executable_id_text": runtime_executable_id_text,
        "compiled_executable_id_text": evidence_executable_id_text,
        "telemetry_invocation_count": sum(telemetry_counts.values()),
        "runtime_evidence_invocation_count": sum(evidence_counts.values()),
        "runtime_evidence_schema_version": evidence.get("schema_version"),
        "runtime_evidence_kernel_count": runtime_kernel_count,
        "kernel_count_evidence": kernel_count_evidence,
        "custom_dispatch_count": custom_dispatch_count,
        "library_call_count": library_call_count,
        "nsys": nsys,
    }, problems


def trace_phase_status(records):
    selected, selected_records = telemetry_scope(records)
    if len(selected) != 1:
        return {
            "wall_ns": None,
            "covered_ns": None,
            "unattributed_ns": None,
            "coverage_ratio": None,
            "phases": [],
        }, ["expected exactly one selected inference trace"]
    root = selected[0]
    trace_id = root.get("trace_id")
    span_id = root.get("span_id")
    ends = [
        record
        for record in selected_records
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
        }, ["selected inference trace does not have exactly one end record"]

    root_start = int(root["timestamp_ns"])
    root_end = int(ends[0]["timestamp_ns"])
    starts = {
        record.get("span_id"): record
        for record in selected_records
        if record.get("kind") == "span_start"
        and record.get("parent_span_id") == span_id
    }
    phases = []
    intervals = []
    for record in selected_records:
        if (
            record.get("kind") != "span_end"
            or record.get("span_id") not in starts
        ):
            continue
        start = int(starts[record["span_id"]]["timestamp_ns"])
        end = int(record["timestamp_ns"])
        if start < root_start or end > root_end or end < start:
            continue
        phases.append(
            {
                "name": record.get("name"),
                "elapsed_ns": end - start,
            }
        )
        intervals.append((start, end))
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    covered = sum(end - start for start, end in merged)
    wall = root_end - root_start
    return {
        "wall_ns": wall,
        "covered_ns": covered,
        "unattributed_ns": max(0, wall - covered),
        "coverage_ratio": covered / wall if wall else None,
        "phases": phases,
    }, []


def find_qdstrm_importer(nsys):
    nsys_path = Path(nsys).resolve()
    candidates = [
        nsys_path.parent.parent / "host-linux-x64" / "QdstrmImporter",
        Path("/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter"),
        Path(
            "/usr/lib/nsight-compute/host/"
            "linux-desktop-glibc_2_11_3-x64/QdstrmImporter"
        ),
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which("QdstrmImporter")
    return Path(found) if found else None


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def collect_telemetry(proc, port, timeout, log_path):
    records = []
    cursor = 0
    missed = False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/traces?since={cursor}",
                timeout=2,
            ) as response:
                batch = json.load(response)
            records.extend(batch["records"])
            cursor = batch["next_cursor"]
            missed |= batch["missed"]
        except OSError:
            if proc.poll() is not None:
                break
        if proc.poll() is not None:
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
        raise RuntimeError(f"inference profile timed out; inspect {log_path}")
    return records, missed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affon", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--device", default="cuda", choices=("cuda", "metal", "cpu")
    )
    parser.add_argument("--capture-records", default=262144, type=int)
    parser.add_argument("--telemetry-records", default=131072, type=int)
    parser.add_argument("--nsys", type=Path)
    parser.add_argument("--no-nsys", action="store_true")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    if args.capture_records < 1 or args.capture_records > 4194304:
        raise RuntimeError(
            "--capture-records must be between 1 and 4194304"
        )
    if args.telemetry_records < 512 or args.telemetry_records > 1048576:
        raise RuntimeError(
            "--telemetry-records must be between 512 and 1048576"
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "inference-profile.json"
    traces_path = output / "telemetry-traces.json"
    log_path = output / "run.log"
    report_path.unlink(missing_ok=True)
    traces_path.unlink(missing_ok=True)

    nsys = None
    if not args.no_nsys:
        found = args.nsys or (
            Path(shutil.which("nsys")) if shutil.which("nsys") else None
        )
        if found is None:
            raise RuntimeError("nsys was not found; pass --nsys or --no-nsys")
        nsys = found

    port = args.port or free_port()
    env = {
        **os.environ,
        "AFFON_INFERENCE_PROFILE_PATH": str(report_path),
        "AFFON_PROFILE_DEVICE": args.device,
        "AFFON_PROFILE_CAPTURE_RECORDS": str(args.capture_records),
        "RUNTIME_TELEMETRY_CONSOLE": "1",
        "RUNTIME_TELEMETRY_CONSOLE_PORT": str(port),
        "RUNTIME_TELEMETRY_TRACE_RECORDS": str(args.telemetry_records),
    }
    command = [str(args.affon.resolve()), str(WORKER)]
    if nsys:
        command = [
            str(nsys),
            "profile",
            "--force-overwrite=true",
            "--trace=cublas-verbose,cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--output",
            str(output / "inference-profile"),
            *command,
        ]

    with log_path.open("w") as log:
        proc = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=log
        )
        records, missed = collect_telemetry(
            proc, port, args.timeout_seconds, log_path
        )
    returncode = proc.wait()
    traces_path.write_text(
        json.dumps({"missed": missed, "records": records}, indent=2)
        + "\n"
    )
    if returncode != 0:
        raise RuntimeError(
            f"inference worker failed with status {returncode}; "
            f"inspect {log_path}"
        )
    if not report_path.exists():
        raise RuntimeError(f"inference worker did not write {report_path}")

    report = json.loads(report_path.read_text())
    capture_status = selected_capture_status(records, nsys is not None)
    metadata = {
        "schema_version": 1,
        "suite": "gpt2_program_inference_profile_capture",
        "command": command,
        "telemetry": {
            "record_count": len(records),
            "missed": missed,
            "required_spans": sorted(REQUIRED_SPANS),
            **capture_status,
        },
        "runtime_evidence_dropped_event_count": report[
            "runtime_evidence"
        ]["dropped_event_count"],
        "nsys": None,
    }
    timeline = None
    if nsys:
        nsys_report = output / "inference-profile.nsys-rep"
        qdstrm_path = output / "inference-profile.qdstrm"
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
                f"Nsight did not produce an importable report; "
                f"inspect {log_path}"
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
            "total_device_activities": timeline.get(
                "total_device_activities"
            ),
            "unmatched_device_activities": timeline.get(
                "unmatched_device_activities"
            ),
            "total_synchronization_activities": timeline.get(
                "total_synchronization_activities"
            ),
            "unmatched_synchronization_activities": timeline.get(
                "unmatched_synchronization_activities"
            ),
            "outside_synchronization_activities": timeline.get(
                "outside_synchronization_activities"
            ),
            "total_cublas_activities": timeline.get(
                "total_cublas_activities"
            ),
            "total_cublas_lifecycle_activities": timeline.get(
                "total_cublas_lifecycle_activities"
            ),
            "unmatched_cublas_activities": timeline.get(
                "unmatched_cublas_activities"
            ),
            "conflicting_correlation_ids": timeline.get(
                "conflicting_correlation_ids"
            ),
        }

    identity, identity_problems = identity_status(
        report, records, timeline
    )
    phases, phase_problems = trace_phase_status(records)
    metadata["identity"] = identity
    metadata["phases"] = phases
    metadata_path = output / "capture.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))

    problems = [*identity_problems, *phase_problems]
    if missed:
        problems.append("telemetry records were missed")
    if capture_status["selected_span_count"] != 1:
        problems.append("expected exactly one selected inference trace")
    if capture_status["missing_spans"]:
        problems.append("selected inference trace is missing required spans")
    if (
        capture_status["profiler_marker_required"]
        and not capture_status["profiler_marker_available"]
    ):
        problems.append(
            "selected inference trace has no available NVTX markers"
        )
    if timeline:
        nsys_metadata = metadata["nsys"]
        for key in (
            "unmatched_device_activities",
            "unmatched_synchronization_activities",
            "unmatched_cublas_activities",
            "conflicting_correlation_ids",
        ):
            if nsys_metadata.get(key, 0) != 0:
                problems.append(f"Nsight capture has nonzero {key}")
    if problems:
        raise RuntimeError(
            f"inference capture is incomplete "
            f"({'; '.join(problems)}); inspect {metadata_path}"
        )


if __name__ == "__main__":
    main()
