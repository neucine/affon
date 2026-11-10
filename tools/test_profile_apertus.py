import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "test"
    / "benchmark"
    / "hf-inference"
    / "profile-apertus.py"
)
SPEC = importlib.util.spec_from_file_location("profile_apertus", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(
    trace_id,
    name,
    kind="span_start",
    attributes=None,
    span_id=None,
    parent_span_id=None,
    timestamp_ns=None,
):
    value = {
        "trace_id": trace_id,
        "name": name,
        "kind": kind,
        "attributes": attributes or {},
    }
    if span_id is not None:
        value["span_id"] = span_id
    if parent_span_id is not None:
        value["parent_span_id"] = parent_span_id
    if timestamp_ns is not None:
        value["timestamp_ns"] = timestamp_ns
    return value


def report(entries=None):
    return {
        "runtime_evidence": {
            "executable_id": 8,
            "entries": entries or [],
        }
    }


def identity_records():
    return [
        record("selected", "apertus.cached_decode.selected"),
        record(
            "selected",
            "compute.execution",
            attributes={"compute.run_id": 4, "compute.executable_id": 8},
        ),
        record(
            "selected",
            "compute.invocation",
            kind="event",
            attributes={"compute.step_index": 0, "compute.invocation_id": 10},
        ),
        record(
            "selected",
            "compute.invocation",
            kind="event",
            attributes={"compute.step_index": 1, "compute.invocation_id": 11},
        ),
    ]


class CaptureStatusTest(unittest.TestCase):
    def test_finds_qdstrm_importer_beside_nsys_target_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "nsight-systems"
            target = root / "target-linux-x64"
            host = root / "host-linux-x64"
            target.mkdir(parents=True)
            host.mkdir()
            nsys = target / "nsys"
            importer = host / "QdstrmImporter"
            nsys.touch()
            importer.touch()
            importer.chmod(0o755)

            self.assertEqual(MODULE.find_qdstrm_importer(nsys), importer.resolve())

    def test_required_spans_must_be_in_the_selected_trace(self):
        records = [
            record("prefill", "apertus.prefill"),
            record("prefill", "affon.bridge.run"),
            record("prefill", "affon.bridge.readback"),
            record("prefill", "compute.execution"),
            record("selected", "apertus.cached_decode.selected"),
            record("selected", "affon.executable.run"),
            record("selected", "affon.bridge.run"),
            record("selected", "inference.generation.inputs"),
            record("selected", "inference.generation.input_cleanup"),
            record("selected", "inference.generation.selection"),
            record("selected", "inference.generation.cache_release"),
            record("selected", "inference.generation.cleanup"),
            record("selected", "compute.execution"),
        ]

        status = MODULE.selected_capture_status(records, False)

        self.assertEqual(status["selected_trace_ids"], ["selected"])
        self.assertEqual(status["missing_spans"], ["affon.bridge.readback"])

    def test_available_nvtx_marker_is_required_when_requested(self):
        records = [
            record("selected", "apertus.cached_decode.selected"),
            record("selected", "affon.executable.run"),
            record("selected", "affon.bridge.run"),
            record("selected", "affon.bridge.readback"),
            record("selected", "inference.generation.inputs"),
            record("selected", "inference.generation.input_cleanup"),
            record("selected", "inference.generation.selection"),
            record("selected", "inference.generation.cache_release"),
            record("selected", "inference.generation.cleanup"),
            record("selected", "compute.execution"),
            {
                "trace_id": "selected",
                "name": "compute.profiler_marker_availability",
                "kind": "event",
                "attributes": {"compute.available": False},
            },
        ]

        status = MODULE.selected_capture_status(records, True)

        self.assertEqual(status["missing_spans"], [])
        self.assertFalse(status["profiler_marker_available"])

    def test_complete_selected_trace_reports_available_markers(self):
        records = [
            record("selected", "apertus.cached_decode.selected"),
            record("selected", "affon.executable.run"),
            record("selected", "affon.bridge.run"),
            record("selected", "affon.bridge.readback"),
            record("selected", "inference.generation.inputs"),
            record("selected", "inference.generation.input_cleanup"),
            record("selected", "inference.generation.selection"),
            record("selected", "inference.generation.cache_release"),
            record("selected", "inference.generation.cleanup"),
            record("selected", "compute.execution"),
            {
                "trace_id": "selected",
                "name": "compute.profiler_marker_availability",
                "kind": "event",
                "attributes": {"compute.available": True},
            },
        ]

        status = MODULE.selected_capture_status(records, True)

        self.assertEqual(status["selected_span_count"], 1)
        self.assertEqual(status["selected_trace_ids"], ["selected"])
        self.assertEqual(status["missing_spans"], [])
        self.assertTrue(status["profiler_marker_available"])

    def test_prefill_workload_uses_its_own_selected_span(self):
        records = [
            record("selected", "apertus.cached_prefill.selected"),
            record("selected", "affon.executable.run"),
            record("selected", "affon.bridge.run"),
            record("selected", "affon.bridge.readback"),
            record("selected", "inference.generation.inputs"),
            record("selected", "inference.generation.input_cleanup"),
            record("selected", "inference.generation.selection"),
            record("selected", "inference.generation.cache_release"),
            record("selected", "inference.generation.cleanup"),
            record("selected", "compute.execution"),
        ]

        status = MODULE.selected_capture_status(
            records, False, workload="prefill"
        )

        self.assertEqual(status["selected_span_count"], 1)
        self.assertEqual(status["selected_trace_ids"], ["selected"])
        self.assertEqual(status["missing_spans"], [])

    def test_identity_joins_telemetry_evidence_and_nsys(self):
        runtime_report = report(
            [
                {"step_index": 0, "invocation_count": 1},
                {"step_index": 1, "invocation_count": 1},
            ]
        )
        timeline = {
            "runs": [
                {
                    "run_id": 4,
                    "executable_id": 8,
                    "steps": [
                        {"step_index": 0, "invocation_ids": [10]},
                        {"step_index": 1, "invocation_ids": [11]},
                    ],
                }
            ]
        }

        summary, problems = MODULE.joined_identity_status(
            runtime_report, identity_records(), timeline
        )

        self.assertEqual(problems, [])
        self.assertEqual(summary["run_id"], 4)
        self.assertEqual(summary["executable_id"], 8)
        self.assertEqual(summary["telemetry_invocation_count"], 2)
        self.assertEqual(summary["runtime_evidence_invocation_count"], 2)
        self.assertEqual(summary["nsys_invocation_count"], 2)

    def test_identity_rejects_different_nsys_step_assignment(self):
        runtime_report = report(
            [
                {"step_index": 0, "invocation_count": 1},
                {"step_index": 1, "invocation_count": 1},
            ]
        )
        timeline = {
            "runs": [
                {
                    "run_id": 4,
                    "executable_id": 8,
                    "steps": [
                        {"step_index": 0, "invocation_ids": [10, 11]},
                    ],
                }
            ]
        }

        _, problems = MODULE.joined_identity_status(
            runtime_report, identity_records(), timeline
        )

        self.assertIn(
            "Nsight invocation IDs or step assignments differ from telemetry",
            problems,
        )

    def test_identity_prefers_exact_64_bit_text_ids(self):
        records = identity_records()
        records[1]["attributes"] = {
            "compute.run_id": 4,
            "compute.run_id_text": "4",
            "compute.executable_id": 9_007_199_254_740_992,
            "compute.executable_id_text": "9007199254740993",
        }
        records[2]["attributes"]["compute.invocation_id"] = 9_007_199_254_740_992
        records[2]["attributes"]["compute.invocation_id_text"] = "9007199254740993"
        runtime_report = report(
            [
                {"step_index": 0, "invocation_count": 1},
                {"step_index": 1, "invocation_count": 1},
            ]
        )
        runtime_report["runtime_evidence"]["executable_id"] = 9_007_199_254_740_993
        timeline = {
            "runs": [
                {
                    "run_id": 4,
                    "executable_id": 9_007_199_254_740_993,
                    "steps": [
                        {"step_index": 0, "invocation_ids": [9_007_199_254_740_993]},
                        {"step_index": 1, "invocation_ids": [11]},
                    ],
                }
            ]
        }

        summary, problems = MODULE.joined_identity_status(
            runtime_report, records, timeline
        )

        self.assertEqual(problems, [])
        self.assertEqual(summary["executable_id"], 9_007_199_254_740_993)
        self.assertEqual(summary["executable_id_text"], "9007199254740993")

    def test_gap_attribution_joins_steps_to_operations_and_submission_cost(self):
        runtime_report = report(
            [
                {
                    "step_index": 0,
                    "operation": "index_select",
                    "physical_implementation": "cuda_indexing_read",
                    "invocation_host_timing_sample_count": 1,
                    "invocation_host_total_ns": 40,
                    "library_call_count": 0,
                    "custom_dispatch_count": 1,
                    "dispatch_count_sample_count": 1,
                    "total_ns": 80,
                },
                {
                    "step_index": 1,
                    "operation": "matmul",
                    "physical_implementation": "cuda_matmul_strided",
                    "invocation_host_timing_sample_count": 1,
                    "invocation_host_total_ns": 20,
                    "library_call_count": 1,
                    "custom_dispatch_count": 0,
                    "dispatch_count_sample_count": 1,
                    "total_ns": 100,
                },
            ]
        )
        selected_run = {
            "classification": {"host_delay_ns": 30},
            "gap_transitions": [
                {
                    "preceding": [{"step_index": 0}],
                    "following": [{"step_index": 1}],
                    "gap_count": 2,
                    "duration_ns": 50,
                    "synchronization_ns": 0,
                    "cublas_api_ns": 10,
                    "cuda_api_ns": 20,
                    "host_delay_ns": 30,
                }
            ],
        }

        attribution, problems = MODULE.gap_attribution_status(
            selected_run, runtime_report
        )

        self.assertEqual(problems, [])
        self.assertEqual(attribution["classified_host_delay_ns"], 30)
        self.assertEqual(attribution["attributed_host_delay_ns"], 30)
        transition = attribution["operation_transitions"][0]
        self.assertEqual(transition["cublas_api_ns"], 10)
        self.assertEqual(transition["preceding_operations"], ["index_select"])
        self.assertEqual(transition["following_operations"], ["matmul"])
        self.assertEqual(transition["preceding_submission_ns"], 40)
        self.assertEqual(
            attribution["host_delay_by_preceding_operation"][0][
                "preceding_operations"
            ],
            ["index_select"],
        )
        self.assertEqual(
            attribution["submission_cost_by_operation"][0]["operation"],
            "index_select",
        )
        by_operation = {
            item["operation"]: item
            for item in attribution["submission_cost_by_operation"]
        }
        self.assertEqual(by_operation["matmul"]["library_call_count"], 1)
        self.assertEqual(by_operation["matmul"]["custom_dispatch_count"], 0)

    def test_direct_child_phases_report_unattributed_time(self):
        records = [
            record(
                "selected",
                "apertus.cached_decode.selected",
                span_id="root",
                timestamp_ns=100,
            ),
            record(
                "selected",
                "apertus.cached_decode.selected",
                kind="span_end",
                span_id="root",
                timestamp_ns=1100,
            ),
            record(
                "selected",
                "affon.executable.run",
                span_id="program",
                parent_span_id="root",
                timestamp_ns=200,
            ),
            record(
                "selected",
                "affon.executable.run",
                kind="span_end",
                span_id="program",
                timestamp_ns=700,
            ),
            record(
                "selected",
                "inference.generation.selection",
                span_id="selection",
                parent_span_id="root",
                timestamp_ns=800,
            ),
            record(
                "selected",
                "inference.generation.selection",
                kind="span_end",
                span_id="selection",
                timestamp_ns=900,
            ),
        ]

        summary, problems = MODULE.trace_phase_status(records)

        self.assertEqual(problems, [])
        self.assertEqual(summary["wall_ns"], 1000)
        self.assertEqual(summary["covered_ns"], 600)
        self.assertEqual(summary["unattributed_ns"], 400)
        self.assertEqual(summary["coverage_ratio"], 0.6)


if __name__ == "__main__":
    unittest.main()
