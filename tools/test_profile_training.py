import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "test"
    / "benchmark"
    / "training"
    / "profile-training.py"
)
SPEC = importlib.util.spec_from_file_location("profile_training", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def report():
    return {
        "runtime_evidence": {
            "kind": "composite_optimization_evidence",
            "components": [
                {
                    "phase": "gradient",
                    "executable_kind": "gradient",
                    "evidence": {
                        "schema_version": 3,
                        "executable_id": 101,
                        "executable_id_text": "101",
                        "phase_memory": [{"run_id": 2}],
                        "dropped_event_count": 0,
                        "kernel_count_evidence": "exact",
                        "entries": [
                            {
                                "step_index": 0,
                                "invocation_count": 1,
                                "kernel_count": 1,
                                "custom_dispatch_count": 1,
                                "library_call_count": 0,
                            },
                            {
                                "step_index": 1,
                                "invocation_count": 1,
                                "kernel_count": 1,
                                "custom_dispatch_count": 1,
                                "library_call_count": 0,
                            },
                        ],
                    },
                },
                {
                    "phase": "optimizer",
                    "executable_kind": "optimizer",
                    "evidence": {
                        "schema_version": 3,
                        "executable_id": 202,
                        "executable_id_text": "202",
                        "phase_memory": [{"run_id": 3}],
                        "dropped_event_count": 0,
                        "kernel_count_evidence": "exact",
                        "entries": [
                            {
                                "step_index": 0,
                                "invocation_count": 1,
                                "kernel_count": 1,
                                "custom_dispatch_count": 1,
                                "library_call_count": 0,
                            },
                        ],
                    },
                },
            ],
        }
    }


def records():
    values = [
        {
            "kind": "span_start",
            "name": "training.step.selected",
            "trace_id": "trace",
            "span_id": "root",
            "parent_span_id": "0000",
            "timestamp_ns": 0,
            "attributes": {},
        },
        {
            "kind": "span_end",
            "name": "training.step.selected",
            "trace_id": "trace",
            "span_id": "root",
            "timestamp_ns": 100,
            "attributes": {},
        },
        {
            "kind": "span_start",
            "name": "affon.executable.run",
            "trace_id": "trace",
            "span_id": "run",
            "parent_span_id": "root",
            "timestamp_ns": 0,
            "attributes": {},
        },
        {
            "kind": "span_end",
            "name": "affon.executable.run",
            "trace_id": "trace",
            "span_id": "run",
            "timestamp_ns": 100,
            "attributes": {},
        },
        {
            "kind": "span_start",
            "name": "affon.bridge.run",
            "trace_id": "trace",
            "span_id": "bridge",
            "parent_span_id": "run",
            "timestamp_ns": 0,
            "attributes": {},
        },
        {
            "kind": "event",
            "name": "compute.profiler_marker_availability",
            "trace_id": "trace",
            "span_id": "bridge",
            "attributes": {"compute.available": True},
        },
    ]
    for run_id, executable_id, span_id, steps in (
        ("2", "1001", "gradient-execution", [0, 1]),
        ("3", "2002", "optimizer-execution", [0]),
    ):
        values.extend(
            [
                {
                    "kind": "span_start",
                    "name": "compute.execution",
                    "trace_id": "trace",
                    "span_id": span_id,
                    "parent_span_id": "bridge",
                    "timestamp_ns": 1,
                    "attributes": {
                        "compute.run_id_text": run_id,
                        "compute.executable_id_text": executable_id,
                    },
                },
                {
                    "kind": "span_end",
                    "name": "compute.execution",
                    "trace_id": "trace",
                    "span_id": span_id,
                    "parent_span_id": "bridge",
                    "timestamp_ns": 99,
                    "attributes": {},
                },
            ]
        )
        for index, step in enumerate(steps):
            values.append(
                {
                    "kind": "event",
                    "name": "compute.invocation",
                    "trace_id": "trace",
                    "span_id": span_id,
                    "attributes": {
                        "compute.step_index": step,
                        "compute.invocation_id_text": f"{run_id}{index}",
                    },
                }
            )
    return values


def timeline():
    return {
        "runs": [
            {
                "run_id_text": "2",
                "executable_id_text": "1001",
                "invocation_count": 2,
                "kernel_count": 2,
                "custom_kernel_count": 2,
                "library_kernel_count": 0,
                "steps": [
                    {
                        "step_index": 0,
                        "invocation_ids_text": ["20"],
                    },
                    {
                        "step_index": 1,
                        "invocation_ids_text": ["21"],
                    },
                ],
            },
            {
                "run_id_text": "3",
                "executable_id_text": "2002",
                "invocation_count": 1,
                "kernel_count": 1,
                "custom_kernel_count": 1,
                "library_kernel_count": 0,
                "steps": [
                    {
                        "step_index": 0,
                        "invocation_ids_text": ["30"],
                    },
                ],
            },
        ]
    }


class TrainingCaptureTest(unittest.TestCase):
    def test_joins_both_components_across_all_sources(self):
        identity, problems = MODULE.component_identity_status(
            report(), records(), timeline()
        )
        self.assertEqual(problems, [])
        self.assertEqual(identity["selected_run_ids"], ["2", "3"])
        gradient, optimizer = identity["components"]
        self.assertEqual(
            gradient["compiled_executable_id_text"], "101"
        )
        self.assertEqual(
            gradient["runtime_executable_id_text"], "1001"
        )
        self.assertEqual(gradient["telemetry_invocation_count"], 2)
        self.assertEqual(
            gradient["runtime_evidence_invocation_count"], 2
        )
        self.assertEqual(
            gradient["nsys"]["invocation_count"], 2
        )
        self.assertEqual(optimizer["run_id_text"], "3")
        self.assertEqual(
            optimizer["runtime_executable_id_text"], "2002"
        )
        self.assertEqual(
            gradient["runtime_evidence_schema_version"], 3
        )
        self.assertEqual(gradient["kernel_count_evidence"], "exact")

    def test_rejects_execution_identity_or_invocation_mismatch(self):
        trace = timeline()
        trace["runs"][1]["executable_id_text"] = "wrong"
        _, problems = MODULE.component_identity_status(
            report(), records(), trace
        )
        self.assertTrue(
            any("no matching Nsight" in problem for problem in problems)
        )

        changed = records()
        for record in changed:
            if (
                record.get("name") == "compute.invocation"
                and record.get("span_id") == "gradient-execution"
            ):
                record["attributes"]["compute.step_index"] = 2
                break
        _, problems = MODULE.component_identity_status(
            report(), changed, timeline()
        )
        self.assertTrue(
            any(
                "invocation IDs or steps differ" in problem
                for problem in problems
            )
        )

    def test_component_evidence_must_be_complete(self):
        changed = report()
        changed["runtime_evidence"]["components"][1]["evidence"][
            "dropped_event_count"
        ] = 1
        _, problems = MODULE.component_identity_status(
            changed, records(), timeline()
        )
        self.assertTrue(
            any("optimizer runtime evidence dropped" in problem for problem in problems)
        )

    def test_requires_schema_three_and_valid_kernel_count_evidence(self):
        changed = report()
        gradient = changed["runtime_evidence"]["components"][0]["evidence"]
        gradient["schema_version"] = 2
        gradient["kernel_count_evidence"] = "unknown"
        _, problems = MODULE.component_identity_status(
            changed, records(), timeline()
        )
        self.assertTrue(
            any("schema version 3" in problem for problem in problems)
        )
        self.assertTrue(
            any(
                "invalid kernel count evidence" in problem
                for problem in problems
            )
        )

    def test_library_calls_require_unknown_fanout_marker(self):
        changed = report()
        gradient = changed["runtime_evidence"]["components"][0]["evidence"]
        gradient["entries"][0]["library_call_count"] = 1
        _, problems = MODULE.component_identity_status(
            changed, records(), timeline()
        )
        self.assertTrue(
            any(
                "kernel count evidence disagrees" in problem
                for problem in problems
            )
        )


if __name__ == "__main__":
    unittest.main()
