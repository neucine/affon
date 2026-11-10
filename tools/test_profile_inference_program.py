import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "test"
    / "benchmark"
    / "hf-inference"
    / "profile-inference-program.py"
)
SPEC = importlib.util.spec_from_file_location(
    "profile_inference_program", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def report():
    return {
        "runtime_evidence": {
            "schema_version": 3,
            "executable_id": 2002,
            "executable_id_text": "2002",
            "phase_memory": [{"run_id": 3}],
            "dropped_event_count": 0,
            "kernel_count_evidence": "exact",
            "entries": [
                {
                    "step_index": 0,
                    "invocation_count": 1,
                    "kernel_count": 2,
                    "custom_dispatch_count": 2,
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
        }
    }


def records():
    return [
        {
            "kind": "span_start",
            "name": MODULE.SELECTED_SPAN,
            "trace_id": "trace",
            "span_id": "root",
            "timestamp_ns": 0,
            "attributes": {},
        },
        {
            "kind": "span_end",
            "name": MODULE.SELECTED_SPAN,
            "trace_id": "trace",
            "span_id": "root",
            "timestamp_ns": 100,
            "attributes": {},
        },
        {
            "kind": "span_start",
            "name": "compute.execution",
            "trace_id": "trace",
            "span_id": "execution",
            "parent_span_id": "root",
            "timestamp_ns": 1,
            "attributes": {
                "compute.run_id_text": "3",
                "compute.executable_id_text": "2002",
            },
        },
        {
            "kind": "span_end",
            "name": "compute.execution",
            "trace_id": "trace",
            "span_id": "execution",
            "parent_span_id": "0000",
            "timestamp_ns": 99,
            "attributes": {},
        },
        {
            "kind": "event",
            "name": "compute.invocation",
            "trace_id": "trace",
            "span_id": "execution",
            "attributes": {
                "compute.step_index": 0,
                "compute.invocation_id_text": "30",
            },
        },
        {
            "kind": "event",
            "name": "compute.invocation",
            "trace_id": "trace",
            "span_id": "execution",
            "attributes": {
                "compute.step_index": 1,
                "compute.invocation_id_text": "31",
            },
        },
    ]


def timeline():
    return {
        "runs": [
            {
                "run_id_text": "3",
                "executable_id_text": "2002",
                "invocation_count": 2,
                "kernel_count": 3,
                "custom_kernel_count": 3,
                "library_kernel_count": 0,
                "cublas_api_count": 0,
                "steps": [
                    {
                        "step_index": 0,
                        "invocation_ids_text": ["30"],
                    },
                    {
                        "step_index": 1,
                        "invocation_ids_text": ["31"],
                    },
                ],
            }
        ]
    }


class InferenceProfileContractTest(unittest.TestCase):
    def test_joins_schema_three_evidence_to_telemetry_and_nsys(self):
        identity, problems = MODULE.identity_status(
            report(), records(), timeline()
        )
        self.assertEqual(problems, [])
        self.assertEqual(identity["run_id_text"], "3")
        self.assertEqual(identity["runtime_evidence_schema_version"], 3)
        self.assertEqual(identity["telemetry_invocation_count"], 2)
        self.assertEqual(identity["runtime_evidence_kernel_count"], 3)
        self.assertEqual(identity["nsys"]["kernel_count"], 3)
        self.assertEqual(
            identity["nsys"]["kernel_count_accounting_gap"], 0
        )

    def test_rejects_missing_schema_three_contract(self):
        changed = report()
        changed["runtime_evidence"]["schema_version"] = 2
        changed["runtime_evidence"]["kernel_count_evidence"] = "unknown"
        _, problems = MODULE.identity_status(
            changed, records(), timeline()
        )
        self.assertTrue(
            any("aggregate schema version 3" in problem for problem in problems)
        )
        self.assertTrue(
            any("invalid kernel count evidence" in problem for problem in problems)
        )

    def test_rejects_library_counts_without_unknown_fanout(self):
        changed = report()
        evidence = changed["runtime_evidence"]
        evidence["entries"][0]["library_call_count"] = 1
        evidence["entries"][0]["custom_dispatch_count"] = 1
        evidence["entries"][0]["kernel_count"] = 2
        trace = timeline()
        trace["runs"][0]["custom_kernel_count"] = 2
        trace["runs"][0]["cublas_api_count"] = 1
        trace["runs"][0]["library_kernel_count"] = 1
        trace["runs"][0]["kernel_count"] = 3
        _, problems = MODULE.identity_status(changed, records(), trace)
        self.assertTrue(
            any(
                "kernel count evidence disagrees" in problem
                for problem in problems
            )
        )

    def test_phase_coverage_handles_zero_parent_end_ids(self):
        phases, problems = MODULE.trace_phase_status(records())
        self.assertEqual(problems, [])
        self.assertEqual(phases["covered_ns"], 98)
        self.assertEqual(phases["unattributed_ns"], 2)
        self.assertEqual(phases["phases"][0]["name"], "compute.execution")


if __name__ == "__main__":
    unittest.main()
