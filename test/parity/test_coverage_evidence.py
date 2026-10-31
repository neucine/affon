import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("coverage_evidence", ROOT / "tools/import-coverage-evidence.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


class CoverageEvidenceTest(unittest.TestCase):
    def test_view_evidence_requires_matching_base_shapes(self):
        cell = "metal/f32/add/strided"
        path_choice = [{"implementation_id": "view.slice", "view_slice_ranges": [
            {"start": 0, "stop": 8, "step": 1}, {"start": 0, "stop": 32, "step": 2}]},
            {"implementation_id": "metal_add"}]
        probe = {"schema_version": 1, "suite": "affon_capability_probe_run",
                 "binary_sha256": "abc", "catalog_sha256": "def", "cases": [{
                     "id": cell, "status": "correct", "shape": [8, 16],
                     "input_shapes": [[8, 32], [8, 32]], "output_shape": [8, 16], "maximum_error": 0,
                     "tolerance": 0.01, "selected_path": path_choice}]}
        benchmark = {"schema_version": 1, "suite": "affon_cell_benchmark",
                     "cell_id": cell, "kind": "operator_peer", "target": "mac-exact",
                     "affon_binary_sha256": "abc", "peer_engine": "pytorch",
                     "selected_path": path_choice, "correctness": {"passed": True},
                     "contract": {"shape": [8, 16], "input_shapes": [[8, 16], [8, 16]], "output_shape": [8, 16],
                                  "dtype": "f32", "layout": "stride_2_columns",
                                  "timing_scope": "resident synchronized", "optimization_level": "safe",
                                  "input_sha256": "fixture-sha"},
                     "affon_samples_ms": [1, 1.1, 1.2], "peer_samples_ms": [0.9, 1, 1.1]}
        with tempfile.TemporaryDirectory() as folder:
            probe_path, benchmark_path = Path(folder) / "probe.json", Path(folder) / "benchmark.json"
            probe_path.write_text(json.dumps(probe))
            benchmark_path.write_text(json.dumps(benchmark))
            with self.assertRaisesRegex(ValueError, "base input shapes differ"):
                evidence.import_benchmark(benchmark_path, evidence.import_probe(probe_path, {cell}))
            benchmark["contract"]["input_shapes"] = [[8, 32], [8, 32]]
            benchmark["contract"]["output_shape"] = [16, 8]
            benchmark_path.write_text(json.dumps(benchmark))
            with self.assertRaisesRegex(ValueError, "output shape differs"):
                evidence.import_benchmark(benchmark_path, evidence.import_probe(probe_path, {cell}))
            benchmark["contract"]["output_shape"] = [8, 16]
            probe["cases"][0]["output_dtype"] = "f32"
            probe_path.write_text(json.dumps(probe))
            benchmark["contract"]["output_dtype"] = "bf16"
            benchmark_path.write_text(json.dumps(benchmark))
            with self.assertRaisesRegex(ValueError, "output dtype differs"):
                evidence.import_benchmark(benchmark_path, evidence.import_probe(probe_path, {cell}))
            benchmark["contract"]["output_dtype"] = "f32"
            benchmark_path.write_text(json.dumps(benchmark))
            outcomes = evidence.import_probe(probe_path, {cell})
            evidence.import_benchmark(benchmark_path, outcomes)
            self.assertIn("operator_peer", outcomes[cell]["evidence"])

    def test_self_baseline_requires_accepted_matching_session_gate(self):
        cell = "metal/f32/matmul/decode_m1"
        path_choice = [{"implementation_id": "metal_matmul"}]
        probe = {"schema_version": 1, "suite": "affon_capability_probe_run",
                 "binary_sha256": "abc", "catalog_sha256": "def", "cases": [{
                     "id": cell, "status": "correct", "shape": [1, 32, 32],
                     "maximum_error": 0, "tolerance": 0.01, "selected_path": path_choice}]}
        benchmark = {"schema_version": 1, "suite": "affon_cell_benchmark", "cell_id": cell,
                     "kind": "self_baseline", "target": "mac-exact", "affon_binary_sha256": "abc",
                     "peer_engine": "affon", "selected_path": path_choice,
                     "correctness": {"passed": True}, "contract": {"shape": [1, 32, 32],
                     "dtype": "f32", "layout": "dense", "timing_scope": "resident synchronized",
                     "optimization_level": "safe", "input_sha256": "fixture-sha"},
                     "affon_samples_ms": [1, 1.1, 1.2], "peer_samples_ms": [1, 1.1, 1.2]}
        with tempfile.TemporaryDirectory() as folder:
            probe_path = Path(folder) / "probe.json"
            benchmark_path = Path(folder) / "benchmark.json"
            evaluation_path = Path(folder) / "evaluation.json"
            probe_path.write_text(json.dumps(probe))
            benchmark_path.write_text(json.dumps(benchmark))
            with self.assertRaisesRegex(ValueError, "accepted session evaluation missing"):
                evidence.import_benchmark(benchmark_path, evidence.import_probe(probe_path, {cell}))
            benchmark["baseline_evaluation"] = str(evaluation_path)
            benchmark_path.write_text(json.dumps(benchmark))
            evaluation = {"suite": "affon_cell_baseline_evaluation", "accepted": False,
                          "cell_id": cell, "target": "mac-exact", "binary_sha256": "abc",
                          "selected_path": path_choice, "contract": benchmark["contract"]}
            evaluation_path.write_text(json.dumps(evaluation))
            with self.assertRaisesRegex(ValueError, "session evaluation rejected"):
                evidence.import_benchmark(benchmark_path, evidence.import_probe(probe_path, {cell}))
            evaluation["accepted"] = True
            evaluation_path.write_text(json.dumps(evaluation))
            with self.assertRaisesRegex(ValueError, "lacks source artifacts"):
                evidence.import_benchmark(benchmark_path, evidence.import_probe(probe_path, {cell}))
            spec = importlib.util.spec_from_file_location("baseline_evaluator", ROOT / "tools/evaluate-cell-baseline.py")
            evaluator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evaluator)
            calibration = {"suite": "affon_cell_baseline_calibration", "cell_id": cell,
                "target": "mac-exact", "binary_sha256": "abc", "harness_sha256": "harness",
                "selected_path": path_choice, "contract": benchmark["contract"],
                "machine_state_at_start": {"power": "AC"}, "machine_state_at_end": {"power": "AC"},
                "started_at": "2026-10-29T10:00:00Z", "ended_at": "2026-10-29T10:10:00Z",
                "runs": [{"median_ms": value} for value in (1, 1.01, 1.02)]}
            first, second = Path(folder) / "a.json", Path(folder) / "b.json"
            first.write_text(json.dumps(calibration))
            calibration["started_at"] = "2026-10-29T11:00:00Z"
            second.write_text(json.dumps(calibration))
            evaluation = evaluator.evaluate([json.loads(first.read_text()), json.loads(second.read_text())],
                                            sources=[first, second])
            evaluation_path.write_text(json.dumps(evaluation))
            outcomes = evidence.import_probe(probe_path, {cell})
            evidence.import_benchmark(benchmark_path, outcomes)
            self.assertIn("self_baseline", outcomes[cell]["evidence"])
            first.write_text("{}")
            with self.assertRaisesRegex(ValueError, "session evaluation source changed"):
                evidence.import_benchmark(benchmark_path, evidence.import_probe(probe_path, {cell}))

    def test_probe_import_and_exact_benchmark_contract(self):
        cell = "metal/f32/matmul/decode_m1"
        path_choice = [{"implementation_id": "metal_matmul", "implementation_version": 1, "configuration_digest": 7}]
        probe = {"schema_version": 1, "suite": "affon_capability_probe_run", "binary_sha256": "abc",
                 "catalog_sha256": "def", "cases": [{"id": cell, "status": "correct", "shape": [1, 32, 32],
                 "maximum_error": 0, "tolerance": 0.01, "selected_path": path_choice}]}
        benchmark = {"schema_version": 1, "suite": "affon_cell_benchmark", "cell_id": cell,
                     "kind": "operator_peer", "target": "mac-exact", "affon_binary_sha256": "abc",
                     "peer_engine": "pytorch", "selected_path": path_choice,
                     "correctness": {"passed": True},
                     "contract": {"shape": [1, 32, 32], "dtype": "f32", "layout": "dense",
                                  "timing_scope": "resident synchronized", "optimization_level": "safe",
                                  "input_sha256": "fixture-sha"},
                     "affon_samples_ms": [1, 1.1, 1.2], "peer_samples_ms": [0.9, 1, 1.1]}
        with tempfile.TemporaryDirectory() as folder:
            probe_path, benchmark_path = Path(folder) / "probe.json", Path(folder) / "benchmark.json"
            probe_path.write_text(json.dumps(probe))
            benchmark_path.write_text(json.dumps(benchmark))
            outcomes = evidence.import_probe(probe_path, {cell})
            evidence.import_benchmark(benchmark_path, outcomes)
            self.assertIn("operator_peer", outcomes[cell]["evidence"])
            benchmark["selected_path"] = [{**path_choice[0], "implementation_id": "other"}]
            benchmark_path.write_text(json.dumps(benchmark))
            with self.assertRaisesRegex(ValueError, "selected implementation differs"):
                evidence.import_benchmark(benchmark_path, {cell: evidence.import_probe(probe_path, {cell})[cell]})
            benchmark["selected_path"] = path_choice
            benchmark["contract"]["compile_options"] = {"dense_inputs": True}
            benchmark_path.write_text(json.dumps(benchmark))
            with self.assertRaisesRegex(ValueError, "compile options differ"):
                evidence.import_benchmark(benchmark_path, {cell: evidence.import_probe(probe_path, {cell})[cell]})
            benchmark["contract"]["compile_options"] = {"dense_inputs": False}
            benchmark["kind"] = "program_peer"
            benchmark_path.write_text(json.dumps(benchmark))
            with self.assertRaisesRegex(ValueError, "Program graph contract missing"):
                evidence.import_benchmark(benchmark_path, {cell: evidence.import_probe(probe_path, {cell})[cell]})
            benchmark["contract"]["program"] = {"id": "op_then_add_self_v1", "ops": ["matmul", "add_self"]}
            benchmark["selected_path"] = path_choice + [{**path_choice[0], "implementation_id": "metal_add"}]
            benchmark_path.write_text(json.dumps(benchmark))
            program_outcomes = evidence.import_probe(probe_path, {cell})
            evidence.import_benchmark(benchmark_path, program_outcomes)
            self.assertIn("program_peer", program_outcomes[cell]["evidence"])
            operator_outcomes = evidence.import_probe(probe_path, {cell})
            benchmark["kind"] = "operator_peer"
            benchmark["selected_path"] = path_choice
            benchmark["contract"].pop("program")
            benchmark_path.write_text(json.dumps(benchmark))
            evidence.import_benchmark(benchmark_path, operator_outcomes)
            benchmark["kind"] = "program_peer"
            benchmark["target"] = "different-mac"
            benchmark["contract"]["program"] = {"id": "op_then_add_self_v1", "ops": ["matmul", "add_self"]}
            benchmark_path.write_text(json.dumps(benchmark))
            with self.assertRaisesRegex(ValueError, "peer target or input contract differs"):
                evidence.import_benchmark(benchmark_path, operator_outcomes)

    def test_historical_report_remains_diagnostic(self):
        with self.assertRaisesRegex(ValueError, "diagnostic only"):
            evidence.import_benchmark(ROOT / "test/benchmark/results/model-gpt2-metal-m4-2026-10-18.json", {})


if __name__ == "__main__":
    unittest.main()
