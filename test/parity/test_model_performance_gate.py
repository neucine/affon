import importlib.util
from pathlib import Path
import unittest


TOOLS = Path(__file__).resolve().parents[2] / "tools"


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOOLS / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


gate = module("performance_gate", "compare-model-performance.py")
normalize = module("performance_normalize", "normalize-model-performance.py")
coverage = module("performance_coverage", "model-coverage.py")


class ModelPerformanceGateTest(unittest.TestCase):
    def report(self, engine="affon", values=(10, 11, 12)):
        return {"target": "machine-1", "engine": engine,
                "cases": [{"id": "model/infer", "correct": True,
                           "contract": {"scope": "synchronized", "shape": [1, 32]},
                           "samples_ms": list(values)}]}

    def test_peer_gap_and_self_regression_are_independent(self):
        current = self.report()
        old = self.report(values=(9.9, 10, 10.1))
        peer = self.report(engine="mlx", values=(4, 5, 6))
        self.assertTrue(gate.compare(current, old, kind="self", maximum_ratio=1.1, absolute_ms=0)["gate_pass"])
        self.assertFalse(gate.compare(current, peer, kind="peer", maximum_ratio=1.25, absolute_ms=0)["gate_pass"])

    def test_rejects_unmatched_work(self):
        current = self.report()
        peer = self.report(engine="mlx")
        peer["cases"][0]["contract"]["shape"] = [1, 16]
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            gate.compare(current, peer, kind="peer", maximum_ratio=2, absolute_ms=0)
        peer["cases"] = []
        with self.assertRaisesRegex(ValueError, "case matrix mismatch"):
            gate.compare(current, peer, kind="peer", maximum_ratio=2, absolute_ms=0)

    def test_vision_adapter_requires_parity(self):
        report = {"passed": False, "runs": {"affon": {}, "onnxruntime": {}}, "parity": []}
        # The adapter checks parity before reading timings.
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "parity"):
                normalize.vision(path, "machine-1")

    def test_hf_adapter_preserves_model_and_workload(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "run.json").write_text(json.dumps({"build_profile": "ReleaseFast", "runs": [
                {"report": "vit-metal.json"}]}))
            (root / "vit-metal.json").write_text(json.dumps({
                "format": "affon-inference-benchmark/v1", "family": "vit", "device": "metal",
                "model_id": "model-a", "revision": "revision-a", "workload": {"width": 320},
                "measured_runs": 3, "metrics": {"forward_with_readback_ms": {"samples": [3, 4, 5]}}}))
            report = normalize.hf(root, "machine-1")["affon"]
            self.assertEqual(report["cases"][0]["contract"]["revision"], "revision-a")
            self.assertEqual(report["cases"][0]["samples_ms"], [3, 4, 5])

    def test_coverage_reports_missing_and_measured_gap_separately(self):
        import json
        import tempfile
        registry = {"required_archetypes": {"mac": ["decoder", "audio"]}, "cases": [
            {"target_group": "mac", "archetype": "decoder", "suite_id": "model", "id": "model/infer", "peer": "mlx", "maximum_peer_ratio": 1.25}]}
        with tempfile.TemporaryDirectory() as folder:
            suite = Path(folder) / "model"
            suite.mkdir()
            inventory = coverage.evaluate(registry, folder, "mac")
            missing = inventory["cases"][0]
            self.assertEqual(missing["status"], "unmeasured")
            self.assertEqual(inventory["archetypes"][1]["status"], "missing_case")
            self.assertFalse(inventory["gate_pass"])
            (suite / "affon.json").write_text(json.dumps({"schema_version": 1, "suite": "affon_model_performance", **self.report()}))
            (suite / "mlx.json").write_text(json.dumps({"schema_version": 1, "suite": "affon_model_performance", **self.report(engine="mlx", values=(4, 5, 6))}))
            measured = coverage.evaluate(registry, folder, "mac")["cases"][0]
            self.assertEqual(measured["peer_status"], "gap")
            self.assertEqual(measured["self_status"], "unmeasured")
            self.assertEqual(measured["status"], "measured_failure")


if __name__ == "__main__":
    unittest.main()
