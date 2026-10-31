import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("dispatch_boundaries", ROOT / "tools/audit-dispatch-boundaries.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class DispatchBoundaryTest(unittest.TestCase):
    def test_source_drift_requires_review(self):
        manifest = json.loads((ROOT / "test/benchmark/dispatch-boundaries.json").read_text())
        manifest["boundaries"][0]["source_assertion"] = "condition that is absent"
        with self.assertRaisesRegex(ValueError, "dispatch source changed"):
            audit.audit(manifest, ROOT / "zig-out/bin/affon", lambda binary, case: {})

    def test_expected_path_transition(self):
        manifest = json.loads((ROOT / "test/benchmark/dispatch-boundaries.json").read_text())
        manifest["boundaries"] = manifest["boundaries"][:1]
        def result(binary, case):
            path = "metal.matmul.simd_gemv" if case["shape"][1] == 64 else "metal.matmul.mps_single"
            return {"status": "correct", "selected_path": [{"implementation_id": path}]}
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / "affon"
            binary.write_bytes(b"test binary")
            report = audit.audit(manifest, binary, result)
        self.assertEqual(report["summary"]["expected_transition"], 1)

    def test_masked_transition_is_reported_as_open_coverage(self):
        manifest = json.loads((ROOT / "test/benchmark/dispatch-boundaries.json").read_text())
        manifest["boundaries"] = manifest["boundaries"][4:5]
        def result(binary, case):
            return {"status": "correct", "selected_path": [{"implementation_id": "metal.matmul.tiled"}]}
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / "affon"
            binary.write_bytes(b"test binary")
            report = audit.audit(manifest, binary, result)
        self.assertEqual(report["summary"]["unobserved_transition"], 1)
        self.assertEqual(report["summary"]["expected_transition"], 0)


if __name__ == "__main__":
    unittest.main()
