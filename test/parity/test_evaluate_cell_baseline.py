import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("evaluate_cell_baseline", ROOT / "tools/evaluate-cell-baseline.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class BaselineEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.session = {"suite": "affon_cell_baseline_calibration", "cell_id": "metal/f32/sum_axis/long_axis",
            "target": "test-target", "binary_sha256": "binary", "harness_sha256": "harness",
            "contract": {"shape": [128, 768]}, "selected_path": [{"implementation_id": "sum"}],
            "started_at": "2026-10-29T10:00:00Z", "ended_at": "2026-10-29T10:10:00Z",
            "machine_state_at_start": {"power_source": "AC", "thermal_status": "normal"},
            "machine_state_at_end": {"power_source": "AC", "thermal_status": "normal"},
            "runs": [{"median_ms": value} for value in (0.30, 0.31, 0.32)]}

    def test_accepts_comparable_repeatable_sessions(self):
        second = copy.deepcopy(self.session)
        second["started_at"] = "2026-10-29T11:00:00Z"
        second["runs"] = [{"median_ms": value} for value in (0.31, 0.32, 0.33)]
        self.assertTrue(evaluator.evaluate([self.session, second])["accepted"])

    def test_rejects_drift_or_missing_conditions(self):
        second = copy.deepcopy(self.session)
        second["started_at"] = "2026-10-29T11:00:00Z"
        second["runs"] = [{"median_ms": value} for value in (0.60, 0.61, 0.62)]
        result = evaluator.evaluate([self.session, second])
        self.assertFalse(result["accepted"])
        self.assertIn("session medians exceed repeatability limit", result["reasons"])
        second["runs"] = self.session["runs"]
        del second["machine_state_at_start"]
        self.assertFalse(evaluator.evaluate([self.session, second])["accepted"])


if __name__ == "__main__":
    unittest.main()
