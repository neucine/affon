import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cell_summary", ROOT / "tools/summarize-cell-performance.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class CellSummaryTest(unittest.TestCase):
    def report(self, ratios):
        return {"suite": "affon_cell_benchmark", "kind": "operator_peer", "cell_id": "cpu/f32/add/small_dense",
                "correctness": {"passed": True}, "rounds": [
                    {"affon_samples_ms": [ratio] * 3, "peer_samples_ms": [1] * 3} for ratio in ratios]}

    def test_consistent_and_mixed_rounds_stay_distinct(self):
        self.assertEqual(summary.summarize([self.report([1.4, 1.5, 1.6])], 1.25)["cases"][0]["status"], "candidate_gap")
        self.assertEqual(summary.summarize([self.report([1.4, 0.9, 1.1])], 1.25)["cases"][0]["status"], "mixed_rounds")

    def test_failed_correctness_rejected(self):
        report = self.report([1.4, 1.5])
        report["correctness"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "correctness failed"):
            summary.summarize([report], 1.25)

    def test_program_summary_has_its_own_kind(self):
        report = self.report([1.1, 1.2, 1.3])
        report["kind"] = "program_peer"
        self.assertEqual(summary.summarize([report], 1.25, "program_peer")["kind"], "program_peer")


if __name__ == "__main__":
    unittest.main()
