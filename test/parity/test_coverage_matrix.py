import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("coverage_matrix", ROOT / "tools/coverage-matrix.py")
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


class CoverageMatrixTest(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT / "test/benchmarks/coverage-policy.json").read_text())
        self.diagnostics = json.loads((ROOT / "test/benchmarks/workload-coverage.json").read_text())

    def test_canonical_vocabulary_is_exhaustively_expanded(self):
        report = matrix.build(self.policy, self.diagnostics)
        self.assertEqual(report["summary"]["operations"], len(self.policy["op_regime"]))
        self.assertEqual(report["summary"]["cells"], len(report["cells"]))
        self.assertEqual(len({row["id"] for row in report["cells"]}), len(report["cells"]))
        self.assertTrue(all(row["support"] == "unknown" for row in report["cells"]))
        add = next(row for row in report['cells'] if row['id'] == 'metal/f32/add/small_dense')
        self.assertEqual((add['device'], add['dtype'], add['operation'], add['operation_family']),
                         ('metal', 'f32', 'add', 'arithmetic'))
        self.assertEqual((add['regime'], add['scope'], add['support']),
                         ('small_dense', 'operation', 'unknown'))
        for dimension in ("backend", "dtype", "family", "backend × dtype"):
            self.assertEqual(sum(group["candidate"] for group in report["summary"]["distribution"][dimension].values()),
                             len(report["cells"]))

    def test_new_operation_requires_policy_review(self):
        self.policy["op_regime"].pop("matmul")
        with self.assertRaisesRegex(ValueError, "operation policy drift"):
            matrix.build(self.policy, self.diagnostics)

    def test_declared_dispatch_variant_has_independent_cell(self):
        rows = {row["id"]: row for row in matrix.build(self.policy, self.diagnostics)["cells"]}
        base = "metal/f32/matmul/decode_m1"
        variant = base + "/dense_gemv"
        self.assertIn(base, rows)
        self.assertEqual(rows[variant]["variant"], {"dense_inputs": True})
        self.assertEqual(rows[variant]["evidence"]["operator_peer"], "missing")

    def test_exclusion_requires_reason_and_evidence_requires_contract(self):
        cell = "metal/f32/matmul/decode_m1"
        with self.assertRaisesRegex(ValueError, "needs reason"):
            matrix.build(self.policy, self.diagnostics, {cell: {"support": "not_applicable"}})
        with self.assertRaisesRegex(ValueError, "artifact SHA-256 and contract SHA-256"):
            matrix.build(self.policy, self.diagnostics, {cell: {"support": "supported", "evidence": {"correctness": {"artifact": "run.json"}}}})
        row = next(row for row in matrix.build(self.policy, self.diagnostics, {cell: {"support": "not_applicable", "reason": "documented contract exclusion"}})["cells"] if row["id"] == cell)
        self.assertEqual(row["support"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
