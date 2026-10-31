import importlib.util
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("capability_probes", ROOT / "tools/probe-capabilities.py")
probes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probes)


class CapabilityProbeTest(unittest.TestCase):
    def test_cuda_batch_catalog_matches_retained_probe(self):
        path = ROOT / "test/parity/catalogs/cuda-first-256.json"
        batch = json.loads(path.read_text())
        full = json.loads((ROOT / "test/benchmarks/capability-probes.json").read_text())
        report = json.loads((ROOT / "test/benchmarks/results/coverage-capability-cuda-first-2026-10-31.json").read_text())
        self.assertEqual(len(batch["cases"]), 256)
        self.assertEqual([case["id"] for case in batch["cases"]],
                         [case["id"] for case in report["cases"]])
        self.assertTrue(all(case["id"].startswith("cuda/") for case in batch["cases"]))
        self.assertTrue(all(case in full["cases"] for case in batch["cases"]))
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), report["catalog_sha256"])

    def test_catalog_cells_are_unique_and_reference_formula_is_independent(self):
        catalog = json.loads((ROOT / "test/benchmarks/capability-probes.json").read_text())
        cases = probes.validate_catalog(catalog, {case["id"] for case in catalog["cases"]})
        self.assertGreaterEqual(len(cases), 8227)
        dot = next(case for case in cases if case["id"] == "metal/f32/dot/dot_or_vector/n2")
        self.assertEqual(probes.expected(dot), [1.203125])
        self.assertEqual(probes.expected_shape(dot), [])
        strided_dot = next(case for case in cases if case["id"] == "metal/f32/dot/strided/n2")
        offset_dot = next(case for case in cases if case["id"] == "metal/f32/dot/dot_or_vector/offset_n2")
        self.assertEqual(probes.expected(strided_dot), [1.4375])
        self.assertEqual(probes.expected(offset_dot), [1.640625])
        one_hot = {"id": "metal/f32/one_hot/small", "op": "one_hot", "dtype": "f32",
                   "shape": [1, 2], "classes": 3}
        self.assertEqual(probes.expected(one_hot), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertEqual(probes.expected_shape(one_hot), [1, 2, 3])
        strided_one_hot = {**one_hot, "id": "metal/f32/one_hot/strided"}
        offset_one_hot = {**one_hot, "id": "metal/f32/one_hot/small/offset_b1_n2_c3"}
        self.assertEqual(probes.expected(strided_one_hot), [1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(probes.expected(offset_one_hot), [0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        invalid_one_hot = {**one_hot, "id": "metal/f32/one_hot/invalid_index", "shape": [1, 3], "classes": 2}
        self.assertEqual(probes.expected(invalid_one_hot), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        batched_one_hot = {**one_hot, "id": "metal/f32/one_hot/batched", "shape": [2, 1, 1], "classes": 7}
        encoded = probes.expected(batched_one_hot)
        self.assertEqual(probes.expected_shape(batched_one_hot), [2, 1, 1, 7])
        self.assertEqual([index for index, value in enumerate(encoded) if value == 1.0], [0, 12])
        add = next(case for case in cases if case["id"] == "cpu/f32/add/small_dense")
        self.assertEqual(probes.expected(add)[0], 1.5)
        self.assertEqual(probes.expected(add)[1], 1.75)
        gather = next(case for case in cases if case['id'] == 'metal/f32/gather/small')
        self.assertEqual(probes.expected_shape(gather), [4, 8])
        self.assertEqual(probes.expected(gather)[:2], [1, 1.375])
        scalar = next(case for case in cases if case['id'] == 'metal/f32/div/scalar')
        self.assertEqual(probes.expected(scalar), [2.0])
        self.assertIsNone(probes.expected_shape(scalar))
        arg = next(case for case in cases if case['id'] == 'metal/f32/argmax_axis/outer_axis')
        self.assertEqual(probes.expected_shape(arg), [256])
        self.assertEqual(probes.expected(arg)[:3], [127, 127, 127])
        masked = next(case for case in cases if case['id'] == 'metal/f32/masked_fill/scalar')
        self.assertEqual(probes.expected(masked), [-2])
        loss = next(case for case in cases if case['id'] == 'metal/f32/cross_entropy_indexed/small')
        self.assertEqual(probes.expected_shape(loss), [1])
        self.assertGreater(probes.expected(loss)[0], 0)
        axis_one = next(case for case in cases if case['id'] == 'metal/f32/index_select/small/axis1_len8')
        self.assertEqual(probes.expected_shape(axis_one), [16, 8])
        self.assertEqual(probes.expected(axis_one)[:2], [1.125, 1.5])
        equal = next(case for case in cases if case['id'] == 'metal/f32/eq/small_dense')
        self.assertEqual(probes.expected(equal)[:3], [0, 1, 0])
        cast_case = next(case for case in cases if case['id'] == 'metal/f32/cast/scalar')
        self.assertEqual(probes.expected(cast_case), [1.1015625])
        bad = {**catalog, "cases": [cases[0], cases[0]]}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            probes.validate_catalog(bad, {cases[0]["id"]})


if __name__ == "__main__":
    unittest.main()
