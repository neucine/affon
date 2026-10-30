import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[2] / "tools/workload-coverage.py"
spec = importlib.util.spec_from_file_location("workload_coverage", path)
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


class WorkloadCoverageTest(unittest.TestCase):
    def test_inventory_exposes_missing_and_broken_links(self):
        manifest = {"schema_version": 1, "suite": "affon_workload_coverage", "cells": [
            {"id": "cpu/f32/reduction/small", "family": "reduction", "regime": "small",
             "required": ["operator", "program", "model"],
             "evidence": {"operator": "affon/tools/workload-coverage.py:inventory",
                          "program": "affon/tools/workload-coverage.py:no_such_case"}}]}
        row = coverage.inventory(manifest)["cells"][0]
        self.assertEqual(row["tiers"], {"operator": "diagnostic", "program": "unverified_case", "model": "missing"})
        self.assertEqual(row["missing"], ["program", "model"])

    def test_duplicate_cells_rejected(self):
        cell = {"id": "same", "family": "matmul", "regime": "small", "required": ["operator"], "evidence": {}}
        with self.assertRaisesRegex(ValueError, "duplicate cell"):
            coverage.inventory({"schema_version": 1, "suite": "affon_workload_coverage", "cells": [cell, cell]})


if __name__ == "__main__":
    unittest.main()
