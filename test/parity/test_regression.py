import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('parity_regression', ROOT / 'test/parity/regression.py')
regression = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regression)


class RegressionTest(unittest.TestCase):
    def manifest(self, cells):
        return {'target': 'same-device', 'iterations': 5, 'samples': 5, 'rounds': 3, 'cells': cells}

    def cell(self, cell='metal/f32/add/small_dense', probe='correct', operator='passed', program='passed', ms=1):
        return {'id': cell, 'probe_status': probe, 'operator_peer': operator,
                'program_peer': program, 'operator_peer_affon_ms': ms,
                'operator_peer_ratio': ms, 'program_peer_affon_ms': ms,
                'program_peer_ratio': ms}

    def test_new_numeric_failure_is_a_regression_and_timing_is_diagnostic(self):
        previous = self.manifest([self.cell()])
        current = self.manifest([self.cell(probe='incorrect', operator='probe_not_correct',
                                           program='probe_not_correct')])
        result = regression.compare(previous, current)
        self.assertEqual(result['correctness_regressions'], 1)
        self.assertEqual(result['cells'][0]['correctness_regressions'],
                         ['probe_status', 'operator_peer', 'program_peer'])
        self.assertEqual(result['timing_suspects'], 0)
        slow = regression.compare(previous, self.manifest([self.cell(ms=2)]))
        self.assertEqual(slow['correctness_regressions'], 0)
        self.assertEqual(slow['timing_suspects'], 1)
        self.assertIn('Diagnostic timing suspects: 1', regression.render(slow))

    def test_comparison_reports_cell_changes_and_skips_unmatched_timing(self):
        previous = self.manifest([self.cell()])
        changed = regression.compare(previous, self.manifest([self.cell(cell='metal/f32/mul/small_dense')]))
        self.assertEqual(len(changed['cells']), 0)
        self.assertEqual(changed['added_cells'], ['metal/f32/mul/small_dense'])
        self.assertEqual(changed['removed_cells'], ['metal/f32/add/small_dense'])
        current = self.manifest([self.cell()])
        current['target'] = 'different-device'
        mismatch = regression.compare(previous, current)
        self.assertFalse(mismatch['target_matches'])
        self.assertEqual(mismatch['cells'][0]['timing'], {})
        current['target'] = 'same-device'
        current['samples'] = 10
        self.assertEqual(regression.compare(previous, current)['cells'][0]['timing'], {})

    def test_curated_catalog_has_declared_complete_cells(self):
        catalog = json.loads((ROOT / 'test/parity/catalogs/metal-regression-smoke.json').read_text())
        outcomes = json.loads((ROOT / 'benchmarks/results/coverage-outcomes-current-mac-2026-10-29.json').read_text())
        cells = catalog['cases']
        self.assertEqual(len(cells), 12)
        self.assertEqual(len({case['id'] for case in cells}), 12)
        for case in cells:
            self.assertTrue(case['id'].startswith('metal/'))
            self.assertTrue(all(kind in outcomes[case['id']]['evidence'] for kind in
                                ('correctness', 'operator_peer', 'program_peer')))

    def test_cli_retains_report_and_fails_on_new_correctness_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            previous, current = base / 'previous.json', base / 'current.json'
            previous.write_text(json.dumps(self.manifest([self.cell()])))
            current.write_text(json.dumps(self.manifest([self.cell(operator='incorrect')])))
            output = base / 'comparison'
            result = subprocess.run([sys.executable, str(ROOT / 'test/parity/regression.py'),
                                     str(previous), str(current), '--output-dir', str(output)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(json.loads((output / 'regression.json').read_text())['correctness_regressions'], 1)
            self.assertIn('operator_peer', (output / 'regression.md').read_text())


if __name__ == '__main__':
    unittest.main()
