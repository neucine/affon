import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import reconcile

FIXTURE = Path(__file__).resolve().parents[1] / 'benchmark/results/parity-cuda-sweep-harness-2026-11-05'


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base, self.replacement, self.output = [self.root / name for name in ('base', 'replacement', 'output')]
        for destination in (self.base, self.replacement):
            shutil.copytree(FIXTURE, destination)

    def run_reconcile(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return reconcile.reconcile(self.base, self.replacement, self.output,
                                       self.base / 'catalog.json', 'Reviewed peer semantic correction',
                                       peer_version='2.11.0+cu128')

    def edit(self, path, update):
        value = json.loads(path.read_text())
        update(value)
        path.write_text(json.dumps(value))

    def test_preserves_failed_source_and_exact_scope_with_explicit_provenance(self):
        path = next((self.base / 'operator_peer').glob('*.json'))
        failure = self.base / 'operator_peer_failures' / path.name
        path.rename(failure)
        self.edit(failure, lambda d: d['correctness'].update(passed=False))
        before = {p: p.read_bytes() for root in (self.base, self.replacement) for p in root.rglob('*') if p.is_file()}
        summary = self.run_reconcile()
        self.assertEqual(summary['cells'], 2)
        self.assertTrue(summary['expected_scope_bound'])
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.assertFalse((self.output / 'run.json').exists())
        provenance = json.loads((self.output / 'reconciliation.json').read_text())
        self.assertEqual(len(provenance['replacements']), 4)
        self.assertTrue(any(str(failure.resolve()) == old['path'] for row in provenance['replacements'] for old in row['prior_reports']))
        self.assertEqual(list((self.output / 'operator_peer_failures').glob('*.json')), [])

    def test_refuses_live_base_before_creating_output(self):
        self.edit(self.base / 'progress.json', lambda d: d.update(requested_phase_finished=False))
        with self.assertRaisesRegex(ValueError, 'still in progress'):
            self.run_reconcile()
        self.assertFalse(self.output.exists())

    def test_refuses_changed_target_protocol_and_case_contract(self):
        original = (self.base / 'manifest.json').read_bytes()
        for key, value in (('target', 'another GPU'), ('iterations', 999)):
            self.edit(self.base / 'manifest.json', lambda d: d.update({key: value}))
            with self.assertRaisesRegex(ValueError, 'replacement changed ' + key):
                self.run_reconcile()
            self.assertFalse(self.output.exists())
            (self.base / 'manifest.json').write_bytes(original)
        self.edit(self.base / 'catalog.json', lambda d: d['cases'][0].update(shape=[17, 29]))
        with self.assertRaisesRegex(ValueError, 'replacement changed case contract'):
            self.run_reconcile()
        self.assertFalse(self.output.exists())

    def test_failed_replacement_cannot_override_base_failure(self):
        path = next((self.replacement / 'operator_peer').glob('*.json'))
        self.edit(path, lambda d: d['correctness'].update(passed=False))
        with self.assertRaises(ValueError):
            self.run_reconcile()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
