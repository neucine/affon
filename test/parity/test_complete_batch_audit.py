import contextlib
import copy
import io
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import audit_complete_batch as complete

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'test/benchmark/results/parity-cuda-sweep-harness-2026-11-05'


class CompleteBatchAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.batch = Path(self.temporary.name) / 'batch'
        shutil.copytree(FIXTURE, self.batch)

    def audit(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return complete.audit(self.batch, **kwargs)

    def edit(self, relative, update):
        path = self.batch / relative
        value = json.loads(path.read_text())
        update(value)
        path.write_text(json.dumps(value) + '\n')

    def peer(self, kind='operator_peer'):
        return next((self.batch / kind).glob('*.json')).relative_to(self.batch)

    def seed(self):
        source = Path(self.temporary.name) / 'source'
        shutil.copytree(self.batch, source)
        identity = json.loads((self.batch / 'run.json').read_text())
        relative = self.peer()
        cell = json.loads((self.batch / relative).read_text())['cell_id']
        provenance = {'suite': 'affon_reviewed_sweep_seed', 'source': str(source),
                      'source_run_identity': identity, 'current_run_identity': identity,
                      'invalidated_operations': [],
                      'numerical_source': {'path': str(source/'capability.json'), 'sha256': hashlib.sha256((source/'capability.json').read_bytes()).hexdigest()},
                      'seeded_peer_reports': [{'id': cell, 'kind': 'operator_peer', 'source': str(source/relative), 'sha256': hashlib.sha256((source/relative).read_bytes()).hexdigest()}]}
        (self.batch/'seeded-evidence.json').write_text(json.dumps(provenance))
        return source, relative, cell

    def test_seed_provenance_verifies_retained_sources_and_relocated_sources(self):
        source, _, _ = self.seed()
        self.assertEqual(self.audit()['seeded_reports_provenance_verified'], 1)
        relocated = source.with_name('relocated')
        source.rename(relocated)
        self.assertEqual(self.audit(seed_source=relocated)['seeded_reports_provenance_verified'], 1)

    def test_seed_changed_source_or_copy_cannot_pass(self):
        source, relative, _ = self.seed()
        original = (source/relative).read_bytes()
        (source/relative).write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError, 'seed source hash mismatch'):
            self.audit()
        (source/relative).write_bytes(original)
        (self.batch/relative).write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError, 'seed copy hash mismatch'):
            self.audit()

    def test_seed_invalidated_operation_and_changed_identity_are_rejected(self):
        _, _, cell = self.seed()
        self.edit('seeded-evidence.json', lambda d: d.update(invalidated_operations=[cell.split('/')[2]]))
        with self.assertRaisesRegex(ValueError, 'invalidated operation reused'):
            self.audit()
        self.edit('seeded-evidence.json', lambda d: d['current_run_identity'].update(binary_sha256='changed'))
        with self.assertRaisesRegex(ValueError, 'seed current identity mismatch'):
            self.audit()

    def test_seed_cannot_redirect_source_or_duplicate_report(self):
        _, _, _ = self.seed()
        original = json.loads((self.batch/'seeded-evidence.json').read_text())
        self.edit('seeded-evidence.json', lambda d: d['seeded_peer_reports'][0].update(source='/unrelated/report.json'))
        with self.assertRaisesRegex(ValueError, 'unexpected seed source path'):
            self.audit()
        (self.batch/'seeded-evidence.json').write_text(json.dumps(original))
        self.edit('seeded-evidence.json', lambda d: d['seeded_peer_reports'].append(d['seeded_peer_reports'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate seeded report'):
            self.audit()

    def test_cuda_peer_version_is_bound_explicitly_or_consistently_inferred(self):
        report = self.audit(expected_catalog=self.batch/'catalog.json', peer_version='2.11.0+cu128')
        self.assertEqual(report['cells'], 2)
        self.assertTrue(report['expected_scope_bound'])
        self.assertEqual(self.audit()['peer_version'], '2.11.0+cu128')
        self.edit(self.peer('program_peer'), lambda d: d.update(peer_version='2.10.0'))
        with self.assertRaisesRegex(ValueError, 'peer version mismatch'):
            self.audit()

    def test_small_green_batch_cannot_satisfy_full_expected_scope(self):
        with self.assertRaisesRegex(ValueError, 'expected catalog scope'):
            self.audit(expected_catalog=ROOT/'test/parity/catalogs/cuda-metal-parity.json')

    def test_same_ids_with_changed_shapes_or_scopes_are_rejected(self):
        declared = json.loads((self.batch/'catalog.json').read_text())
        for key, value in (('shape', [7, 19]), ('scope', 'program')):
            changed = copy.deepcopy(declared)
            changed['cases'][0][key] = value
            if key == 'scope': changed['cases'][0]['piece'] = 'different_graph'
            expected = self.batch/'expected.json'
            expected.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'expected case contract mismatch'):
                self.audit(expected_catalog=expected)

    def test_wrong_probe_catalog_and_current_binary_are_rejected(self):
        self.edit('capability.json', lambda d: d.update(catalog_sha256='wrong'))
        with self.assertRaisesRegex(ValueError, 'probe catalog hash mismatch'):
            self.audit()
        binary = self.batch/'wrong-binary'
        binary.write_bytes(b'changed build')
        with self.assertRaisesRegex(ValueError, 'expected binary hash mismatch'):
            self.audit(binary=binary)

    def test_pass_boolean_cannot_override_raw_numerical_failure(self):
        self.edit('capability.json', lambda d: d['cases'][0].update(checksum_error=d['cases'][0]['checksum_tolerance']+1))
        with self.assertRaisesRegex(ValueError, 'error exceeds tolerance'):
            self.audit()
        result = subprocess.run([sys.executable, '-O', str(ROOT/'test/parity/audit_complete_batch.py'), str(self.batch)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('error exceeds tolerance', result.stderr)

    def test_raw_round_sample_and_flattened_sample_disagreements_are_rejected(self):
        self.edit(self.peer(), lambda d: d['affon_samples_ms'].__setitem__(0, d['affon_samples_ms'][0]+1))
        with self.assertRaisesRegex(ValueError, 'flattened samples differ'):
            self.audit()

    def test_nonfinite_round_and_green_but_excessive_error_are_rejected(self):
        path = self.peer()
        original = json.loads((self.batch/path).read_text())
        self.edit(path, lambda d: d['rounds'][0]['peer_samples_ms'].__setitem__(0, float('nan')))
        with self.assertRaisesRegex(ValueError, 'invalid samples'):
            self.audit()
        (self.batch/path).write_text(json.dumps(original))
        self.edit(path, lambda d: d['correctness'].update(max_abs_error=d['correctness']['tolerance']+1))
        with self.assertRaisesRegex(ValueError, 'error exceeds tolerance'):
            self.audit()


if __name__ == '__main__':
    unittest.main()
