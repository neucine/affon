import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import promote


class PromotionTests(unittest.TestCase):
    def fixture(self, root):
        batch = root / 'batch'
        batch.mkdir()
        ids = ['cuda/f32/add/old', 'cuda/f32/add/classified', 'cuda/f32/add/new']
        old = {ids[0]: {'status': 'supported', 'evidence': {'old': 'retained'}},
               ids[1]: {'status': 'unsupported', 'reason': 'previous gap'},
               'unrelated': {'status': 'unchanged'}}
        ledger = root / 'ledger.json'
        ledger.write_text(json.dumps(old))
        rows = []
        fresh = {}
        for cell in ids:
            row = {'id': cell, 'probe_status': 'correct',
                   'operator_peer': 'passed', 'program_peer': 'passed'}
            fresh[cell] = {'probe_contract': {'path': []}, 'evidence': {}}
            for kind in ('operator_peer', 'program_peer'):
                name = kind + cell.replace('/', '__') + '.json'
                (batch / name).write_text('new ' + name)
                row[kind + '_artifact'] = name
                fresh[cell]['evidence'][kind] = {}
            rows.append(row)
        (batch / 'manifest.json').write_text(json.dumps({'cells': rows}))
        aggregates = ('cell-peer-current-mac-2026-10-29', 'cell-program-peer-mac-2026-10-29')
        for aggregate in aggregates:
            destination = root / 'results' / aggregate / (ids[0].replace('/', '__') + '.json')
            destination.parent.mkdir(parents=True)
            destination.write_text('old report ' + aggregate)
        module = SimpleNamespace(import_probe=lambda *_: copy.deepcopy(fresh),
                                 import_benchmark=lambda *_: None)
        return batch, ledger, old, ids, module

    def test_mixed_promotion_preserves_prior_ledger_and_available_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch, ledger, old, ids, module = self.fixture(root)
            archive = root / 'archive'
            with patch.object(promote, 'RESULTS', root / 'results'), patch.object(promote, 'importer', return_value=module):
                promote.promote(batch, ledger, archive)
            self.assertEqual(json.loads((archive / 'ledger-before.json').read_text()),
                             {cell: old[cell] for cell in ids[:2]})
            reports = json.loads((archive / 'reports-before.json').read_text())
            self.assertEqual(sum(row['existed'] for row in reports), 2)
            self.assertEqual(len(reports), 6)
            for kind in ('operator_peer', 'program_peer'):
                saved = list((archive / kind).glob('*.json'))
                self.assertEqual(len(saved), 1)
                self.assertTrue(saved[0].read_text().startswith('old report '))
            updated = json.loads(ledger.read_text())
            self.assertEqual(updated['unrelated'], old['unrelated'])
            for cell in ids:
                for evidence in updated[cell]['evidence'].values():
                    self.assertTrue(Path(evidence['artifact']).read_text().startswith('new '))
                self.assertNotIn('probe_contract', updated[cell])

    def test_replacement_requires_unused_archive_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch, ledger, old, _, module = self.fixture(root)
            archive = root / 'archive'
            archive.mkdir()
            for candidate in (None, archive):
                with patch.object(promote, 'importer', return_value=module):
                    with self.assertRaises(ValueError):
                        promote.promote(batch, ledger, candidate)
                self.assertEqual(json.loads(ledger.read_text()), old)
                self.assertEqual(list(archive.iterdir()), [])

    def test_invalid_evidence_does_not_create_archive_or_replace_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch, ledger, old, _, module = self.fixture(root)
            def reject(*_):
                raise ValueError('invalid peer evidence')
            module.import_benchmark = reject
            archive = root / 'archive'
            with patch.object(promote, 'RESULTS', root / 'results'), patch.object(promote, 'importer', return_value=module):
                before = {p: p.read_bytes() for p in (root / 'results').rglob('*.json')}
                with self.assertRaisesRegex(ValueError, 'invalid peer'):
                    promote.promote(batch, ledger, archive)
            self.assertFalse(archive.exists())
            self.assertEqual(json.loads(ledger.read_text()), old)
            self.assertEqual({p: p.read_bytes() for p in before}, before)


if __name__ == '__main__':
    unittest.main()
