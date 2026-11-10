import json
from pathlib import Path
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

import refresh


class RefreshTests(unittest.TestCase):
    def test_concurrent_refreshes_publish_complete_files(self):
        source = Path(__file__).resolve().parents[1] / 'benchmark/results/parity-cuda-sweep-harness-2026-11-05'
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory) / 'batch'
            shutil.copytree(source, batch)
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(refresh.refresh, [batch]*12))
            manifest = json.loads((batch/'manifest.json').read_text())
            self.assertEqual(manifest, results[-1])
            self.assertEqual((batch/'report.md').read_text(), refresh.report.render(manifest))
            self.assertEqual(list(batch.glob('.manifest.json-*')), [])

    def test_interrupted_sweep_is_reported_as_stopped(self):
        source = Path(__file__).resolve().parents[1] / 'benchmark/results/parity-cuda-sweep-harness-2026-11-05'
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory) / 'batch'
            shutil.copytree(source, batch)
            state = json.loads((batch / 'progress.json').read_text())
            state.update(requested_phase_finished=False, interrupted=True,
                         stop_reason='Peer context failed; evidence retained.')
            (batch / 'progress.json').write_text(json.dumps(state))
            result = refresh.refresh(batch)
            self.assertFalse(result['in_progress'])
            self.assertIn('Validation stopped: Peer context failed', (batch / 'report.md').read_text())

    def test_partial_checkpoint_updates_report_without_claiming_complete_parity(self):
        source = Path(__file__).resolve().parents[1] / 'benchmark/results/parity-cuda-sweep-harness-2026-11-05'
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory) / 'batch'
            shutil.copytree(source, batch)
            missing = next((batch / 'program_peer').glob('*.json'))
            missing.unlink()
            state = json.loads((batch / 'progress.json').read_text())
            state['requested_phase_finished'] = False
            (batch / 'progress.json').write_text(json.dumps(state))
            before = {p: p.read_bytes() for p in batch.rglob('*.json') if p.name != 'manifest.json'}
            result = refresh.refresh(batch)
            self.assertTrue(result['in_progress'])
            self.assertEqual(sum(r['operator_peer'] == 'passed' for r in result['cells']), 2)
            self.assertEqual(sum(r['program_peer'] == 'passed' for r in result['cells']), 1)
            self.assertEqual(sum(r['program_peer'] == 'not_run' for r in result['cells']), 1)
            text = (batch / 'report.md').read_text()
            self.assertIn('Program checks pending: 1', text)
            self.assertIn('Validation is in progress', text)
            self.assertEqual({p: p.read_bytes() for p in before}, before)


if __name__ == '__main__':
    unittest.main()
