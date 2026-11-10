import json
from pathlib import Path
import tempfile
import unittest

import sweep


class SweepTests(unittest.TestCase):
    def test_resume_rejects_changed_binary_reference_target_and_protocol(self):
        identity = {'binary': 'a', 'reference': 'r', 'target': 'gpu', 'samples': 5}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sweep.bind_identity(root, identity, False)
            sweep.bind_identity(root, identity, True)
            with self.assertRaises(FileExistsError):
                sweep.bind_identity(root, identity, False)
            for field in identity:
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    sweep.bind_identity(root, {**identity, field: 'changed'}, True)

    def test_interrupted_probe_resumes_only_unfinished_cells_and_retains_failures(self):
        cases = [{'id': f'cuda/f32/add/small/{index}'} for index in range(3)]
        calls = []
        def probe(binary, case):
            calls.append(case['id'])
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return {'id': case['id'], 'status': 'correct'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'numerical').mkdir()
            with self.assertRaises(KeyboardInterrupt):
                sweep.probe_cases(cases, None, root, probe, lambda *_: None)
            self.assertEqual(sweep.load_probe(root, cases[0]['id'])['status'], 'correct')
            def finish(binary, case):
                calls.append(case['id'])
                return {'id': case['id'], 'status': 'incorrect'}
            rows = sweep.probe_cases(cases, None, root, finish, lambda *_: None)
            self.assertEqual(calls.count(cases[0]['id']), 1)
            self.assertEqual([r['status'] for r in rows.values()], ['correct', 'incorrect', 'incorrect'])
            sweep.probe_cases(cases, None, root, lambda *_: self.fail('completed cell reran'), lambda *_: None)

    def test_probe_exception_is_failed_evidence_and_does_not_stop_other_cells(self):
        cases = [{'id': 'cuda/f32/add/a'}, {'id': 'cuda/f32/add/b'}]
        def probe(binary, case):
            if case['id'].endswith('/a'):
                raise TimeoutError('worker timed out')
            return {'id': case['id'], 'status': 'correct'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'numerical').mkdir()
            rows = sweep.probe_cases(cases, None, root, probe, lambda *_: None)
            self.assertEqual(rows[cases[0]['id']]['status'], 'probe_exception')
            self.assertEqual(rows[cases[1]['id']]['status'], 'correct')

    def test_partial_temporary_write_cannot_replace_completed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            sweep.atomic_json(path, {'status': 'correct'})
            path.with_suffix('.json.tmp').write_text('{')
            self.assertEqual(json.loads(path.read_text()), {'status': 'correct'})

    def test_peer_checkpoint_rejects_conflicting_pass_and_failure(self):
        cell = 'cuda/f32/add/small'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('operator_peer', 'operator_peer_failures'):
                (root / name).mkdir()
                (root / name / sweep.filename(cell)).write_text(json.dumps({'cell_id': cell, 'kind': 'operator_peer', 'correctness': {'passed': True}}))
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                sweep.peer_checkpoint(root, 'operator_peer', cell)


if __name__ == '__main__':
    unittest.main()
