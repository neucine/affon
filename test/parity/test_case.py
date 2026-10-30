import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('parity_case', ROOT / 'test/parity/case.py')
case_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(case_module)


class CaseTest(unittest.TestCase):
    def test_axes_are_explicit_and_status_is_separate(self):
        case = {'id': 'metal/bf16/topk/small/offset_axis0_r3_c3_k2',
                'op': 'topk', 'dtype': 'bf16', 'shape': [3, 3], 'k': 2, 'axis': 0}
        axes = case_module.normalize(case, case_module.families())
        self.assertEqual((axes['device'], axes['dtype'], axes['operation_family'], axes['operation']),
                         ('metal', 'bf16', 'indexing', 'topk'))
        self.assertEqual((axes['shape'], axes['layout'], axes['regime'], axes['variant']),
                         ([3, 3], 'offset', 'small', 'offset_axis0_r3_c3_k2'))
        self.assertEqual(axes['scope'], 'operation')
        self.assertNotIn('status', axes)
        self.assertEqual(case_module.families()['add'], 'arithmetic')

    def test_id_contract_and_piece_scope_are_checked(self):
        case = {'id': 'metal/f32/add/small_dense', 'op': 'mul', 'dtype': 'f32', 'shape': [2, 4]}
        with self.assertRaisesRegex(ValueError, 'disagree'):
            case_module.normalize(case, case_module.families())
        case['op'] = 'add'
        case['scope'] = 'program'
        with self.assertRaisesRegex(ValueError, 'identity missing'):
            case_module.normalize(case, case_module.families())
        case['piece'] = 'add_then_relu_v1'
        self.assertEqual(case_module.normalize(case, case_module.families())['piece'], 'add_then_relu_v1')

    def test_a_named_case_does_not_need_the_coverage_plan(self):
        case = {'id': 'metal/f32/add/small_dense/new_shape', 'op': 'add',
                'dtype': 'f32', 'shape': [3, 5]}
        self.assertEqual(case_module.normalize(case)['operation_family'], 'arithmetic')
        self.assertEqual(case_module.contract_hash(case), case_module.contract_hash(
            {**case, 'family': 'arithmetic', 'layout': 'dense', 'scope': 'operation'}))
        self.assertNotEqual(case_module.contract_hash(case), case_module.contract_hash(
            {**case, 'shape': [3, 6]}))


if __name__ == '__main__':
    unittest.main()
