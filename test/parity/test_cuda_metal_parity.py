import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('cuda_metal_parity', ROOT / 'test/parity/catalogs/generators/generate-cuda-metal-parity.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class CudaMetalParityTests(unittest.TestCase):
    def test_mirror_preserves_contract_and_requires_supported_source(self):
        case = {'id': 'metal/bf16/topk/strided/axis0', 'op': 'topk',
                'dtype': 'bf16', 'shape': [7, 33], 'axis': 0, 'k': 3,
                'largest': False, 'dense_inputs': False}
        unsupported = {**case, 'id': 'metal/bf16/topk/small'}
        ledger = {case['id']: {'support': 'supported'},
                  unsupported['id']: {'support': 'unsupported'},
                  'cuda/bf16/add/small': {'support': 'supported'}}
        mirrored = generator.mirror({'cases': [case, unsupported]}, ledger)
        self.assertEqual(mirrored, [{**case, 'id': 'cuda/bf16/topk/strided/axis0'}])
        self.assertEqual(case['id'], 'metal/bf16/topk/strided/axis0')
        with self.assertRaisesRegex(ValueError, 'missing from catalog'):
            generator.mirror({'cases': []}, ledger)

    def test_smoke_covers_each_dtype_operation_and_representative_views(self):
        cases = [{'id': f'cuda/{dtype}/{op}/{layout}', 'dtype': dtype, 'op': op,
                  'shape': [3, 7]} for dtype in ('f32', 'bf16')
                 for op in ('add', 'mul') for layout in
                 ('dense', 'strided', 'offset', 'transposed', 'batched', 'outer_axis')]
        selected = generator.smoke(cases)
        self.assertEqual({(case['dtype'], case['op']) for case in selected},
                         {(case['dtype'], case['op']) for case in cases})
        self.assertEqual(len(selected), len({case['id'] for case in selected}))
        for layout in ('strided', 'offset', 'transposed', 'batched', 'outer_axis'):
            for dtype in ('f32', 'bf16'):
                self.assertTrue(any(case['dtype'] == dtype and layout in case['id']
                                    for case in selected))


    def test_plan_registration_is_idempotent_and_preserves_existing_contracts(self):
        source = {'id': 'metal/f32/add/strided/custom', 'op': 'add', 'dtype': 'f32', 'shape': [5, 19]}
        target = {**source, 'id': 'cuda/f32/add/strided/custom'}
        catalog = {'cases': [source]}
        policy = {'variants': {source['id']: {'dense_inputs': False}}}
        updated, planned = generator.update_plan(catalog, policy, [target])
        self.assertEqual(updated['cases'], [source, target])
        self.assertEqual(planned['variants'][target['id']], {'dense_inputs': False})
        self.assertEqual(generator.update_plan(updated, planned, [target]), (updated, planned))
        self.assertEqual(catalog, {'cases': [source]})
        self.assertNotIn(target['id'], policy['variants'])
        with self.assertRaisesRegex(ValueError, 'contract conflict'):
            generator.update_plan(updated, planned, [{**target, 'shape': [6, 19]}])
        with self.assertRaisesRegex(ValueError, 'variant conflict'):
            generator.update_plan(updated, {'variants': {source['id']: {'dense_inputs': False}, target['id']: {'dense_inputs': True}}}, [target])


if __name__ == '__main__':
    unittest.main()
