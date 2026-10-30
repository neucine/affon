#!/usr/bin/env python3
"""Keep the previously unknown runnable Metal f32/BF16 contracts in one audit."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def main():
    master_path = ROOT / 'benchmarks/capability-probes.json'
    master = json.loads(master_path.read_text())
    known = {case['id']: case for case in master['cases']}
    ids = ['metal/bf16/contiguous/view_only', 'metal/f32/contiguous/view_only']
    for op in ('add', 'sub', 'mul', 'div', 'eq', 'lt', 'gt'):
        ids.extend((f'metal/f32/{op}/offset', f'metal/f32/{op}/offset/r32_c127'))
    for op in ('variance_all', 'std_all'):
        ids.extend((f'metal/f32/{op}/long_axis', f'metal/f32/{op}/strided/r64_c127'))
    cases = [known[cell] for cell in ids]
    cases.extend([
        {'id': 'metal/f32/topk/strided', 'op': 'topk', 'dtype': 'f32',
         'shape': [33, 65], 'axis': 1, 'k': 17},
        {'id': 'metal/bf16/topk/strided', 'op': 'topk', 'dtype': 'bf16',
         'shape': [33, 65], 'axis': 0, 'k': 16},
        {'id': 'metal/f32/quantized_matmul/groups_and_bits', 'op': 'quantized_matmul',
         'dtype': 'f32', 'shape': [2, 64, 33], 'bits': 2, 'group_size': 16},
        {'id': 'metal/bf16/quantized_matmul/groups_and_bits', 'op': 'quantized_matmul',
         'dtype': 'bf16', 'shape': [2, 64, 33], 'bits': 8, 'group_size': 32},
        {'id': 'metal/f32/reduce_to_shape/strided', 'op': 'reduce_to_shape',
         'dtype': 'f32', 'shape': [33, 65], 'target_shape': [1, 65]},
    ])
    assert len(cases) == 25
    for case in cases:
        if case['id'] not in known:
            master['cases'].append(case)
    master_path.write_text(json.dumps(master, indent=2) + '\n')
    output = Path(__file__).resolve().parent.parent / 'metal-remaining-runnable-audit-25.json'
    output.write_text(json.dumps({'schema_version': 1, 'suite': 'affon_capability_probes',
                                  'cases': cases}, indent=2) + '\n')
    print(f'{len(cases)} cells in {output}')


if __name__ == '__main__':
    main()
