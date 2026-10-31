#!/usr/bin/env python3
"""Maintain rank-three Metal dense loss parity across batches, classes and spatial widths."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def main():
    master_path = ROOT / 'test/benchmark/capability-probes.json'
    policy_path = ROOT / 'test/benchmark/coverage-policy.json'
    master = json.loads(master_path.read_text())
    policy = json.loads(policy_path.read_text())
    known = {case['id'] for case in master['cases']}
    cases = []
    for dtype in ('f32', 'bf16'):
        for op in ('cross_entropy', 'log_softmax_nll'):
            shapes = [(3, 11, 7)] + [(batch, classes, spatial)
                                       for batch in (2, 4)
                                       for classes in (3, 7, 33, 65)
                                       for spatial in (2, 5, 17, 31)]
            for index, (batch, classes, spatial) in enumerate(shapes):
                base = f'metal/{dtype}/{op}/batched'
                cell = base if index == 0 else f'{base}/dense_b{batch}_c{classes}_s{spatial}'
                case = {'id': cell, 'op': op, 'dtype': dtype,
                        'shape': [batch, classes, spatial]}
                cases.append(case)
                if cell not in known:
                    master['cases'].append(case)
                    known.add(cell)
                if index:
                    policy['variants'][cell] = {'dense_inputs': False}
    assert len(cases) == 132
    master_path.write_text(json.dumps(master, indent=2) + '\n')
    policy_path.write_text(json.dumps(policy, indent=2) + '\n')
    output = Path(__file__).resolve().parent.parent / 'metal-batched-dense-loss-132.json'
    output.write_text(json.dumps({'schema_version': 1, 'suite': 'affon_capability_probes',
                                  'cases': cases}, indent=2) + '\n')
    output.with_name('metal-batched-dense-loss-canonical-4.json').write_text(json.dumps(
        {'schema_version': 1, 'suite': 'affon_capability_probes',
         'cases': [case for case in cases if case['id'].count('/') == 3]}, indent=2) + '\n')
    print(f'{len(cases)} cells in {output}')


if __name__ == '__main__':
    main()
