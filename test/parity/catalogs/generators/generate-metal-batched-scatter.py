#!/usr/bin/env python3
"""Maintain rank-three Metal scatter-add parity over both inner axes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def main():
    master_path = ROOT / 'test/benchmarks/capability-probes.json'
    policy_path = ROOT / 'test/benchmarks/coverage-policy.json'
    master = json.loads(master_path.read_text())
    policy = json.loads(policy_path.read_text())
    known = {case['id'] for case in master['cases']}
    cases = []
    for dtype in ('f32', 'bf16'):
        for axis in (1, 2):
            shapes = [(3, 7, 11)] if axis == 1 else []
            shapes += [(batch, rows, cols) for batch in (2, 4)
                       for rows in (3, 7, 16, 33) for cols in (5, 17, 65, 129)]
            for index, (batch, rows, cols) in enumerate(shapes):
                base = f'metal/{dtype}/scatter_add/batched'
                count = max(1, (rows if axis == 1 else cols) // 2)
                cell = base if axis == 1 and index == 0 else f'{base}/dense_axis{axis}_b{batch}_r{rows}_c{cols}_n{count}'
                index_shape = [batch, count if axis == 1 else rows,
                               count if axis == 2 else cols]
                case = {'id': cell, 'op': 'scatter_add', 'dtype': dtype,
                        'shape': [batch, rows, cols], 'index_shape': index_shape,
                        'index_axis': axis}
                cases.append(case)
                if cell not in known:
                    master['cases'].append(case)
                    known.add(cell)
                if cell != base:
                    policy['variants'][cell] = {'dense_inputs': False}
    assert len(cases) == 130
    master_path.write_text(json.dumps(master, indent=2) + '\n')
    policy_path.write_text(json.dumps(policy, indent=2) + '\n')
    output = Path(__file__).resolve().parent.parent / 'metal-batched-scatter-130.json'
    output.write_text(json.dumps({'schema_version': 1, 'suite': 'affon_capability_probes',
                                  'cases': cases}, indent=2) + '\n')
    output.with_name('metal-batched-scatter-canonical-2.json').write_text(json.dumps(
        {'schema_version': 1, 'suite': 'affon_capability_probes',
         'cases': [case for case in cases if case['id'].count('/') == 3]}, indent=2) + '\n')
    print(f'{len(cases)} cells in {output}')


if __name__ == '__main__':
    main()
