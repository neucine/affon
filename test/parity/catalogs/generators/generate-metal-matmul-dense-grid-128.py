#!/usr/bin/env python3
"""Maintain a 128-cell Metal f32/BF16 dense matmul shape grid."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    master_path = ROOT / 'test/benchmarks/capability-probes.json'
    policy_path = ROOT / 'test/benchmarks/coverage-policy.json'
    master = json.loads(master_path.read_text())
    policy = json.loads(policy_path.read_text())
    master['cases'] = [case for case in master['cases'] if '/matmul/dense/grid_' not in case['id']]
    policy['variants'] = {key: value for key, value in policy['variants'].items()
                          if '/matmul/dense/grid_' not in key}
    known = {case['id'] for case in master['cases']}
    cases = []
    for dtype in ('f32', 'bf16'):
        for rows in (2, 8, 32, 128):
            for inner in (5, 33, 65, 127):
                for cols in (7, 31, 63, 129):
                    regime = 'small_batch' if rows <= 8 else 'large_matrix'
                    cell = f'metal/{dtype}/matmul/{regime}/grid_m{rows}_k{inner}_n{cols}'
                    case = {'id': cell, 'op': 'matmul', 'dtype': dtype,
                            'shape': [rows, inner, cols]}
                    cases.append(case)
                    if cell not in known:
                        master['cases'].append(case)
                    policy['variants'][cell] = {'dense_inputs': False}
    assert len(cases) == 128
    write(master_path, master)
    write(policy_path, policy)
    write(Path(__file__).resolve().parent.parent / 'metal-matmul-dense-grid-128.json',
          {'schema_version': 1, 'suite': 'affon_capability_probes', 'cases': cases})
    print(f'{len(cases)} cells, {len(master["cases"])} master cases')


if __name__ == '__main__':
    main()
