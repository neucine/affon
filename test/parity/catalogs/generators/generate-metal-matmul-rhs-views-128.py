#!/usr/bin/env python3
"""Maintain Metal matmul transposed and strided RHS parity cells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    master_path = ROOT / 'test/benchmark/capability-probes.json'
    policy_path = ROOT / 'test/benchmark/coverage-policy.json'
    master = json.loads(master_path.read_text())
    policy = json.loads(policy_path.read_text())
    known = {case['id'] for case in master['cases']}
    cases = []
    for dtype in ('f32', 'bf16'):
        for layout in ('transposed', 'strided'):
            for rows in (1, 4, 16, 64):
                for inner in (7, 31, 64, 129):
                    for cols in (17, 65):
                        cell = f'metal/{dtype}/matmul/{layout}/rhs_m{rows}_k{inner}_n{cols}'
                        case = dict(id=cell, op='matmul', dtype=dtype, shape=[rows, inner, cols])
                        cases.append(case)
                        if cell not in known:
                            master['cases'].append(case)
                        policy['variants'][cell] = {'dense_inputs': False}
    write(master_path, master)
    write(policy_path, policy)
    write(Path(__file__).resolve().parent.parent / 'metal-matmul-rhs-views-128.json',
          dict(schema_version=1, suite='affon_capability_probes', cases=cases))
    print(f'{len(cases)} cells, {len(master["cases"])} master cases')


if __name__ == '__main__':
    main()
