#!/usr/bin/env python3
"""Maintain Metal matmul RHS layout and tile-boundary parity cells."""
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
    known = {case['id'] for case in master['cases']}
    cases = []
    for dtype in ('f32', 'bf16'):
        for layout in ('transposed', 'strided'):
            for rows in (1, 7, 33, 65):
                for inner in (8, 32, 63, 128):
                    for cols in (8, 32, 64, 128):
                        cell = f'metal/{dtype}/matmul/{layout}/rhs_m{rows}_k{inner}_n{cols}'
                        case = dict(id=cell, op='matmul', dtype=dtype, shape=[rows, inner, cols])
                        cases.append(case)
                        if cell not in known:
                            master['cases'].append(case)
                        policy['variants'][cell] = {'dense_inputs': False}
    write(master_path, master)
    write(policy_path, policy)
    write(Path(__file__).resolve().parent.parent / 'metal-matmul-rhs-boundaries-256.json',
          dict(schema_version=1, suite='affon_capability_probes', cases=cases))
    print(f'{len(cases)} cells, {len(master["cases"])} master cases')


if __name__ == '__main__':
    main()
