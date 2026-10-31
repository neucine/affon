#!/usr/bin/env python3
"""Maintain the Metal variance/std axis view parity catalog."""
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
        for op in ('variance_axis', 'std_axis'):
            for layout, regime in (('strided', 'strided'), ('offset', 'ragged')):
                for rows in (3, 16, 33, 128):
                    for cols in (5, 32, 65, 129):
                        cell = f'metal/{dtype}/{op}/{regime}/{layout}_r{rows}_c{cols}'
                        case = dict(id=cell, op=op, dtype=dtype, shape=[rows, cols])
                        cases.append(case)
                        if cell not in known:
                            master['cases'].append(case)
                        policy['variants'][cell] = {'dense_inputs': False}
    write(master_path, master)
    write(policy_path, policy)
    write(Path(__file__).resolve().parent.parent / 'metal-variance-views-128.json',
          dict(schema_version=1, suite='affon_capability_probes', cases=cases))
    print(f'{len(cases)} cells, {len(master["cases"])} master cases')


if __name__ == '__main__':
    main()
