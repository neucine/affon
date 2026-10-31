#!/usr/bin/env python3
"""Maintain the Metal f32 reduce-to-shape targets and views catalog."""
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
    for layout in ('dense', 'offset'):
        for rows in (3, 16, 33, 128):
            for cols in (5, 32, 65, 129):
                for regime, target in (('short_axis', [1, cols]), ('long_axis', [rows, 1]),
                                       ('ragged', [1, 1]), ('outer_axis', [cols])):
                    label = 'dense' if layout == 'dense' else 'offset'
                    cell = f'metal/f32/reduce_to_shape/{regime}/{label}_r{rows}_c{cols}'
                    case = dict(id=cell, op='reduce_to_shape', dtype='f32', shape=[rows, cols],
                                target_shape=target)
                    cases.append(case)
                    if cell not in known:
                        master['cases'].append(case)
                    policy['variants'][cell] = {'dense_inputs': False}
    write(master_path, master)
    write(policy_path, policy)
    write(Path(__file__).resolve().parent.parent / 'metal-reduce-to-shape-128.json',
          dict(schema_version=1, suite='affon_capability_probes', cases=cases))
    print(f'{len(cases)} cells, {len(master["cases"])} master cases')


if __name__ == '__main__':
    main()
